"""Colab-only A/B training on immutable V2 partitions. No split generation.

Validation, comparison and checkpoint-contract helpers can run without training.
run_architecture refuses Windows, non-Colab and non-CUDA execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score

VERSION = "c3_features_v2_20261022"
CLASSES = {"root": ["Acceptable", "Not Acceptable"], "leaf": ["Both Sides", "One Side"]}
PREPROCESSING = {"image_size": 224, "content_size": 208, "exif_transpose": True,
    "jpeg_draft_size": [1024, 1024], "resize": "aspect-preserving contain; centered padding",
    "fill_rgb": [127, 127, 127], "mean": [.485, .456, .406], "std": [.229, .224, .225],
    "train_augmentation": "original plus two fixed views: horizontal flip, +/-4 degrees, color jitter .08; no crops"}
CONFIG = {"seed": 42, "batch_size": 16, "max_epochs": 40, "patience": 6,
    "head_lr": .001, "layer4_lr": .00001, "weight_decay": .01, "min_delta": .00001,
    "selection": "minimum validation weighted cross-entropy; weights from training only",
    "threshold": "argmax; fixed", "batchnorm": "frozen running statistics in both architectures"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or ":" in relative or "\\" in relative or not path.is_relative_to(root.resolve()):
        raise ValueError("Unsafe dataset path")
    return path


def validate_partition(frame, split):
    fields = ["Plant_ID", "Sample_ID", "sample_key", "Image_ID", "sha256", "image_path", "label", "view_type", "status", "exclusion_reason"]
    if not set(fields).issubset(split.columns) or split.sample_key.duplicated().any():
        raise ValueError("Ambiguous/missing split fields")
    if set(split.split) != {"train", "validation", "test"} or set(split.sample_key) != set(frame.sample_key):
        raise ValueError("Partition coverage differs from manifest")
    left, right = frame.set_index("sample_key").sort_index(), split.set_index("sample_key").sort_index()
    for column in fields:
        if column != "sample_key" and not left[column].equals(right[column]):
            raise ValueError(f"Split/manifest mismatch: {column}")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        for column in ["sample_key", "sha256", "image_path"]:
            if set(split.loc[split.split.eq(a), column]) & set(split.loc[split.split.eq(b), column]):
                raise ValueError(f"Partition leakage: {column}")
    for _, group in split.groupby("split"):
        if group.label.nunique() != 2:
            raise ValueError("A global partition lacks a target class")


def validate_dataset(root):
    root = Path(root).resolve()
    checks = json.loads((root / "checksums.json").read_text())
    required = {"release.json", "source/quality_dataset_v2.xlsx"}
    required.update(f"{task}_usable.csv" for task in CLASSES)
    required.update(f"splits/{task}/fold_{fold}_split.csv" for task in CLASSES for fold in (1,2,3))
    if not required.issubset(checks):
        raise ValueError("Critical dataset files missing from checksum index")
    for relative, expected in checks.items():
        if digest(safe_path(root, relative)) != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    release = json.loads((root / "release.json").read_text())
    if release["dataset_version"] != VERSION:
        raise ValueError("Wrong dataset version")
    if digest(root / "source/quality_dataset_v2.xlsx") != release["workbook_sha256"]:
        raise ValueError("Workbook snapshot checksum mismatch")
    tables, partitions = {}, {}
    for task, classes in CLASSES.items():
        frame = pd.read_csv(root / f"{task}_usable.csv", keep_default_na=False)
        required = {"Plant_ID", "Sample_ID", "sample_key", "Image_ID", "sha256", "image_path", "label", "view_type", "status", "exclusion_reason"}
        if not required.issubset(frame.columns) or frame.empty or frame.sample_key.duplicated().any():
            raise ValueError("Invalid manifest")
        if not frame.status.eq("usable").all() or not frame.exclusion_reason.eq("").all() or not frame.label.isin(classes).all():
            raise ValueError("Unverified/unsupported manifest row")
        allowed = {f"P{i:02d}" for i in range(1 if task == "root" else 2, 9)}
        if not frame.Plant_ID.isin(allowed).all() or not frame.view_type.eq("Root" if task == "root" else "Front").all():
            raise ValueError("Wrong plant applicability/view")
        if not frame.sample_key.eq(frame.Plant_ID + ":" + frame.Sample_ID).all():
            raise ValueError("Composite identity mismatch")
        if len(frame) != release["tasks"][task]["usable_images"] or frame.label.value_counts().to_dict() != release["tasks"][task]["class_counts"]:
            raise ValueError("Release class support mismatch")
        for row in frame.itertuples():
            if checks.get(row.image_path) != row.sha256:
                raise ValueError("Image not in verified checksum index")
        test_keys, splits = [], []
        for fold in (1, 2, 3):
            path = root / f"splits/{task}/fold_{fold}_split.csv"
            split = pd.read_csv(path, keep_default_na=False)
            validate_partition(frame, split)
            test_keys.extend(split.loc[split.split.eq("test"), "sample_key"])
            splits.append(split)
        if len(test_keys) != len(set(test_keys)) or set(test_keys) != set(frame.sample_key):
            raise ValueError("OOF coverage differs from one test prediction per sample")
        tables[task], partitions[task] = frame, splits
    return release, tables, partitions


def require_colab_gpu():
    if platform.system() == "Windows":
        raise RuntimeError("Local Windows training is disabled; use hosted Colab T4")
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Training requires hosted Google Colab") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab CUDA GPU")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_model(model, architecture):
    if architecture not in {"A", "B"}:
        raise ValueError("Architecture must be A or B")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fc.") or (architecture == "B" and name.startswith("layer4.")))
    model.eval()
    if architecture == "B":
        model.layer4.train()
    model.fc.train()
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    return model


def class_weights(labels):
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=2).float()
    if counts.min() == 0:
        raise ValueError("Training lacks a class")
    return counts.sum() / (2 * counts)


def image_views(path, seed):
    with Image.open(path) as source:
        source.draft("RGB", (1024, 1024))
        image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (208, 208), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (224, 224), (127, 127, 127))
        canvas.paste(image, ((224-image.width)//2, (224-image.height)//2))
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])
    augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(4, fill=127),
                                  transforms.ColorJitter(brightness=.08, contrast=.08, saturation=.08)])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.stack([normalize(canvas)] + [normalize(augment(canvas)) for _ in range(2)])


def scores(truth, predicted, classes):
    report = classification_report(truth, predicted, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
    both = len(set(truth)) == 2
    return {"confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(), "matrix_order": classes,
        "macro_f1": float(f1_score(truth, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "macro_f1_definition": "fixed two-class macro; missing-class strata do not demonstrate two-class discrimination",
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)) if both else None,
        "minority_precision": report[classes[1]]["precision"] if sum(np.asarray(predicted) == 1) else None,
        "minority_recall": report[classes[1]]["recall"] if sum(np.asarray(truth) == 1) else None,
        "classification_report": report}


def checkpoint_payload(model, task, architecture, metadata):
    return {"model_state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "architecture": "resnet18", "experiment_architecture": architecture, "class_names": CLASSES[task],
        "class_to_idx": {name: index for index, name in enumerate(CLASSES[task])}, "task": task,
        "image_size": 224, "preprocessing": PREPROCESSING, "dataset_version": VERSION, **metadata}


def save_confusion(metrics, path, title):
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=metrics["matrix_order"], yticklabels=metrics["matrix_order"],
           xlabel="Predicted", ylabel="Expert", title=title)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_architecture(dataset_root, output_root, architecture):
    require_colab_gpu()  # before downloads, model construction or output writes
    if architecture not in {"A", "B"}:
        raise ValueError("Unknown architecture")
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output.is_relative_to(root) or "baseline_v1" in output.parts:
        raise ValueError("Output must be a separate Experiment 2 run directory")
    release, tables, partitions = validate_dataset(root)
    out = output / architecture
    out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for task, frame in tables.items():
        task_out = out / task
        task_out.mkdir()
        labels = torch.tensor(frame.label.map({v:i for i,v in enumerate(CLASSES[task])}).to_numpy(), dtype=torch.long)
        images = torch.stack([image_views(safe_path(root, row.image_path), CONFIG["seed"]+index) for index,row in enumerate(frame.itertuples())])
        oof, fold_reports = [], []
        for fold, split in enumerate(partitions[task], 1):
            seed_all(CONFIG["seed"]+fold)
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 2)
            configure_model(model, architecture)
            model.cuda()
            aligned = frame[["sample_key"]].merge(split[["sample_key", "split"]], validate="one_to_one")
            indices = {part: np.flatnonzero(aligned.split.eq(part)) for part in ["train", "validation", "test"]}
            train, validation, test = (indices[k] for k in ["train", "validation", "test"])
            weights = class_weights(labels[train]).cuda()
            criterion = nn.CrossEntropyLoss(weight=weights)
            groups = [{"params": list(model.fc.parameters()), "lr": CONFIG["head_lr"]}]
            if architecture == "B":
                groups.append({"params": list(model.layer4.parameters()), "lr": CONFIG["layer4_lr"]})
            optimizer = torch.optim.AdamW(groups, weight_decay=CONFIG["weight_decay"])
            loader = DataLoader(TensorDataset(images[train].flatten(0, 1), labels[train].repeat_interleave(3)),
                batch_size=CONFIG["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(CONFIG["seed"]+fold))
            val_loader = DataLoader(TensorDataset(images[validation, 0], labels[validation]), batch_size=16)
            history, best, best_epoch, stale = [], float("inf"), 0, 0
            split_hash = digest(root / f"splits/{task}/fold_{fold}_split.csv")
            metadata = {"fold": fold, "config": CONFIG, "workbook_sha256": release["workbook_sha256"],
                "split_sha256": split_hash, "manifest_sha256": digest(root / f"{task}_usable.csv"),
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
                "gpu": torch.cuda.get_device_name(0), "class_weights": weights.cpu().tolist()}
            for epoch in range(1, CONFIG["max_epochs"]+1):
                configure_model(model, architecture)
                total, denominator = 0., 0.
                for batch, target in loader:
                    optimizer.zero_grad()
                    batch, target = batch.cuda(), target.cuda()
                    loss = criterion(model(batch), target)
                    loss.backward()
                    optimizer.step()
                    weight = weights[target].sum().item()
                    total += loss.item()*weight
                    denominator += weight
                model.eval()
                val_total, val_weight = 0., 0.
                with torch.no_grad():
                    for batch, target in val_loader:
                        batch, target = batch.cuda(), target.cuda()
                        val_total += nn.functional.cross_entropy(model(batch), target, weight=weights, reduction="sum").item()
                        val_weight += weights[target].sum().item()
                val_loss = val_total / val_weight
                history.append({"epoch":epoch,"training_loss":total/denominator,"validation_loss":val_loss})
                if val_loss < best-CONFIG["min_delta"]:
                    best, best_epoch, stale = val_loss, epoch, 0
                    torch.save(checkpoint_payload(model, task, architecture, {**metadata,"best_epoch":epoch}), task_out / f"fold_{fold}.pth")
                else:
                    stale += 1
                if stale >= CONFIG["patience"]:
                    break
            checkpoint = torch.load(task_out/f"fold_{fold}.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            probability=[]
            with torch.no_grad():
                for batch in DataLoader(images[test, 0], batch_size=16):
                    probability.extend(torch.softmax(model(batch.cuda()),dim=1).cpu().tolist())
            predicted=np.argmax(probability,axis=1)
            metrics=scores(labels[test].numpy(),predicted,CLASSES[task])
            fold_reports.append({**metadata,"best_epoch":best_epoch,"history":history,"metrics":metrics})
            split.to_csv(task_out/f"fold_{fold}_split.csv",index=False)
            for index,pred,probs in zip(test,predicted,probability):
                oof.append({**frame.iloc[index].to_dict(),"fold":fold,"true_index":int(labels[index]),
                    "predicted_index":int(pred),"prediction":CLASSES[task][pred],"minority_probability":probs[1]})
            print(f"{architecture}/{task} fold {fold}: best epoch {best_epoch}; {metrics}",flush=True)
            del model,optimizer
            torch.cuda.empty_cache()
        predictions=pd.DataFrame(oof).sort_values("sample_key")
        if predictions.sample_key.duplicated().any() or set(predictions.sample_key)!=set(frame.sample_key):
            raise RuntimeError("Incomplete OOF coverage")
        metrics=scores(predictions.true_index,predictions.predicted_index,CLASSES[task])
        per_plant={plant:{"support":group.label.value_counts().to_dict(),
            "metrics":scores(group.true_index,group.predicted_index,CLASSES[task])} for plant,group in predictions.groupby("Plant_ID")}
        report={"dataset_version":VERSION,"architecture":architecture,"task":task,"oof_metrics":metrics,
            "per_plant":per_plant,"folds":fold_reports,"class_names":CLASSES[task],"preprocessing":PREPROCESSING,
            "workbook_sha256":release["workbook_sha256"],"config":CONFIG,
            "majority_baseline":scores(predictions.true_index,np.zeros(len(predictions),dtype=int),CLASSES[task]),
            "note":"Fold checkpoints only; no test-selected best fold or full-data deployment refit. Reused outer data has informed development, not a fresh external test."}
        predictions.to_csv(task_out/"out_of_fold_predictions.csv",index=False)
        (task_out/"evaluation.json").write_text(json.dumps(report,indent=2))
        save_confusion(metrics,task_out/"confusion_matrix.png",f"{architecture} {task} OOF")
        pd.DataFrame([{ "Plant_ID":plant,**value["metrics"],"support":value["support"]} for plant,value in per_plant.items()]).to_csv(task_out/"per_plant.csv",index=False)
    return str(out)


def compare_results(output_root):
    root=Path(output_root)
    rows=[]
    for task in CLASSES:
        reports={arch:json.loads((root/arch/task/"evaluation.json").read_text()) for arch in ["A","B"]}
        a,b=reports["A"],reports["B"]
        for field in ["dataset_version","workbook_sha256","class_names","preprocessing","config"]:
            if a[field]!=b[field]:
                raise ValueError(f"Incomparable results: {field}")
        if [f["split_sha256"] for f in a["folds"]]!=[f["split_sha256"] for f in b["folds"]]:
            raise ValueError("A/B split hashes differ")
        predictions={arch:pd.read_csv(root/arch/task/"out_of_fold_predictions.csv").set_index("sample_key").sort_index() for arch in ["A","B"]}
        for field in ["label","sha256","fold","true_index","Plant_ID","Sample_ID"]:
            if not predictions['A'][field].equals(predictions['B'][field]):
                raise ValueError("A/B OOF populations differ")
        paired=predictions['A'][["Plant_ID","label","fold"]].copy()
        paired['A_prediction']=predictions['A'].prediction
        paired['B_prediction']=predictions['B'].prediction
        paired.to_csv(root/f"{task}_paired_comparison.csv")
        for plant in ["ALL"]+sorted(a['per_plant']):
            left=a['oof_metrics'] if plant=='ALL' else a['per_plant'][plant]['metrics']
            right=b['oof_metrics'] if plant=='ALL' else b['per_plant'][plant]['metrics']
            for metric in ["macro_f1","balanced_accuracy","minority_precision","minority_recall"]:
                av,bv=left[metric],right[metric]
                rows.append({"task":task,"Plant_ID":plant,"metric":metric,"A":av,"B":bv,
                    "B_minus_A":bv-av if av is not None and bv is not None else None})
    pd.DataFrame(rows).to_csv(root/"model_comparison.csv",index=False)
    lines=["# Experiment 2 A/B comparison","","All results use V2 corrected targets and identical saved train/validation/test partitions.","",
        "| Task | Metric | A | B | B - A |","|---|---|---:|---:|---:|"]
    for row in rows:
        if row['Plant_ID']=='ALL':
            lines.append(f"| {row['task']} | {row['metric']} | {row['A']} | {row['B']} | {row['B_minus_A']} |")
    lines += ["","See model_comparison.csv for per-plant differences and paired CSVs for sample disagreements. All fold results remain in evaluation.json. Small minority support, absent-class strata and previously reviewed outer folds limit inference; no automatic winner/deployment checkpoint is selected."]
    (root/"comparison_report.md").write_text('\n'.join(lines))
    return rows
