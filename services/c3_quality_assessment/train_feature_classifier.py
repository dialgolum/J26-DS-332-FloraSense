"""Colab GPU frozen ImageNet ResNet18 baselines with nested grouped evaluation.

python -m services.c3_quality_assessment.train_feature_classifier --task both
Uses only audited *_usable.csv manifests. Outer folds are never used for early
stopping. The final model is refit on all samples for the median inner-selected
epoch count; it has no independent test score. Fold models retain their own
validation-selected checkpoints. No API, workbook, or legacy model writes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image, ImageOps
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from .data_loader import ROOT
from .prepare_feature_data import EXCEL_PATH, sample_key, validate_split_groups


CLASSES = {"root": ["Acceptable", "Not Acceptable"], "leaf": ["Both Sides", "One Side"]}
MANIFEST_DIR = ROOT / "data/quality_assessment/processed/feature_classification"
LEGACY_MODEL = ROOT / "services/c3_quality_assessment/notebooks/quality_resnet18.pth"
PREPROCESSING = {"image_size": 224, "content_size": 208, "resize": "aspect-preserving contain, centered padding",
                 "fill_rgb": [127, 127, 127], "exif_transpose": True,
                 "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                 "augmentation": "horizontal flip p=0.5, rotation +/-4 degrees, brightness/contrast/saturation +/-8%; no crops"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_verified_manifest(path: Path, task: str) -> pd.DataFrame:
    audit = json.loads((path.parent / "dataset_audit.json").read_text(encoding="utf-8"))
    bundle_path = path.parent / "bundle.json"
    portable = bundle_path.is_file()
    if portable:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle["workbook_sha256"] != audit["workbook_sha256"]:
            raise ValueError("Portable audit provenance mismatch")
        if bundle["file_sha256"].get(path.name) != file_hash(path):
            raise ValueError("Portable manifest checksum mismatch")
        if bundle["file_sha256"].get("dataset_audit.json") != file_hash(path.parent / "dataset_audit.json"):
            raise ValueError("Portable audit checksum mismatch")
    elif audit["workbook_sha256"] != file_hash(EXCEL_PATH):
        raise ValueError("Workbook changed; regenerate the dataset manifests")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"Plant_ID", "Sample_ID", "sample_key", "image_path", "label", "sha256", "view_type", "status", "exclusion_reason"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Missing manifest columns or empty verified dataset")
    if not frame.status.eq("usable").all() or not frame.exclusion_reason.eq("").all():
        raise ValueError("Training accepts only usable records with no exclusions")
    if not frame.label.isin(CLASSES[task]).all():
        raise ValueError("Unknown feature labels")
    if not frame.view_type.eq("Root" if task == "root" else "Front").all():
        raise ValueError("Wrong image view for task")
    allowed = {f"P{i:02d}" for i in range(1 if task == "root" else 2, 9)}
    if not frame.Plant_ID.isin(allowed).all():
        raise ValueError("Unsupported plant type")
    if any(row.sample_key != sample_key(row.Plant_ID, row.Sample_ID) for row in frame.itertuples()):
        raise ValueError("Composite sample identity mismatch")
    if portable:
        def resolve_portable(value):
            relative = Path(value)
            resolved = (path.parent / relative).resolve()
            if relative.is_absolute() or ":" in value or "\\" in value or not resolved.is_relative_to(path.parent.resolve()):
                raise ValueError(f"Unsafe portable image path: {value}")
            if bundle["file_sha256"].get(value) != frame.loc[frame.image_path.eq(value), "sha256"].iloc[0]:
                raise ValueError("Image provenance mismatch")
            return str(resolved)
        frame["image_path"] = frame.image_path.map(resolve_portable)
    for row in frame.itertuples():
        if file_hash(row.image_path) != row.sha256:
            raise ValueError(f"Image changed since validation: {row.image_path}")
    # Fail closed on ambiguous file ownership; preparation quarantines these.
    for field in ["image_path", "sha256"]:
        if frame.groupby(field).sample_key.nunique().gt(1).any():
            raise ValueError(f"Shared {field} across samples")
    if len(frame) != audit["tasks"][task]["usable_images"] or frame.label.value_counts().to_dict() != audit["tasks"][task]["class_counts"]:
        raise ValueError("Manifest counts disagree with dataset audit")
    return frame.reset_index(drop=True)


def grouped_folds(frame, labels, folds=3, seed=42, minimum=2):
    """Find deterministic label-stratified folds with both classes in each part."""
    labels = np.asarray(labels)
    best_parts, best_imbalance = None, float("inf")
    for attempt in range(50):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + attempt)
        parts = list(splitter.split(frame, labels, frame.sample_key))
        if all(np.bincount(labels[a], minlength=2).min() >= minimum and
               np.bincount(labels[b], minlength=2).min() >= minimum for a, b in parts):
            # Choose by label support only, never by model predictions/scores.
            target = np.bincount(labels, minlength=2) / folds
            imbalance = sum(float(np.square(np.bincount(labels[b], minlength=2) - target).sum()) for _, b in parts)
            if imbalance < best_imbalance:
                best_parts, best_imbalance = parts, imbalance
    if best_parts is not None:
        for a, b in best_parts:
            validate_split_groups(frame.iloc[a], frame.iloc[b])
        return best_parts
    raise ValueError("Too few independent minority groups for requested folds; collect more labels")


def evaluation_metrics(truth, predictions, classes):
    report = classification_report(truth, predictions, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0)
    return {"confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
            "matrix_order": classes, "macro_f1": float(f1_score(truth, predictions, average="macro", labels=[0, 1], zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)) if len(set(truth)) == 2 else None,
            "minority_class": classes[1], "minority_precision": report[classes[1]]["precision"],
            "minority_recall": report[classes[1]]["recall"], "classification_report": report}


def extract_features(frame, backbone, augmentation_views, seed, device="cuda"):
    """Cache outputs of a fixed, eval-mode encoder; no dataset fitting occurs.

    Index 0 is deterministic evaluation; additional views are training-only.
    This is safe across folds only because the ImageNet encoder is never updated.
    """
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])
    augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(4, fill=127),
                                  transforms.ColorJitter(brightness=.08, contrast=.08, saturation=.08)])
    features = []
    backbone.eval()
    for index, row in enumerate(frame.itertuples()):
        with Image.open(row.image_path) as source:
            source.draft("RGB", (1024, 1024))
            image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (208, 208), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (224, 224), (127, 127, 127))
            canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
        seed_all(seed + index)
        views = [normalize(canvas)] + [normalize(augment(canvas)) for _ in range(augmentation_views)]
        with torch.inference_mode():
            features.append(backbone(torch.stack(views).to(device)).cpu().clone())
        if (index + 1) % 20 == 0 or index + 1 == len(frame):
            print(f"features {index + 1}/{len(frame)}", flush=True)
    # Normal tensors are needed for linear-layer gradient computation.
    return torch.stack(features).clone()


def fit_head(features, labels, train_indices, validation_indices, config, seed, fixed_epochs=None):
    seed_all(seed)
    device = config["device"]
    head = nn.Linear(features.shape[-1], 2).to(device)
    y = torch.as_tensor(labels, dtype=torch.long)
    train_indices = np.asarray(train_indices)
    counts = torch.bincount(y[train_indices], minlength=2).float()
    if counts.min() == 0:
        raise ValueError("Training split lacks a class")
    weights = counts.sum() / (2 * counts)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x_train = features[train_indices].reshape(-1, features.shape[-1])
    y_train = y[train_indices].repeat_interleave(features.shape[1])
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, history = None, []
    for epoch in range(1, (fixed_epochs or config["max_epochs"]) + 1):
        head.train()
        total = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(head(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y_batch)
        record = {"epoch": epoch, "training_loss": total / len(y_train)}
        if validation_indices is not None:
            head.eval()
            with torch.no_grad():
                val_loss = loss_fn(head(features[validation_indices, 0].to(device)), y[validation_indices].to(device)).item()
            record["validation_loss"] = val_loss
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
        history.append(record)
        if validation_indices is not None and stale >= config["patience"]:
            break
    if validation_indices is not None:
        head.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
    return head.eval(), history, best_epoch, weights.tolist()


def save_model(backbone, head, path, metadata):
    model = models.resnet18(weights=None)
    state = backbone.state_dict()
    state.update({"fc.weight": head.weight.detach().cpu(), "fc.bias": head.bias.detach().cpu()})
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(state)
    torch.save({"model_state_dict": model.state_dict(), **metadata}, path)


def train_task(task, args, output):
    frame = load_verified_manifest(args.manifest_dir / f"{task}_usable.csv", task)
    audit = json.loads((args.manifest_dir / "dataset_audit.json").read_text())
    labels = frame.label.map({label: index for index, label in enumerate(CLASSES[task])}).to_numpy()
    if np.bincount(labels, minlength=2)[1] > np.bincount(labels, minlength=2)[0]:
        raise ValueError("Expected class 1 to be the minority for this initial experiment")
    outer_parts = grouped_folds(frame, labels, args.folds, args.seed, minimum=3)
    # Pretrained weights only; the legacy general-quality checkpoint is never loaded.
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.to("cuda")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    features = extract_features(frame, backbone, args.augmentation_views, args.seed)
    config = {"learning_rate": args.learning_rate, "weight_decay": .01, "batch_size": 16,
              "max_epochs": args.max_epochs, "patience": args.patience, "seed": args.seed,
              "outer_folds": args.folds, "inner_folds": 3, "augmentation_views": args.augmentation_views,
              "backbone": "resnet18 IMAGENET1K_V1 frozen incl. BatchNorm eval", "trainable": "fc only",
              "checkpoint_selection": "minimum inner-validation weighted cross-entropy; patience early stopping",
              "decision": "argmax (fixed); no outer-fold threshold tuning", "device": "cuda",
              "gpu_name": torch.cuda.get_device_name(0)}
    config["versions"] = {"torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__}
    task_dir = output / task
    task_dir.mkdir()
    metadata = {"task": task, "architecture": "resnet18", "class_names": CLASSES[task], "image_size": 224,
                "preprocessing": PREPROCESSING, "training_config": config, "workbook_sha256": audit["workbook_sha256"],
                "manifest_sha256": file_hash(args.manifest_dir / f"{task}_usable.csv"),
                "class_to_idx": {name: index for index, name in enumerate(CLASSES[task])}}
    predictions, fold_reports, epochs = [], [], []
    for fold, (development, test) in enumerate(outer_parts, start=1):
        relative_train, relative_validation = grouped_folds(frame.iloc[development].reset_index(drop=True),
                                                           labels[development], 3, args.seed + fold, minimum=2)[0]
        train, validation = development[relative_train], development[relative_validation]
        for a, b in [(train, validation), (train, test), (validation, test)]:
            validate_split_groups(frame.iloc[a], frame.iloc[b])
        split_records = pd.concat([frame.iloc[index].assign(split=name) for name, index in
                                   [("train", train), ("validation", validation), ("test", test)]])
        split_records.to_csv(task_dir / f"fold_{fold}_split.csv", index=False)
        head, history, best_epoch, weights = fit_head(features, labels, train, validation, config, args.seed + fold)
        epochs.append(best_epoch)
        with torch.no_grad():
            probabilities = torch.softmax(head(features[test, 0].to("cuda")), dim=1).cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        metrics = evaluation_metrics(labels[test], predicted, CLASSES[task])
        fold_reports.append({"fold": fold, "best_epoch": best_epoch, "epochs_run": len(history),
            "class_weights": weights, "split_class_counts": {name: np.bincount(labels[index], minlength=2).tolist()
                for name, index in [("train", train), ("validation", validation), ("test", test)]}, "metrics": metrics,
            "history": history})
        save_model(backbone, head, task_dir / f"fold_{fold}.pth", {**metadata, "fold": fold, "best_epoch": best_epoch})
        for index, prediction, probability in zip(test, predicted, probabilities):
            predictions.append({**frame.iloc[index].to_dict(), "fold": fold, "true_index": int(labels[index]),
                                "predicted_index": int(prediction), "prediction": CLASSES[task][prediction],
                                "minority_probability": float(probability[1])})
        print(f"{task} fold {fold}: best_epoch={best_epoch}, macro_F1={metrics['macro_f1']:.3f}, balanced_accuracy={metrics['balanced_accuracy']:.3f}", flush=True)
    oof = pd.DataFrame(predictions).sort_values("sample_key")
    if len(oof) != len(frame) or oof.groupby("sample_key").fold.nunique().gt(1).any():
        raise RuntimeError("Out-of-fold coverage is invalid")
    pooled = evaluation_metrics(oof.true_index, oof.predicted_index, CLASSES[task])
    baseline = evaluation_metrics(oof.true_index, np.zeros(len(oof), dtype=int), CLASSES[task])
    per_plant = {plant: {"samples": len(group), "class_support": group.label.value_counts().to_dict(),
                        "metrics": evaluation_metrics(group.true_index, group.predicted_index, CLASSES[task])}
                 for plant, group in oof.groupby("Plant_ID")}
    fixed_epochs = max(1, int(np.median(epochs)))
    final, final_history, _, final_weights = fit_head(features, labels, np.arange(len(frame)), None, config,
                                                    args.seed + 1000, fixed_epochs=fixed_epochs)
    save_model(backbone, final, task_dir / "final_model.pth", {**metadata, "refit_epochs": fixed_epochs,
               "evaluation_note": "Full-data refit; CV evaluates the procedure, not this final checkpoint independently."})
    report = {**metadata, "evaluation": f"{args.folds}-way grouped outer CV with separate inner train/validation; no independent external test set",
              "usable_samples": len(frame), "class_counts": frame.label.value_counts().to_dict(),
              "oof_metrics": pooled, "majority_baseline": baseline, "per_plant": per_plant, "folds": fold_reports,
              "final_refit": {"epochs": fixed_epochs, "class_weights": final_weights, "history": final_history},
              "limitations": ["Small minority support: estimates vary substantially by fold; no production claim.",
                  "No P08 evidence; no One Side P05 examples.",
                  "Composite sample_key assumes workbook samples identify physical plants; cross-ID recaptures and near-duplicates need human verification.",
                  "Per-plant balanced accuracy is not informative for plant types lacking either class.",
                  "Frozen-backbone linear probe, not full-network fine-tuning. Fixed encoder features may be cached across folds; augmentation features enter training only.",
                  "Final full-data refit is untested independently. No model chosen by outer-test scores."],
              "model_path": str(task_dir / "final_model.pth")}
    oof.to_csv(task_dir / "out_of_fold_predictions.csv", index=False)
    (task_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{task} OOF: {json.dumps(pooled)}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["root", "leaf", "both"], default="both")
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/checkpoints/c3_features" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--augmentation-views", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.patience < 1 or args.augmentation_views < 0 or args.folds < 2:
        parser.error("Invalid epoch/patience/augmentation/fold configuration")
    # Deliberately fail before downloading weights or creating artifacts locally.
    if platform.system() == "Windows":
        raise RuntimeError("Training is restricted to Google Colab with GPU; local Windows training is disabled")
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Open the supplied notebook in Google Colab") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a T4 GPU in Colab: Runtime > Change runtime type")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(ROOT / "models/checkpoints/.torch_cache"))
    protected = {path: file_hash(path) for path in [LEGACY_MODEL, EXCEL_PATH] if path.is_file()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = {}
    try:
        for task in (["root", "leaf"] if args.task == "both" else [args.task]):
            reports[task] = train_task(task, args, args.output_dir)
        (args.output_dir / "summary.json").write_text(json.dumps({task: {"model_path": report["model_path"],
            "oof_metrics": report["oof_metrics"]} for task, report in reports.items()}, indent=2), encoding="utf-8")
    except Exception as exc:
        (args.output_dir / "error.json").write_text(json.dumps({"error": str(exc), "type": type(exc).__name__}, indent=2), encoding="utf-8")
        raise
    finally:
        if any(file_hash(path) != digest for path, digest in protected.items()):
            raise RuntimeError("Protected workbook or legacy checkpoint changed during training")
    print(f"Experiment outputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
