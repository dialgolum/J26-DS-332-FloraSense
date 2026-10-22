"""
Evaluate on the REAL nursery images (data/c2_disease/raw).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from services.c2_disease_detection import config
from services.c2_disease_detection.evaluation.metrics.calibration_metrics import (
    expected_calibration_error,
    ood_auroc,
    ood_rejection_rate,
    selective_metrics,
    threshold_sweep,
)
from services.c2_disease_detection.evaluation.metrics.few_shot_metrics import (
    aggregate_runs,
    classification_summary,
)
from services.c2_disease_detection.training.scripts.nursery_data import load_splits
from services.c2_disease_detection.training.scripts.train_nursery import extract_features
from services.c2_disease_detection.vision.backbone.mobilenet_loader import (
    EmbeddingNet,
    get_device,
    load_checkpoint,
)
from services.c2_disease_detection.vision.inference.classifier import DiseaseEngine
from services.c2_disease_detection.vision.preprocessing.augmentations import get_inference_transform


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def proto_probs(support: Dict[str, np.ndarray], queries: np.ndarray, classes: List[str], T: float) -> np.ndarray:
    protos = _unit(np.stack([support[c].mean(0) for c in classes]))
    logits = _unit(queries) @ protos.T / T
    return F.softmax(torch.as_tensor(logits), dim=1).numpy()


def run_trial(method: Dict, classes: List[str], k: int, rng: np.random.Generator, T: float, tau: float):
    emb, ood, test = method["emb"], method["ood"], method.get("test")
    support, q_x, q_y = {}, [], []
    for c in classes:
        perm = rng.permutation(len(emb[c]))
        support[c] = emb[c][perm[:k]]
        if test is not None:
            q_x.append(test[c])
            q_y += [c] * len(test[c])
        else:
            q_x.append(emb[c][perm[k:]])
            q_y += [c] * (len(perm) - k)
    q_x = np.concatenate(q_x)
    if method["kind"] == "proto":
        probs = proto_probs(support, q_x, classes, T)
        ood_probs = proto_probs(support, ood, classes, T) if len(ood) else np.zeros((0, len(classes)))
    else:
        X = _unit(np.concatenate([support[c] for c in classes]))
        y = np.concatenate([[i] * len(support[c]) for i, c in enumerate(classes)])
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(X, y)
        probs = clf.predict_proba(_unit(q_x))
        ood_probs = clf.predict_proba(_unit(ood)) if len(ood) else np.zeros((0, len(classes)))
    pred = [classes[i] for i in probs.argmax(1)]
    summ = classification_summary(q_y, pred, classes)
    conf = probs.max(1)
    correct = (np.asarray(pred) == np.asarray(q_y)).astype(float)
    return summ, conf, correct, ood_probs.max(1) if len(ood_probs) else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--tau", type=float, default=config.TAU)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--baselines", default="none")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--ignore-splits", action="store_true")
    ap.add_argument("--out-prefix", default="nursery_eval")
    args = ap.parse_args()

    device = get_device()
    model, extra = load_checkpoint(args.checkpoint, device)
    T = args.temperature or float(extra.get("temperature", config.DEFAULT_TEMPERATURE))
    tf = get_inference_transform()

    splits_path = args.splits or (config.SPLITS_PATH if config.SPLITS_PATH.exists() and not args.ignore_splits else None)
    if splits_path:
        sdf = load_splits(splits_path)
        paths = {c: [Path(x) for x in sdf[(sdf["label"] == c) & sdf["split"].isin(["train", "val"])]["path"]] for c in config.CORE_CLASSES}
        test_paths = {c: [Path(x) for x in sdf[(sdf["label"] == c) & (sdf["split"] == "test")]["path"]] for c in config.CORE_CLASSES}
        ood_paths = [Path(x) for x in sdf[sdf["label"] == config.OOD_FOLDER]["path"]]
    else:
        paths = {c: config.list_images(args.raw_dir / c) for c in config.CORE_CLASSES}
        test_paths = None
        ood_paths = config.list_images(args.raw_dir / config.OOD_FOLDER)

    paths = {c: p for c, p in paths.items() if p}
    if test_paths:
        test_paths = {c: test_paths[c] for c in paths}

    def eligible(k):
        return [c for c in paths if len(paths[c]) >= k + (0 if test_paths else 1) and (not test_paths or len(test_paths[c]) >= 1)]

    def run_extract(plist, tag):
        if not plist:
            return np.zeros((0, model.feature_dim), np.float32)
        f, ok = extract_features(model, plist, tf, device, desc=f"backbone:{tag}")
        return f[ok]

    raw_f = {c: run_extract(p, c) for c, p in paths.items()}
    raw_test = {c: run_extract(test_paths[c], c + "-test") for c in paths} if test_paths else None
    raw_ood = run_extract(ood_paths, "unknown")

    with torch.no_grad():
        proj = lambda a: model.project(torch.from_numpy(a).to(device)).cpu().numpy() if len(a) else np.zeros((0, model.embedding_dim), np.float32)
        proposed_emb = {c: proj(a) for c, a in raw_f.items()}
        proposed_ood = proj(raw_ood)
        proposed_test = {c: proj(a) for c, a in raw_test.items()} if raw_test else None

    methods = {
        "Proposed: ProtoNet (meta-trained head)": dict(emb=proposed_emb, ood=proposed_ood, test=proposed_test, kind="proto"),
        f"Baseline: {model.backbone_name} ImageNet features, no meta-training": dict(emb=raw_f, ood=raw_ood, test=raw_test, kind="proto"),
    }

    results = {"temperature": T, "tau": args.tau, "repeats": args.repeats, "methods": {}}
    main_name = next(iter(methods))
    for mname, method in methods.items():
        results["methods"][mname] = {}
        for k in args.k:
            classes = eligible(k)
            if len(classes) < 2:
                continue
            rng = np.random.default_rng(args.seed)
            runs = []
            for _ in range(args.repeats):
                summ, conf, corr, oc = run_trial(method, classes, k, rng, T, args.tau)
                row = {x: summ[x] for x in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
                if mname == main_name:
                    row.update(selective_metrics(conf, corr, args.tau))
                    row["ece"] = expected_calibration_error(conf, corr)
                    row["ood_rejection"] = ood_rejection_rate(oc, args.tau)
                    row["ood_auroc"] = ood_auroc(conf, oc)
                runs.append(row)
            results["methods"][mname][f"{k}-shot"] = aggregate_runs(runs)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mpath = config.REPORTS_DIR / f"{args.out_prefix}.md"
    f = lambda d, key: f"{d[key]['mean']:.3f} +/- {d[key]['std']:.3f}"
    lines = [f"# Nursery evaluation (T={T:.3f}, tau={args.tau})\n",
             "| Method | K | Accuracy | Precision | Recall | Macro-F1 |", "|---|---|---|---|---|---|"]
    for mname, ks in results["methods"].items():
        for kname, d in ks.items():
            lines.append(f"| {mname} | {kname} | {f(d,'accuracy')} | {f(d,'macro_precision')} | {f(d,'macro_recall')} | {f(d,'macro_f1')} |")
    mpath.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()