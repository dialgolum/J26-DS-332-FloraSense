"""
Few-shot evaluation metrics: Accuracy, Precision, Recall, Macro-F1 across
N-way K-shot episodes, plus episode sampling and mean +/- std aggregation.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def classification_summary(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> Dict:
    """Accuracy + macro and per-class precision / recall / F1 + confusion matrix."""
    labels = list(labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, lab in enumerate(labels)
        },
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def sample_episode(
    indices_by_class: Mapping[str, Sequence[int]],
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Random N-way K-shot episode. Support and query never overlap."""
    eligible = [c for c, idx in indices_by_class.items() if len(idx) >= k_shot + 1]
    if len(eligible) < n_way:
        raise ValueError(f"Need {n_way} classes with >= {k_shot + 1} images, found {len(eligible)}")
    classes = list(rng.choice(eligible, size=n_way, replace=False))
    support, query = {}, {}
    for c in classes:
        perm = rng.permutation(np.asarray(indices_by_class[c]))
        support[c] = perm[:k_shot]
        query[c] = perm[k_shot : k_shot + q_query]
    return classes, support, query


def aggregate_runs(runs: Sequence[Mapping]) -> Dict[str, Dict[str, float]]:
    """Mean and std of every scalar metric across repeated random support draws."""
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    return {
        k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
        for k in keys
    }