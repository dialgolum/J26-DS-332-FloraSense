"""
Confidence calibration and out-of-distribution (OOD) detection metrics.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def ood_auroc(id_conf: np.ndarray, ood_conf: np.ndarray) -> float:
    """AUROC of P_max for separating known (positive) from unknown (negative)."""
    if len(id_conf) == 0 or len(ood_conf) == 0:
        return float("nan")
    y = np.r_[np.ones(len(id_conf)), np.zeros(len(ood_conf))]
    return float(roc_auc_score(y, np.r_[id_conf, ood_conf]))


def ood_rejection_rate(ood_conf: np.ndarray, tau: float) -> float:
    """Fraction of unknown images correctly rejected (P_max < tau)."""
    ood_conf = np.asarray(ood_conf, float)
    return float((ood_conf < tau).mean()) if len(ood_conf) else float("nan")


def selective_metrics(conf: np.ndarray, correct: np.ndarray, tau: float) -> Dict[str, float]:
    """Coverage and accuracy on the accepted (confident) subset."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    accepted = conf >= tau
    return {
        "coverage": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else float("nan"),
        "escalation_rate": float(1.0 - accepted.mean()),
    }


def threshold_sweep(
    id_conf: np.ndarray,
    id_correct: np.ndarray,
    ood_conf: np.ndarray,
    taus: Sequence[float] = tuple(np.round(np.arange(0.30, 0.96, 0.05), 2)),
) -> List[Dict[str, float]]:
    """Trade-off table used to justify tau on validation data."""
    rows = []
    for t in taus:
        sel = selective_metrics(id_conf, id_correct, t)
        rows.append(
            {
                "tau": float(t),
                **sel,
                "ood_rejection": ood_rejection_rate(ood_conf, t),
            }
        )
    return rows


def fit_temperature(
    cosine_scores: np.ndarray,
    labels: np.ndarray,
    grid: Sequence[float] = tuple(np.round(np.geomspace(0.02, 1.0, 40), 4)),
) -> float:
    """Pick T minimising NLL of softmax(cos / T) on validation data."""
    s = np.asarray(cosine_scores, float)
    y = np.asarray(labels, int)
    best_t, best_nll = float(grid[0]), np.inf
    for t in grid:
        z = s / t
        z -= z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(len(y)), y].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t