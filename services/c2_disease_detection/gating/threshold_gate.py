"""
Confidence gate (tau = 0.60) and export-quality policy.

Rule
----
    P_max = max_k softmax(logits)_k
    P_max >= tau  -> CONFIRMED          (automated diagnosis is allowed)
    P_max <  tau  -> UNKNOWN_ANOMALY    (automated decision halted, expert review)

Two companion uncertainty signals are reported for the reviewing agronomist:
  * margin  = P_(1) - P_(2)         (small margin => two diseases look alike)
  * entropy = -sum p log p / log C  (1.0 => completely undecided)

Export flag sent to Component 3 (Export Quality):
  healthy                    -> PASS
  abiotic (nutrient/sun/water stress, not contagious) -> FAIL
  bacterial / fungi / any enrolled pathogen           -> QUARANTINE
  UNKNOWN_ANOMALY            -> QUARANTINE (hold until an expert has looked)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from services.c2_disease_detection.config import ABIOTIC_LABELS, HEALTHY_LABEL, TAU

CONFIRMED = "CONFIRMED"
UNKNOWN_ANOMALY = "UNKNOWN_ANOMALY"

ADVISORY: Dict[str, str] = {
    "healthy": "No foliar pathology detected. Plant may proceed to export quality inspection.",
    "bacterial": (
        "Bacterial leaf disease suspected. Isolate the batch, avoid overhead watering, "
        "remove affected leaves, sanitise tools and apply a copper-based bactericide "
        "after agronomist confirmation."
    ),
    "fungi": (
        "Fungal leaf disease suspected. Improve airflow, reduce leaf wetness and humidity, "
        "remove infected foliage and apply a suitable fungicide after agronomist confirmation."
    ),
    "abiotic": (
        "Abiotic stress suspected (nutrient imbalance, sun scorch, over/under-watering or "
        "chemical injury). Not contagious: review fertiliser, irrigation and light/shade levels."
    ),
    UNKNOWN_ANOMALY: (
        "The system is not confident enough to diagnose this leaf. The image has been sent to "
        "the agronomist review queue. Hold the plant until it is reviewed."
    ),
}
GENERIC_PATHOGEN_ADVICE = (
    "Newly registered condition. Follow the guidance recorded by the agronomist who enrolled it "
    "and isolate affected plants until confirmed."
)


@dataclass
class GateDecision:
    status: str
    top_prediction: str
    confidence: float
    margin: float
    entropy: float
    tau: float
    accepted: bool
    probabilities: Dict[str, float] = field(default_factory=dict)
    top_k: List[Dict[str, float]] = field(default_factory=list)

    @property
    def disease_class(self) -> str:
        """Class used for downstream hand-offs (UNKNOWN_ANOMALY when gated out)."""
        return self.top_prediction if self.accepted else UNKNOWN_ANOMALY


class ThresholdGate:
    def __init__(self, tau: float = TAU) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.tau = tau

    def evaluate(self, probs: Sequence[float], class_names: Sequence[str], top_k: int = 3) -> GateDecision:
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 1 or p.shape[0] != len(class_names) or p.shape[0] == 0:
            raise ValueError("probs must be a 1-D vector aligned with class_names")
        order = np.argsort(-p)
        p_max = float(p[order[0]])
        runner_up = float(p[order[1]]) if len(p) > 1 else 0.0
        ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
        accepted = p_max >= self.tau
        return GateDecision(
            status=CONFIRMED if accepted else UNKNOWN_ANOMALY,
            top_prediction=class_names[int(order[0])],
            confidence=p_max,
            margin=p_max - runner_up,
            entropy=ent,
            tau=self.tau,
            accepted=accepted,
            probabilities={n: float(v) for n, v in zip(class_names, p)},
            top_k=[
                {"label": class_names[int(i)], "probability": float(p[i])}
                for i in order[: min(top_k, len(p))]
            ],
        )

    def accept_mask(self, prob_matrix: np.ndarray) -> np.ndarray:
        """Vectorised gate for evaluation: True where P_max >= tau."""
        return np.asarray(prob_matrix).max(axis=1) >= self.tau


def export_quality_flag(decision: GateDecision) -> str:
    if not decision.accepted:
        return "QUARANTINE"
    if decision.top_prediction == HEALTHY_LABEL:
        return "PASS"
    if decision.top_prediction in ABIOTIC_LABELS:
        return "FAIL"
    return "QUARANTINE"


def advisory_for(decision: GateDecision) -> str:
    if not decision.accepted:
        return ADVISORY[UNKNOWN_ANOMALY]
    return ADVISORY.get(decision.top_prediction, GENERIC_PATHOGEN_ADVICE)