"""
Metric distance matcher + end-to-end inference engine.

Classification (Prototypical Networks)
--------------------------------------
Given a query embedding z and class prototypes c_k:

    cosine   : logit_k = cos(z, c_k) / T
    euclid   : logit_k = -||z - c_k||^2 / T
    p(k | x) = softmax(logit)_k

T is the softmax temperature. Small T sharpens the distribution (more confident);
it is fitted on validation episodes so that P_max is calibrated, which is what
makes a fixed gate tau = 0.60 meaningful.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from services.c2_disease_detection import config
from services.c2_disease_detection.gating.threshold_gate import GateDecision, ThresholdGate
from services.c2_disease_detection.vision.backbone.mobilenet_loader import (
    EmbeddingNet,
    get_device,
    load_checkpoint,
)
from services.c2_disease_detection.vision.preprocessing.augmentations import (
    ImageSource,
    get_inference_transform,
    load_image,
)
from services.c2_disease_detection.vision.prototypes.prototype_store import PrototypeStore


def prototype_logits(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float,
    metric: str = "cosine",
) -> torch.Tensor:
    """(B,D) x (C,D) -> (B,C) logits."""
    if metric == "cosine":
        return F.normalize(embeddings, dim=1) @ F.normalize(prototypes, dim=1).T / temperature
    if metric == "euclidean":
        return -torch.cdist(embeddings, prototypes).pow(2) / temperature
    raise ValueError("metric must be 'cosine' or 'euclidean'")


class PrototypeClassifier:
    """Stateless distance matcher over a PrototypeStore."""

    def __init__(self, temperature: float = config.DEFAULT_TEMPERATURE, metric: str = "cosine") -> None:
        self.temperature = float(temperature)
        self.metric = metric

    def predict_proba(self, embeddings: np.ndarray, store: PrototypeStore) -> Tuple[List[str], np.ndarray]:
        names, protos = store.matrix()
        if not names:
            raise RuntimeError("Prototype store is empty. Run build_prototypes.py first.")
        z = torch.as_tensor(np.atleast_2d(embeddings), dtype=torch.float32)
        c = torch.as_tensor(protos, dtype=torch.float32)
        probs = F.softmax(prototype_logits(z, c, self.temperature, self.metric), dim=1)
        return names, probs.numpy()


@dataclass
class Diagnosis:
    decision: GateDecision
    latency_ms: float
    embedding: np.ndarray


class DiseaseEngine:
    """Loads the embedding net + prototype store and serves diagnoses."""

    def __init__(
        self,
        model: EmbeddingNet,
        store: PrototypeStore,
        gate: ThresholdGate,
        temperature: float = config.DEFAULT_TEMPERATURE,
        metric: str = "cosine",
        device: Optional[torch.device] = None,
        prototype_path: Optional[Path] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.store = store
        self.gate = gate
        self.classifier = PrototypeClassifier(temperature, metric)
        self.prototype_path = Path(prototype_path) if prototype_path else config.PROTOTYPE_PATH
        self._transform = get_inference_transform()
        self._lock = threading.RLock()

    @classmethod
    def from_disk(
        cls,
        checkpoint_path: Path = config.CHECKPOINT_PATH,
        prototype_path: Path = config.PROTOTYPE_PATH,
        tau: float = config.TAU,
        metric: str = "cosine",
        device: Optional[torch.device] = None,
    ) -> "DiseaseEngine":
        device = device or get_device()
        model, extra = load_checkpoint(checkpoint_path, device)
        store = PrototypeStore.load_or_empty(prototype_path, dim=model.embedding_dim)
        temperature = 0.007
        return cls(model, store, ThresholdGate(tau), temperature, metric, device, prototype_path)

    # ------------------------------------------------------------ embedding
    @torch.no_grad()
    def embed_images(self, sources: Sequence[ImageSource], batch_size: int = 16) -> np.ndarray:
        """Embed any number of images -> (n, 128) float32 array."""
        out: List[np.ndarray] = []
        for i in range(0, len(sources), batch_size):
            batch = torch.stack([self._transform(load_image(s)) for s in sources[i : i + batch_size]])
            out.append(self.model.embed(batch.to(self.device)).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.model.embedding_dim), np.float32)

    # ------------------------------------------------------------ diagnosis
    def diagnose(self, source: ImageSource, top_k: int = 3) -> Diagnosis:
        start = time.perf_counter()
        emb = self.embed_images([source])
        with self._lock:
            names, probs = self.classifier.predict_proba(emb, self.store)
        decision = self.gate.evaluate(probs[0], names, top_k=top_k)
        return Diagnosis(decision, (time.perf_counter() - start) * 1000.0, emb[0])

    def save_prototypes(self) -> None:
        with self._lock:
            self.store.save(self.prototype_path)

    @property
    def temperature(self) -> float:
        return self.classifier.temperature

    @property
    def metric(self) -> str:
        return self.classifier.metric
