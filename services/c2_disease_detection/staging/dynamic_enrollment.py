"""
Dynamic enrollment: register a NEW disease / plant condition without retraining.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from services.c2_disease_detection import config
from services.c2_disease_detection.vision.preprocessing.augmentations import ImageSource
from services.c2_disease_detection.vision.prototypes.prototype_store import PrototypeStore

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
RESERVED_NAMES = {"unknown", "unknown_anomaly", "none", "null"}
COHESION_WARN = 0.60
DUPLICATE_WARN = 0.95


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class EnrollmentError(ValueError):
    pass


class DynamicEnrollment:
    def __init__(
        self,
        embed_fn: Callable[[Sequence[ImageSource]], np.ndarray],
        store: PrototypeStore,
        min_images: int = config.MIN_ENROLL_IMAGES,
    ) -> None:
        self.embed_fn = embed_fn
        self.store = store
        self.min_images = min_images

    @staticmethod
    def normalise_name(raw: str) -> str:
        name = re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")
        if not NAME_PATTERN.match(name) or name in RESERVED_NAMES:
            raise EnrollmentError(
                "Class name must be 2-40 characters, start with a letter and contain only "
                "letters, digits or underscores (e.g. 'anthracnose')."
            )
        return name

    def _quality(self, embeddings: np.ndarray, centroid: np.ndarray, exclude: Optional[str] = None) -> Dict:
        cohesion = float((_unit(embeddings) @ _unit(centroid)).mean())
        nearest, nearest_sim = None, None
        for rec in self.store.records():
            if rec.name == exclude:
                continue
            sim = float(_unit(centroid) @ _unit(rec.centroid))
            if nearest_sim is None or sim > nearest_sim:
                nearest, nearest_sim = rec.name, sim
        warnings = []
        if cohesion < COHESION_WARN:
            warnings.append("Images look dissimilar (low cohesion): check labels / mixed symptoms.")
        if nearest_sim is not None and nearest_sim > DUPLICATE_WARN:
            warnings.append(f"Very similar to existing class '{nearest}': possible duplicate.")
        return {
            "cohesion": cohesion,
            "nearest_class": nearest,
            "nearest_similarity": nearest_sim,
            "warnings": warnings,
        }

    def enroll_new_class(
        self,
        name: str,
        images: Sequence[ImageSource],
        display_name: Optional[str] = None,
        kind: str = "enrolled",
    ) -> Dict:
        name = self.normalise_name(name)
        if name in self.store:
            raise EnrollmentError(f"Class '{name}' already exists. Use add-to-class to extend it.")
        if len(images) < self.min_images:
            raise EnrollmentError(
                f"At least {self.min_images} validated images are required, got {len(images)}."
            )
        emb = self.embed_fn(images)
        centroid = emb.mean(axis=0)
        quality = self._quality(emb, centroid)
        rec = self.store.set_class(name, emb, kind=kind, display_name=display_name or name)
        return {"class_name": rec.name, "n_support": rec.n_support, "kind": rec.kind, **quality}

    def add_to_class(self, name: str, images: Sequence[ImageSource]) -> Dict:
        if name not in self.store:
            raise EnrollmentError(f"Class '{name}' does not exist.")
        if len(images) < 1:
            raise EnrollmentError("At least one image is required.")
        emb = self.embed_fn(images)
        rec = self.store.add_support(name, emb)
        quality = self._quality(emb, rec.centroid, exclude=name)
        return {"class_name": rec.name, "n_support": rec.n_support, "kind": rec.kind, **quality}