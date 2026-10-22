"""
Storage and serialisation of class centroid vectors (the "disease knowledge base").

Prototypical Networks (Snell et al., 2017): the prototype of class k is the mean
of its support embeddings

    c_k = (1 / |S_k|) * sum_{x_i in S_k} f_theta(x_i)

New evidence is added with an exact running mean, so enrolling more images never
needs the old images again:

    c' = (n * c + sum_j f(x_j)) / (n + m)

The store is thread-safe and persisted as JSON with an atomic write, so a crash
during a save can never corrupt the knowledge base (Data Integrity requirement).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PrototypeRecord:
    name: str
    centroid: np.ndarray
    n_support: int
    kind: str = "core"                 # "core" (trained at setup) | "enrolled" (added later)
    display_name: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "centroid": [float(v) for v in self.centroid],
            "n_support": int(self.n_support),
            "kind": self.kind,
            "display_name": self.display_name or self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrototypeRecord":
        return cls(
            name=d["name"],
            centroid=np.asarray(d["centroid"], dtype=np.float32),
            n_support=int(d["n_support"]),
            kind=d.get("kind", "core"),
            display_name=d.get("display_name", d["name"]),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
        )


class PrototypeStore:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._records: Dict[str, PrototypeRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------- queries
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: str) -> bool:
        return name in self._records

    @property
    def names(self) -> List[str]:
        with self._lock:
            return list(self._records)

    def get(self, name: str) -> PrototypeRecord:
        with self._lock:
            if name not in self._records:
                raise KeyError(f"Unknown prototype class '{name}'")
            return self._records[name]

    def records(self) -> List[PrototypeRecord]:
        with self._lock:
            return list(self._records.values())

    def matrix(self) -> Tuple[List[str], np.ndarray]:
        """(class names, (C, D) centroid matrix) in a consistent order."""
        with self._lock:
            names = list(self._records)
            if not names:
                return [], np.zeros((0, self.dim), dtype=np.float32)
            return names, np.stack([self._records[n].centroid for n in names]).astype(np.float32)

    # ------------------------------------------------------------ mutation
    def _check(self, embeddings: np.ndarray) -> np.ndarray:
        e = np.asarray(embeddings, dtype=np.float32)
        if e.ndim != 2 or e.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings of shape (n, {self.dim}), got {e.shape}")
        if e.shape[0] == 0:
            raise ValueError("At least one embedding is required")
        return e

    def set_class(
        self,
        name: str,
        embeddings: np.ndarray,
        kind: str = "core",
        display_name: Optional[str] = None,
        overwrite: bool = False,
    ) -> PrototypeRecord:
        """Create a class whose centroid is the mean of `embeddings`."""
        e = self._check(embeddings)
        with self._lock:
            if name in self._records and not overwrite:
                raise ValueError(f"Class '{name}' already exists (use overwrite=True)")
            created = self._records[name].created_at if name in self._records else _now()
            rec = PrototypeRecord(
                name=name,
                centroid=e.mean(axis=0),
                n_support=e.shape[0],
                kind=kind,
                display_name=display_name or name,
                created_at=created,
            )
            self._records[name] = rec
            return rec

    def add_support(self, name: str, embeddings: np.ndarray) -> PrototypeRecord:
        """Exact running-mean update of an existing class."""
        e = self._check(embeddings)
        with self._lock:
            rec = self.get(name)
            n, m = rec.n_support, e.shape[0]
            rec.centroid = (rec.centroid * n + e.sum(axis=0)) / (n + m)
            rec.n_support = n + m
            rec.updated_at = _now()
            return rec

    def remove_class(self, name: str) -> None:
        with self._lock:
            if name not in self._records:
                raise KeyError(f"Unknown prototype class '{name}'")
            del self._records[name]

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "dim": self.dim,
                "saved_at": _now(),
                "classes": [r.to_dict() for r in self._records.values()],
            }
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)  # atomic on Windows, macOS and Linux
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeStore":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(dim=int(payload["dim"]))
        for d in payload["classes"]:
            rec = PrototypeRecord.from_dict(d)
            store._records[rec.name] = rec
        return store

    @classmethod
    def load_or_empty(cls, path: str | Path, dim: int = 128) -> "PrototypeStore":
        return cls.load(path) if Path(path).exists() else cls(dim=dim)