"""
Human-in-the-loop staging buffer for low-confidence (UNKNOWN_ANOMALY) cases.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from services.c2_disease_detection import config

PENDING = "PENDING"
RESOLVED = "RESOLVED"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewQueue:
    def __init__(self, path: Path = config.REVIEW_QUEUE_PATH, upload_dir: Path = config.REVIEW_UPLOAD_DIR) -> None:
        self.path = Path(path)
        self.upload_dir = Path(upload_dir)
        self._lock = threading.RLock()
        self._items: List[Dict] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                self._items = json.load(fh)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._items, fh, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def submit(
        self,
        image_bytes: bytes,
        top_prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        species: Optional[str] = None,
        plant_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict:
        ticket_id = uuid.uuid4().hex[:12]
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "").suffix.lower()
        suffix = suffix if suffix in config.IMAGE_EXTS else ".jpg"
        image_path = self.upload_dir / f"{ticket_id}{suffix}"
        image_path.write_bytes(image_bytes)
        item = {
            "ticket_id": ticket_id,
            "status": PENDING,
            "created_at": _now(),
            "species": species,
            "plant_id": plant_id,
            "image_file": image_path.name,
            "top_prediction": top_prediction,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "resolution": None,
        }
        with self._lock:
            self._items.append(item)
            self._persist()
        return dict(item)

    def list(self, status: Optional[str] = None) -> List[Dict]:
        with self._lock:
            return [dict(i) for i in self._items if status is None or i["status"] == status]

    def get(self, ticket_id: str) -> Dict:
        with self._lock:
            for item in self._items:
                if item["ticket_id"] == ticket_id:
                    return dict(item)
        raise KeyError(f"Unknown review ticket '{ticket_id}'")

    def image_path(self, ticket_id: str) -> Path:
        return self.upload_dir / self.get(ticket_id)["image_file"]

    def resolve(self, ticket_id: str, label: str, reviewer: str, notes: Optional[str] = None) -> Dict:
        with self._lock:
            for item in self._items:
                if item["ticket_id"] == ticket_id:
                    if item["status"] == RESOLVED:
                        raise ValueError(f"Ticket '{ticket_id}' is already resolved")
                    item["status"] = RESOLVED
                    item["resolution"] = {
                        "label": label,
                        "reviewer": reviewer,
                        "notes": notes,
                        "resolved_at": _now(),
                    }
                    self._persist()
                    return dict(item)
        raise KeyError(f"Unknown review ticket '{ticket_id}'")

    def stats(self) -> Dict[str, int]:
        with self._lock:
            pending = sum(1 for i in self._items if i["status"] == PENDING)
            return {"pending": pending, "resolved": len(self._items) - pending, "total": len(self._items)}