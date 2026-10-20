"""
Central configuration for Component 2 (FloraSense disease detection).
"""
from __future__ import annotations

import os
from pathlib import Path

SERVICE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SERVICE_DIR.parents[1]

# ----------------------------------------------------------------- data paths
DATA_DIR: Path = Path(os.getenv("C2_DATA_DIR", REPO_ROOT / "data" / "c2_disease"))
RAW_DIR: Path = DATA_DIR / "raw"
REVIEW_UPLOAD_DIR: Path = DATA_DIR / "review_uploads"
MANIFEST_PATH: Path = DATA_DIR / "dataset_manifest1.csv"
TELEMETRY_PATH: Path = DATA_DIR / "telemetry.jsonl"
METADATA_PATH: Path = DATA_DIR / "metadata.csv"
SPLITS_PATH: Path = DATA_DIR / "splits.csv"

# --------------------------------------------------------------- model paths
MODEL_DIR: Path = Path(os.getenv("C2_MODEL_DIR", REPO_ROOT / "models" / "c2_disease"))
CHECKPOINT_PATH: Path = MODEL_DIR / "backbone.pt"
PROTOTYPE_PATH: Path = MODEL_DIR / "prototypes.json"
REVIEW_QUEUE_PATH: Path = MODEL_DIR / "review_queue.json"
REPORTS_DIR: Path = SERVICE_DIR / "evaluation" / "reports"

# ----------------------------------------------------------------- label sets
CORE_CLASSES = ("healthy", "bacterial", "fungi", "abiotic")
OOD_FOLDER = "unknown"
HEALTHY_LABEL = "healthy"
ABIOTIC_LABELS = frozenset({"abiotic"})
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ------------------------------------------------------------------- tunables
TAU: float = float(os.getenv("C2_TAU", "0.60"))
MIN_ENROLL_IMAGES: int = int(os.getenv("C2_MIN_ENROLL", "5"))
DEFAULT_TEMPERATURE: float = 0.10
EMBEDDING_DIM: int = 128

# -------------------------------------------------------------- integrations
ADMIN_API_KEY: str = os.getenv("C2_ADMIN_KEY", "florasense-admin")
C4_TELEMETRY_URL: str = os.getenv("C4_TELEMETRY_URL", "")
MAX_UPLOAD_MB: int = int(os.getenv("C2_MAX_UPLOAD_MB", "15"))

def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)