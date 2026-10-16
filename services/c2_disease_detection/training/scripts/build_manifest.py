import os
import csv
from pathlib import Path

RAW_DIR = Path("data/c2-disease/raw")
OUTPUT_CSV = Path("data/c2-disease/dataset_manifest.csv")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SYMPTOM_MAP = {
    "healthy": "Optimal foliar health, absence of active infectious lesions.",
    "bacterial": "Water-soaked lesions, soft tissue rot, and bacterial exudate margins.",
    "fungi": "Necrotic circular spots, target rings, and fungal spore margins.",
    "abiotic": "Non-pathogenic physiological stress, tip burn, scorch, or nutrient deficiency.",
    "unknown": "Unverified symptom profile held for agronomist diagnostic review."
}

rows = []

for root, dirs, files in os.walk(RAW_DIR):
    for f in sorted(files):
        ext = Path(f).suffix.lower()
        if ext in VALID_EXTS:
            rel_folder = os.path.relpath(root, RAW_DIR).lower()
            disease_class = rel_folder if rel_folder != "." else "unknown"
            
            symptom = SYMPTOM_MAP.get(disease_class, "Field nursery foliar condition.")

            rows.append({
                "filename": f,
                "relative_path": os.path.join(root, f).replace("\\", "/"),
                "plant_species": "Ornamental Foliage",
                "disease_class": disease_class,
                "symptom_description": symptom,
                "source": "Mike Flora Nursery"
            })

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "relative_path", "plant_species", "disease_class", "symptom_description", "source"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Manifest successfully generated with {len(rows)} images mapped to your categories.")