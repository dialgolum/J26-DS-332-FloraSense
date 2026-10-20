import csv
import os
from pathlib import Path

# Configured strictly for local Mike Flora nursery images
RAW_DIR = Path("data/c2_disease/raw")
OUTPUT_CSV = Path("data/c2_disease/dataset_manifest1.csv")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SYMPTOM_MAP = {
    "healthy": "Optimal foliar health, absence of active infectious lesions.",
    "bacterial": "Water-soaked lesions, soft tissue rot, and bacterial exudate margins.",
    "fungi": "Necrotic circular spots, target rings, and fungal spore margins.",
    "abiotic": "Non-pathogenic physiological stress, tip burn, scorch, or nutrient deficiency.",
    "unknown": "Unverified symptom profile held for agronomist diagnostic review."
}


def build_manifest(raw_dir: Path = RAW_DIR, output_csv: Path = OUTPUT_CSV) -> int:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory does not exist: {raw_dir}")

    rows = []
    # Scan through class subdirectories: healthy, bacterial, fungi, abiotic, unknown
    for root, dirs, files in os.walk(raw_dir):
        dirs.sort()
        for filename in sorted(files):
            # Skip hidden files like .gitkeep or non-image assets
            if Path(filename).suffix.lower() not in VALID_EXTS:
                continue

            # Class label derived from immediate subfolder name
            rel_folder = os.path.relpath(root, raw_dir).lower()
            disease_class = rel_folder if rel_folder != "." else "unknown"
            symptom = SYMPTOM_MAP.get(disease_class, "Field nursery foliar condition.")

            # Store clean forward-slash relative paths
            clean_rel_path = os.path.join(root, filename).replace("\\", "/")

            rows.append({
                "filename": filename,
                "relative_path": clean_rel_path,
                "plant_species": "Ornamental Foliage",
                "disease_class": disease_class,
                "symptom_description": symptom,
                "source": "Mike Flora Nursery",
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open(mode="w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "filename",
                "relative_path",
                "plant_species",
                "disease_class",
                "symptom_description",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    count = build_manifest()
    print(f"Manifest successfully generated with {count} nursery images at: {OUTPUT_CSV}")