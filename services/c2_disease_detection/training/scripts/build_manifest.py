import csv
import os
from pathlib import Path

RAW_DIR = Path("data/c2_disease/public")
OUTPUT_CSV = Path("data/c2_disease/dataset_manifest.csv")
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
        raise FileNotFoundError(f"Image directory does not exist: {raw_dir}")

    rows = []
    for root, dirs, files in os.walk(raw_dir):
        dirs.sort()
        for filename in sorted(files):
            if Path(filename).suffix.lower() not in VALID_EXTS:
                continue

            rel_folder = os.path.relpath(root, raw_dir).lower()
            disease_class = rel_folder if rel_folder != "." else "unknown"
            symptom = SYMPTOM_MAP.get(disease_class, "Field nursery foliar condition.")

            rows.append({
                "filename": filename,
                "relative_path": os.path.join(root, filename).replace("\\", "/"),
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
    image_count = build_manifest()
    print(f"Manifest successfully generated with {image_count} images mapped to your categories.")