import csv
import os
from pathlib import Path

# Paths configured for healthy images in the raw folder
HEALTHY_DIR = Path("data/c2_disease/raw/healthy")
OUTPUT_CSV = Path("data/c2_disease/healthy_manifest1.csv")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SYMPTOM_DESCRIPTION = "Optimal foliar health, absence of active infectious lesions."


def build_healthy_manifest(healthy_dir: Path = HEALTHY_DIR, output_csv: Path = OUTPUT_CSV) -> int:
    if not healthy_dir.is_dir():
        raise FileNotFoundError(f"Healthy image directory does not exist: {healthy_dir}")

    rows = []
    # Scan through healthy folder recursively to capture direct or nested files
    for root, dirs, files in os.walk(healthy_dir):
        dirs.sort()
        for filename in sorted(files):
            # Skip hidden files or non-image assets
            if Path(filename).suffix.lower() not in VALID_EXTS:
                continue

            full_path = Path(root) / filename
            # Store clean forward-slash relative path
            clean_rel_path = str(full_path.as_posix())

            rows.append({
                "filename": filename,
                "relative_path": clean_rel_path,
                "plant_species": "Ornamental Foliage",
                "disease_class": "healthy",
                "symptom_description": SYMPTOM_DESCRIPTION,
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
    count = build_healthy_manifest()
    print(f"Healthy manifest successfully generated with {count} images at: {OUTPUT_CSV}")