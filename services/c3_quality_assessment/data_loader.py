
from pathlib import Path

import pandas as pd

# Repository root: three levels above this Python file
ROOT = Path(__file__).resolve().parents[2]

EXCEL_PATH = (
    ROOT
    / "data"
    / "quality_assessment"
    / "annotations"
    / "quality_dataset.xlsx"
)

SHEET_NAMES = [
    "Plant_Types",
    "Requirement_Profiles",
    "Samples",
    "Height_Assessments",
    "Quality_Annotations",
    "Images",
]


def load_dataset():
    """Load the quality assessment workbook."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel workbook not found: {EXCEL_PATH}"
        )

    sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)

    missing = set(SHEET_NAMES) - set(sheets.keys())
    if missing:
        raise ValueError(f"Missing Excel sheets: {sorted(missing)}")

    for sheet_name in SHEET_NAMES:
        sheets[sheet_name].columns = (
            sheets[sheet_name].columns.str.strip()
        )

    return sheets


def resolve_image_path(relative_path):
    """Convert a workbook relative path to a local file path."""
    cleaned = str(relative_path).strip().replace("\\", "/")
    return ROOT / cleaned


def validate_dataset(sheets):
    """Print basic data and image-path validation results."""
    print("\n--- QUALITY DATASET VALIDATION ---")

    for name in SHEET_NAMES:
        df = sheets[name]
        print(f"{name}: {len(df)} rows")

    images = sheets["Images"]
    missing_images = []

    for _, row in images.iterrows():
        path_value = row.get("Relative_Path")

        if pd.isna(path_value) or not str(path_value).strip():
            missing_images.append(
                (row.get("Image_ID"), "Missing relative path")
            )
            continue

        image_path = resolve_image_path(path_value)

        if not image_path.is_file():
            missing_images.append(
                (row.get("Image_ID"), str(image_path))
            )

    print(f"\nImage records: {len(images)}")
    print(f"Missing image files: {len(missing_images)}")

    for image_id, path in missing_images[:10]:
        print(f"  {image_id}: {path}")

    print("\nValidation finished.")


if __name__ == "__main__":
    dataset = load_dataset()
    validate_dataset(dataset)
