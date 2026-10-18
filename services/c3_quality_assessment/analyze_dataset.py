
from pathlib import Path

import pandas as pd


# Repository root: two levels above this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WORKBOOK_PATH = (
    PROJECT_ROOT
    / "data"
    / "quality_assessment"
    / "annotations"
    / "quality_dataset.xlsx"
)

IMAGE_ROOT = PROJECT_ROOT / "data" / "quality_assessment" / "original"


def main():
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Workbook not found: {WORKBOOK_PATH}"
        )

    sheets = pd.read_excel(WORKBOOK_PATH, sheet_name=None)

    print("\n========== DATASET SUMMARY ==========")

    for sheet_name, df in sheets.items():
        print(f"\n--- {sheet_name} ---")
        print(f"Rows: {len(df)}")
        print("Columns:", list(df.columns))

        if df.empty:
            print("This sheet is empty.")
            continue

        print("\nMissing values:")
        missing = df.isna().sum()
        print(missing[missing > 0].to_string() or "None")

    print("\n========== LABEL DISTRIBUTIONS ==========")

    label_columns = {
        "Plant_Types": [
            "Height_Check",
            "Leaf_Count_Check",
            "Leaf_Distribution_Check",
            "Root_Distribution",
            "Oasis_Condition",
            "Overall_Quality",
        ],
        "Height_Assessments": ["Height_Result"],
        "Quality_Annotations": [
            "Leaf_Distribution",
            "Root_Distribution",
            "Oasis_Damaged",
            "General_visual_Quality",
        ],
    }

    for sheet_name, columns in label_columns.items():
        df = sheets.get(sheet_name)

        if df is None:
            continue

        print(f"\n--- {sheet_name} ---")

        for column in columns:
            if column in df.columns:
                print(f"\n{column}:")
                print(
                    df[column]
                    .fillna("<MISSING>")
                    .astype(str)
                    .value_counts(dropna=False)
                    .to_string()
                )

    print("\n========== IMAGE FILE CHECK ==========")

    images = sheets.get("Images")

    if images is None:
        print("Images sheet not found.")
        return

    print("Image columns:", list(images.columns))

    path_columns = [
        column
        for column in images.columns
        if "path" in column.lower()
        or "file" in column.lower()
    ]

    if not path_columns:
        print(
            "Could not identify an image-path column automatically. "
            "Check the Image columns printed above."
        )
        return

    path_column = path_columns[0]
    missing_files = []

    for value in images[path_column].dropna():
        image_path = PROJECT_ROOT / str(value)

        if not image_path.exists():
            missing_files.append(str(value))

    print(f"Image records: {len(images)}")
    print(f"Path column checked: {path_column}")
    print(f"Missing files: {len(missing_files)}")

    for path in missing_files:
        print("MISSING:", path)

    print("\nDataset audit finished.")


if __name__ == "__main__":
    main()
