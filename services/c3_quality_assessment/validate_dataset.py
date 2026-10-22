
from pathlib import Path
import pandas as pd

# Repository root: the folder containing README.md and data/
ROOT = Path(__file__).resolve().parents[2]

EXCEL_FILE = (
    ROOT
    / "data"
    / "quality_assessment"
    / "annotations"
    / "quality_dataset.xlsx"
)

REPORT_DIR = ROOT / "data" / "quality_assessment" / "processed"

EXPECTED_SHEETS = [
    "Plant_Types",
    "Requirement_Profiles",
    "Samples",
    "Height_Assessments",
    "Quality_Annotations",
    "Images",
]

# IDs that should identify unique records in these sheets.
PRIMARY_KEYS = {
    "Plant_Types": ["Plant_ID"],
    "Requirement_Profiles": ["Requirement_ID"],
    "Samples": ["Sample_ID"],
    "Images": ["Image_ID"],
}

# Relationships to check when both columns are present.
FOREIGN_KEYS = [
    ("Samples", "Plant_ID", "Plant_Types", "Plant_ID"),
    (
        "Samples",
        "Requirement_ID",
        "Requirement_Profiles",
        "Requirement_ID",
    ),
    (
        "Requirement_Profiles",
        "Plant_ID",
        "Plant_Types",
        "Plant_ID",
    ),
]


def main():
    if not EXCEL_FILE.exists():
        raise FileNotFoundError(
            f"Excel workbook not found:\n{EXCEL_FILE}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    workbook = pd.ExcelFile(EXCEL_FILE)
    print("Workbook:", EXCEL_FILE)
    print("\nAvailable sheets:", workbook.sheet_names)

    missing_sheets = [
        name for name in EXPECTED_SHEETS
        if name not in workbook.sheet_names
    ]

    if missing_sheets:
        print("\nWARNING: Expected sheets not found:", missing_sheets)

    data = {
        sheet: pd.read_excel(EXCEL_FILE, sheet_name=sheet)
        for sheet in workbook.sheet_names
    }

    issues = []

    # 1. Check each sheet's structure, missing values and key IDs.
    for sheet_name, df in data.items():
        print(f"\n{'=' * 55}")
        print(f"SHEET: {sheet_name}")
        print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        print("Columns:", df.columns.tolist())

        if df.empty:
            issues.append({
                "Sheet": sheet_name,
                "Issue": "Sheet is empty",
                "Details": "",
            })
            continue

        # Report missing cells without assuming every field is mandatory.
        missing = df.isna().sum()
        for column, count in missing[missing > 0].items():
            print(f"Missing values — {column}: {count}")

        # Check duplicate primary IDs only where configured.
        for key in PRIMARY_KEYS.get(sheet_name, []):
            if key not in df.columns:
                issues.append({
                    "Sheet": sheet_name,
                    "Issue": "Expected ID column missing",
                    "Details": key,
                })
                continue

            duplicate_mask = (
                df[key].notna()
                & df[key].duplicated(keep=False)
            )

            if duplicate_mask.any():
                duplicates = df.loc[duplicate_mask, key].tolist()
                issues.append({
                    "Sheet": sheet_name,
                    "Issue": f"Duplicate {key}",
                    "Details": str(duplicates),
                })
                print(f"WARNING: Duplicate {key}: {duplicates}")

    # 2. Check relationships between sheets when applicable.
    for child_sheet, child_col, parent_sheet, parent_col in FOREIGN_KEYS:
        if child_sheet not in data or parent_sheet not in data:
            continue

        child = data[child_sheet]
        parent = data[parent_sheet]

        if child_col not in child.columns or parent_col not in parent.columns:
            # Some workbooks intentionally link requirements separately.
            continue

        valid_ids = set(parent[parent_col].dropna().astype(str).str.strip())
        values = child[child_col].dropna().astype(str).str.strip()

        invalid = sorted(set(values) - valid_ids)

        if invalid:
            issues.append({
                "Sheet": child_sheet,
                "Issue": f"Unmatched {child_col}",
                "Details": str(invalid),
            })
            print(
                f"\nWARNING: {child_sheet}.{child_col} "
                f"contains IDs not found in "
                f"{parent_sheet}.{parent_col}: {invalid}"
            )

    # 3. Verify image paths against the repository.
    if "Images" in data:
        images = data["Images"]

        if "Relative_Path" not in images.columns:
            issues.append({
                "Sheet": "Images",
                "Issue": "Relative_Path column missing",
                "Details": "Check the exact column name in Excel.",
            })
        else:
            for _, row in images.iterrows():
                image_id = row.get("Image_ID", "Unknown")
                relative_path = row.get("Relative_Path")

                if pd.isna(relative_path) or not str(relative_path).strip():
                    issues.append({
                        "Sheet": "Images",
                        "Issue": "Missing image path",
                        "Details": str(image_id),
                    })
                    continue

                # Paths in Excel should be relative to the repository root.
                path_text = str(relative_path).strip().replace("\\", "/")
                image_path = (ROOT / path_text).resolve()

                # Prevent paths from accidentally escaping the repository.
                if not image_path.is_relative_to(ROOT.resolve()):
                    issues.append({
                        "Sheet": "Images",
                        "Issue": "Path outside repository",
                        "Details": f"{image_id}: {path_text}",
                    })
                    continue

                if not image_path.is_file():
                    issues.append({
                        "Sheet": "Images",
                        "Issue": "Image file not found",
                        "Details": f"{image_id}: {path_text}",
                    })

            print("\nImage path checks completed.")

    # 4. Save a report for fixing errors.
    report = pd.DataFrame(
        issues,
        columns=["Sheet", "Issue", "Details"],
    )

    report_file = REPORT_DIR / "dataset_validation_report.csv"
    report.to_csv(report_file, index=False)

    print(f"\n{'=' * 55}")
    print("VALIDATION FINISHED")
    print("Issues found:", len(issues))
    print("Report saved to:", report_file)

    if issues:
        print("\nReview the report before using the dataset.")
    else:
        print("No issues detected by these checks.")


if __name__ == "__main__":
    main()
