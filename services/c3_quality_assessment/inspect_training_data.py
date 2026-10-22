
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = (
    ROOT / "data" / "quality_assessment"
    / "annotations" / "quality_dataset.xlsx"
)

sheets = pd.read_excel(WORKBOOK, sheet_name=None)
images = sheets["Images"]
annotations = sheets["Quality_Annotations"]

print("Image view types:")
print(images["View_Type"].value_counts(dropna=False))

print("\nImages per sample:")
print(images.groupby("Sample_ID").size().describe())

print("\nQuality annotation columns:")
print(annotations.columns.tolist())

print("\nImage records with sample IDs:")
print(
    images[
        ["Image_ID", "Sample_ID", "Image_Type",
         "View_Type", "Relative_Path"]
    ].head(12).to_string(index=False)
)

print("\nQuality labels:")
print(
    annotations[
        ["Sample_ID", "General_visual_Quality"]
    ].to_string(index=False)
)

merged = images.merge(
    annotations[["Sample_ID", "General_visual_Quality"]],
    on="Sample_ID",
    how="left",
    validate="many_to_one",
)

print("\nImage-to-label mapping:")
print(
    merged[
        ["Sample_ID", "View_Type", "Relative_Path",
         "General_visual_Quality"]
    ].to_string(index=False)
)

print("\nLabel counts after joining images:")
print(merged["General_visual_Quality"].value_counts(dropna=False))
