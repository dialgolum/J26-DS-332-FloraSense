
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

WORKBOOK = (
    ROOT / "data" / "quality_assessment"
    / "annotations" / "quality_dataset.xlsx"
)

OUTPUT_DIR = ROOT / "data" / "quality_assessment" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)

    images = sheets["Images"]
    annotations = sheets["Quality_Annotations"]

    required_image_columns = {
        "Sample_ID", "Relative_Path"
    }
    required_annotation_columns = {
        "Sample_ID", "General_visual_Quality"
    }

    if not required_image_columns.issubset(images.columns):
        raise ValueError(
            f"Images sheet needs: {required_image_columns}"
        )

    if not required_annotation_columns.issubset(
        annotations.columns
    ):
        raise ValueError(
            "Quality_Annotations is missing required columns."
        )

    labels = annotations[
        ["Sample_ID", "General_visual_Quality"]
    ].copy()

    if labels["Sample_ID"].duplicated().any():
        raise ValueError(
            "Duplicate Sample_IDs in Quality_Annotations."
        )

    data = images.merge(
        labels,
        on="Sample_ID",
        how="left",
        validate="many_to_one",
    )

    data["Image_Path"] = data["Relative_Path"].map(
        lambda p: str(ROOT / str(p))
        if pd.notna(p) else ""
    )

    data["File_Exists"] = data["Image_Path"].map(
        lambda p: bool(p) and Path(p).is_file()
    )

    data["General_visual_Quality"] = (
        data["General_visual_Quality"]
        .astype("string")
        .str.strip()
    )

    valid_labels = {"Acceptable", "Not Acceptable"}

    data["Label_Valid"] = data[
        "General_visual_Quality"
    ].isin(valid_labels)

    print("Image records:", len(data))
    print("Distinct plant samples:", data["Sample_ID"].nunique())
    print("Missing image files:", (~data["File_Exists"]).sum())
    print("Unlabelled/unsupported records:", (~data["Label_Valid"]).sum())

    usable = data[
        data["File_Exists"] & data["Label_Valid"]
    ].copy()

    print("\nUsable image records:", len(usable))
    print("Usable distinct samples:", usable["Sample_ID"].nunique())
    print("\nLabel counts:")
    print(usable["General_visual_Quality"].value_counts())

    # Save the full manifest, including excluded rows for traceability.
    data.to_csv(
        OUTPUT_DIR / "quality_image_manifest.csv",
        index=False,
    )

    if usable["Sample_ID"].nunique() < 4:
        print(
            "\nNot enough distinct samples for a useful "
            "holdout split. Collect more labelled samples first."
        )
        return

    # Split by plant sample, not by image.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            usable,
            y=usable["General_visual_Quality"],
            groups=usable["Sample_ID"],
        )
    )

    train = usable.iloc[train_idx].copy()
    test = usable.iloc[test_idx].copy()

    # Check that both classes exist in each split.
    for name, part in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{name} label counts:")
        print(part["General_visual_Quality"].value_counts())
        print(f"{name} samples:", part["Sample_ID"].nunique())

    train.to_csv(OUTPUT_DIR / "quality_train.csv", index=False)
    test.to_csv(OUTPUT_DIR / "quality_test.csv", index=False)

    overlap = set(train["Sample_ID"]) & set(test["Sample_ID"])
    print("\nSample overlap between splits:", len(overlap))

    if overlap:
        raise RuntimeError("Sample leakage detected!")

    if (
        train["General_visual_Quality"].nunique() < 2
        or test["General_visual_Quality"].nunique() < 2
    ):
        print(
            "\nWARNING: One split lacks a class. "
            "Do not treat its evaluation as meaningful. "
            "We should adjust the split or collect more samples."
        )

    print("\nManifest and split files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
