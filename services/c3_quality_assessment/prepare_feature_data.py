"""Prepare audited feature manifests without changing source data or training models.

Run: python -m services.c3_quality_assessment.prepare_feature_data
Only *_usable.csv should be consumed by future training code. No split is made
here: future evaluation must group by sample_key (all views and augmentations).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from .data_loader import EXCEL_PATH, ROOT


TASKS = {
    "root": ("Root", "Root_Distribution", {f"P{i:02d}" for i in range(1, 9)}),
    "leaf": ("Front", "Leaf_Distribution", {f"P{i:02d}" for i in range(2, 9)}),
}
KEY = ["Plant_ID", "Sample_ID"]
COLUMNS = [*KEY, "sample_key", "Image_ID", "view_type", "image_path",
           "label", "raw_label", "sha256", "status", "exclusion_reason"]


def clean(value) -> str:
    return "" if pd.isna(value) else " ".join(str(value).split())


def sample_key(plant_id, sample_id) -> str:
    """Preserve local IDs while namespacing physical sample identity by plant."""
    return f"{clean(plant_id)}:{clean(sample_id)}"


def normalize_label(value, task: str) -> str | None:
    """Explicit aliases only; unknown values never become inferred labels."""
    token = re.sub(r"[\s_-]+", "", clean(value).casefold())
    aliases = {
        "root": {"acceptable": "Acceptable", "notacceptable": "Not Acceptable",
                 "unacceptable": "Not Acceptable"},
        "leaf": {"bothsides": "Both Sides", "bothside": "Both Sides",
                 "oneside": "One Side", "onesides": "One Side"},
    }
    return aliases[task].get(token)


def validate_split_groups(train: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Guard future splits against sample, canonical-path, or exact-file leakage."""
    for column in ["sample_key", "image_path", "sha256"]:
        left = {clean(v) for v in train[column] if clean(v)}
        right = {clean(v) for v in evaluation[column] if clean(v)}
        if left & right:
            raise ValueError(f"Data leakage in {column}: {sorted(left & right)}")


def prepare(sheets: dict[str, pd.DataFrame], root: Path) -> tuple[dict, dict]:
    """Return manifests and an audit; this function never writes files."""
    root = root.resolve()
    required = {"Samples": KEY, "Quality_Annotations": KEY + ["Root_Distribution", "Leaf_Distribution"],
                "Images": KEY + ["Image_ID", "View_Type", "Relative_Path"]}
    tables = {}
    for name, columns in required.items():
        if name not in sheets or not set(columns).issubset(sheets[name].columns):
            raise ValueError(f"{name} requires columns: {columns}")
        tables[name] = sheets[name].copy()
        for column in KEY:
            tables[name][column] = tables[name][column].map(clean)
        if (tables[name][KEY] == "").any().any():
            raise ValueError(f"{name} contains a blank composite key")
    samples, annotations, images = (tables[n] for n in required)
    if samples.duplicated(KEY).any():
        raise ValueError("Samples contains duplicate composite keys")

    # Preserve duplicate annotation evidence rather than allowing a many-to-many join.
    annotation_groups = annotations.groupby(KEY, sort=False)
    annotation_summary = []
    for key, group in annotation_groups:
        record = dict(zip(KEY, key))
        record["annotation_count"] = len(group)
        for task, (_, column, _) in TASKS.items():
            values = {normalize_label(v, task) for v in group[column]}
            record[column] = group.iloc[0][column]
            record[f"{task}_conflict"] = len(values - {None}) > 1
        annotation_summary.append(record)
    summary = pd.DataFrame(annotation_summary, columns=KEY + ["annotation_count", "Root_Distribution",
                            "root_conflict", "Leaf_Distribution", "leaf_conflict"])
    context = samples.merge(summary, on=KEY, how="left", validate="one_to_one")

    # Canonicalize paths and inspect every reference, including non-training views.
    def canonical(value):
        text = clean(value).replace("\\", "/")
        return str((root / text).resolve()) if text else ""

    images["image_path"] = images.Relative_Path.map(canonical)
    images["view_type"] = images.View_Type.map(lambda v: clean(v).title())
    images["sample_key"] = [sample_key(*k) for k in images[KEY].itertuples(index=False, name=None)]
    checks = {}
    for text in sorted(set(images.image_path)):
        reasons, digest = [], ""
        path = Path(text) if text else None
        if path is None:
            reasons.append("missing_image_path")
        elif not path.is_relative_to(root):
            reasons.append("path_outside_repository")
        elif not path.is_file():
            reasons.append("missing_file")
        else:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                reasons.append("unreadable_image")
        checks[text] = {"reasons": reasons, "sha256": digest}
    images["sha256"] = images.image_path.map(lambda p: checks[p]["sha256"])
    duplicate_paths = set(images.loc[images.image_path.ne("") & images.image_path.duplicated(False), "image_path"])
    shared_hashes = {digest for digest, group in images[images.sha256.ne("")].groupby("sha256")
                     if group.sample_key.nunique() > 1}
    duplicate_ids = set(images.loc[images.Image_ID.duplicated(False), "Image_ID"])
    valid_keys = set(map(tuple, samples[KEY].values))

    def image_reasons(row):
        reasons = list(checks[row.image_path]["reasons"])
        if row.image_path in duplicate_paths:
            reasons.append("duplicate_path_reference")
        if row.sha256 in shared_hashes:
            reasons.append("shared_image_content_across_samples")
        if row.Image_ID in duplicate_ids:
            reasons.append("duplicate_image_id")
        if (row.Plant_ID, row.Sample_ID) not in valid_keys:
            reasons.append("sample_not_found")
        if row.image_path:
            name = Path(row.image_path).stem
            prefix = f"{row.Plant_ID}_{row.Sample_ID}"
            if not (name == prefix or name.startswith(prefix + "_")):
                reasons.append("filename_sample_mismatch")
            suffix = name.rsplit("_", 1)[-1].casefold()
            if suffix in {"root", "front", "stem", "side", "focus"} and suffix != row.view_type.casefold():
                reasons.append("filename_view_mismatch")
        if row.Plant_ID == "P02" and row.Sample_ID in {f"S{i:03d}" for i in range(12, 17)}:
            # The owner has corrected these links. Release only exact, validated
            # plant/sample/view associations; all other exclusion checks remain.
            path = Path(row.image_path) if row.image_path else None
            expected = f"{row.Plant_ID}_{row.Sample_ID}_{row.view_type.lower()}"
            if path is None or path.stem != expected or path.parent.name != row.Plant_ID:
                reasons.append("p02_corrected_linkage_not_verified")
        return reasons

    images["reasons"] = [image_reasons(row) for row in images.itertuples()]
    manifests, task_audits = {}, {}
    for task, (view, column, plants) in TASKS.items():
        selected = images[images.Plant_ID.isin(plants) & images.view_type.eq(view)].copy()
        # Include missing-view samples and orphan images as explicit exclusions.
        merged = context[context.Plant_ID.isin(plants)].merge(
            selected, on=KEY, how="outer", validate="one_to_many", indicator=True)
        merged["label"] = merged[column].map(lambda v: normalize_label(v, task))
        conflict_paths = set()
        conflict_hashes = set()
        for field, target in [("image_path", conflict_paths), ("sha256", conflict_hashes)]:
            for value, group in merged.dropna(subset=[field]).groupby(field):
                if value and group.label.nunique() > 1:
                    target.add(value)
        records = []
        for _, row in merged.iterrows():
            reasons = list(row["reasons"]) if isinstance(row.get("reasons"), list) else ["missing_view_image"]
            if pd.isna(row.get("annotation_count")):
                reasons.append("missing_annotation")
            elif row.annotation_count != 1:
                reasons.append("duplicate_annotation")
            annotation_conflict = row.get(f"{task}_conflict")
            if (pd.notna(annotation_conflict) and bool(annotation_conflict)) or row.image_path in conflict_paths or row.sha256 in conflict_hashes:
                reasons.append("conflicting_labels")
            if row.label is None:
                reasons.append("missing_or_unsupported_label")
            records.append({"Plant_ID": row.Plant_ID, "Sample_ID": row.Sample_ID,
                "sample_key": sample_key(row.Plant_ID, row.Sample_ID), "Image_ID": clean(row.Image_ID),
                "view_type": view, "image_path": clean(row.image_path), "label": row.label or "",
                "raw_label": "" if pd.isna(row[column]) else str(row[column]), "sha256": clean(row.sha256),
                "status": "excluded" if reasons else "usable",
                "exclusion_reason": ";".join(dict.fromkeys(reasons))})
        manifest = pd.DataFrame(records, columns=COLUMNS)
        usable = manifest[manifest.status.eq("usable")]
        manifests[task] = manifest
        task_audits[task] = {"candidate_records": len(manifest), "usable_images": len(usable),
            "usable_samples": int(usable.sample_key.nunique()), "class_counts": usable.label.value_counts().to_dict(),
            "candidate_plant_counts": manifest.Plant_ID.value_counts().to_dict(),
            "plant_class_counts": {plant: group.label.value_counts().to_dict() for plant, group in usable.groupby("Plant_ID")},
            "excluded_records": int(manifest.status.eq("excluded").sum()),
            "exclusion_reason_counts": manifest.loc[manifest.status.eq("excluded"), "exclusion_reason"].str.split(";").explode().value_counts().to_dict()}
    audit = {"source_rows": {name: len(frame) for name, frame in sheets.items()},
        "sample_key": "Plant_ID:Sample_ID", "tasks": task_audits,
        "image_issue_counts": images.reasons.explode().dropna().value_counts().to_dict(),
        "normalization_policy": "Whitespace/case and explicit separator/spelling aliases only; unknown labels excluded. Original labels retained in raw_label.",
        "image_issues": [{"Image_ID": clean(row.Image_ID), "sample_key": row.sample_key,
                           "image_path": row.image_path, "reasons": row.reasons}
                          for row in images.itertuples() if row.reasons],
        "duplicate_references": [{"image_path": path, "Image_IDs": images.loc[images.image_path.eq(path), "Image_ID"].tolist()}
                                 for path in sorted(duplicate_paths)],
        "p02_corrected_reference_validation": [
            {"Image_ID": clean(row.Image_ID), "sample_key": row.sample_key,
             "image_path": row.image_path, "passed_image_checks": not bool(row.reasons),
             "reasons": row.reasons}
            for row in images.itertuples()
            if row.Plant_ID == "P02" and row.Sample_ID in {f"S{i:03d}" for i in range(12, 17)}],
        "orphan_annotations": [sample_key(*key) for key in annotation_groups.groups if key not in valid_keys],
        "evaluation_policy": "No splits generated. Group all views/crops/augmentations by sample_key; validate sample, path and SHA-256 disjointness. Verify physical identity and near-duplicates before evaluation.",
        "limitations": ["Owner-corrected P02 S012-S016 links require exact plant-directory/sample/view filenames and all ordinary validation checks; label exclusions still apply.",
                        "Shared paths and content across samples are excluded, never silently deduplicated.",
                        "P08 has no usable data unless verified image and annotation records exist.",
                        "No labels inferred from general quality or buyer acceptance."]}
    return manifests, audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=EXCEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/quality_assessment/processed/feature_classification")
    args = parser.parse_args(argv)
    before = hashlib.sha256(args.workbook.read_bytes()).hexdigest()
    sheets = pd.read_excel(args.workbook, sheet_name=None)
    manifests, audit = prepare(sheets, ROOT)
    if hashlib.sha256(args.workbook.read_bytes()).hexdigest() != before:
        raise RuntimeError("Workbook changed during preparation")
    audit["workbook_sha256"] = before
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, manifest in manifests.items():
        manifest.to_csv(args.output_dir / f"{task}_manifest.csv", index=False)
        manifest[manifest.status.eq("usable")].to_csv(args.output_dir / f"{task}_usable.csv", index=False)
        manifest[manifest.status.eq("excluded")].to_csv(args.output_dir / f"{task}_excluded.csv", index=False)
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit["tasks"], indent=2))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
