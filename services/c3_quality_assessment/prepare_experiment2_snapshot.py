"""Freeze corrected C3 data and reuse Experiment 1 partitions; never train.

python -B -m services.c3_quality_assessment.prepare_experiment2_snapshot
Creates a new release only; existing releases and Experiment 1 remain intact.
"""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pandas as pd

from .data_loader import ROOT, EXCEL_PATH
from .prepare_feature_data import prepare, KEY, validate_split_groups

VERSION = "c3_features_v2_20261022"
CORRECTIONS = {"P06:S007": ("Acceptable", "Not Acceptable"),
               "P07:S002": ("Not Acceptable", "Acceptable"),
               "P07:S013": ("Not Acceptable", "Acceptable")}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/quality_assessment/processed/experiment2/v2_20261022")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "data/quality_assessment/processed").resolve()):
        parser.error("Output must be under ignored processed data")
    if output.exists():
        parser.error("Release already exists; do not overwrite a frozen snapshot")
    baseline = ROOT / "models/checkpoints/c3_features/baseline_v1"
    original_zip = ROOT / "data/quality_assessment/processed/colab/florasense_c3_training.zip"
    protected = {str(p.relative_to(ROOT)): digest(p) for p in baseline.rglob("*") if p.is_file()}
    protected[str(original_zip.relative_to(ROOT))] = digest(original_zip)
    protected[str(EXCEL_PATH.relative_to(ROOT))] = digest(EXCEL_PATH)
    workbook_bytes = EXCEL_PATH.read_bytes()
    workbook_hash = hashlib.sha256(workbook_bytes).hexdigest()
    manifests, audit = prepare(pd.read_excel(BytesIO(workbook_bytes), sheet_name=None), ROOT)
    audit.update(dataset_version=VERSION, workbook_sha256=workbook_hash)
    if audit["image_issues"] or audit["orphan_annotations"]:
        raise ValueError("Unresolved source image issues; inspect preparation audit")
    payloads, images, split_reports = {}, {}, {}
    changes = []
    with zipfile.ZipFile(original_zip) as archive:
        original_meta = json.loads(archive.read("bundle.json"))
        for task, manifest in manifests.items():
            usable = manifest[manifest.status.eq("usable")].copy()
            old = pd.read_csv(BytesIO(archive.read(f"{task}_usable.csv")), keep_default_na=False)
            joined = old.merge(usable, on=KEY, suffixes=("_v1", "_v2"), validate="one_to_one", how="outer", indicator=True)
            if not joined["_merge"].eq("both").all():
                raise ValueError("Sample population differs from Experiment 1")
            for field in ["sha256", "Image_ID", "sample_key", "view_type"]:
                if not joined[field + "_v1"].eq(joined[field + "_v2"]).all():
                    raise ValueError(f"Image identity/content changed: {field}")
            actual = {}
            for row in joined[joined.label_v1.ne(joined.label_v2)].itertuples():
                actual[row.sample_key_v2] = (row.label_v1, row.label_v2)
                changes.append({"task": task, "sample_key": row.sample_key_v2,
                    "previous_label": row.label_v1, "corrected_label": row.label_v2,
                    "status": "researcher-confirmed intentional correction", "confirmed_date": "2026-10-22"})
            if actual != (CORRECTIONS if task == "root" else {}):
                raise ValueError(f"Unexpected {task} target changes: {actual}")
            for row in usable.itertuples():
                path = Path(row.image_path)
                if digest(path) != row.sha256:
                    raise ValueError("Image changed during source validation")
                name = f"images/{row.Plant_ID}/{path.name}"
                images[name] = path
            usable["image_path"] = [f"images/{row.Plant_ID}/{Path(row.image_path).name}" for row in usable.itertuples()]
            payloads[f"{task}_usable.csv"] = usable.to_csv(index=False).encode()
            split_reports[task] = []
            oof = pd.read_csv(baseline / task / "out_of_fold_predictions.csv")
            test_coverage = []
            for fold in [1, 2, 3]:
                original_path = baseline / task / f"fold_{fold}_split.csv"
                saved = pd.read_csv(original_path, keep_default_na=False)
                if saved.sample_key.duplicated().any() or set(saved.sample_key) != set(usable.sample_key):
                    raise ValueError("Saved split coverage is ambiguous")
                if set(saved.split) != {"train", "validation", "test"}:
                    raise ValueError("Unexpected partition name")
                for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                    validate_split_groups(saved[saved.split.eq(a)], saved[saved.split.eq(b)])
                new = usable.merge(saved[KEY + ["split"]], on=KEY, validate="one_to_one")
                for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
                    validate_split_groups(new[new.split.eq(a)], new[new.split.eq(b)])
                test = set(new.loc[new.split.eq("test"), "sample_key"])
                if test != set(oof.loc[oof.fold.eq(fold), "sample_key"]):
                    raise ValueError("Outer test membership differs from original OOF evaluation")
                test_coverage.extend(test)
                payloads[f"splits/{task}/fold_{fold}_split.csv"] = new.to_csv(index=False).encode()
                split_reports[task].append({"fold": fold, "original_split_sha256": digest(original_path),
                    "membership_identical": True, "sample_path_hash_disjoint": True,
                    "class_support": {part: group.label.value_counts().to_dict() for part, group in new.groupby("split")}})
            if len(test_coverage) != len(set(test_coverage)) or set(test_coverage) != set(usable.sample_key):
                raise ValueError("Outer test coverage is not exactly once per sample")
    branch_path = ROOT / "data/quality_assessment/processed/error_analysis/baseline_v1/review_20261021/plant_branch_review_20261021/leaf_six_FN_branch_origin_review.csv"
    payloads["review/leaf_branch_origin_ambiguities.csv"] = branch_path.read_bytes()
    payloads["source/quality_dataset_v2.xlsx"] = workbook_bytes
    payloads["annotation_guidelines.md"] = (ROOT / "docs/quality_assessment/annotation_guidelines.md").read_bytes()
    release = {"dataset_version": VERSION, "release_date": "2026-10-22", "ground_truth": "current workbook; researcher-confirmed corrected annotations",
        "workbook_sha256": workbook_hash, "experiment1_workbook_sha256": original_meta["workbook_sha256"],
        "experiment1_zip_sha256": digest(original_zip), "confirmed_root_corrections": changes,
        "tasks": audit["tasks"], "split_validation": split_reports,
        "unresolved": ["Leaf branch qualification/visibility review remains: P02:S012, P04:S003, P06:S004, P06:S013, P07:S002, P07:S013. Existing One Side labels retained.",
            "P07:S002 small opposite growth needs expert qualifying-branch decision; P04:S003 junctions obscured.",
            "P08 lacks verified feature annotations/images; P05 has no One Side samples; corrected P07 has no Not Acceptable roots.",
            "Exact-image checks cannot identify every different-photo recapture of one physical plant."],
        "readiness": "Dataset and fixed partitions validated. Future training must load supplied splits, never regenerate them. No Experiment 2 training implementation supplied.",
        "comparison_policy": "Re-establish A on v2 for a fair corrected-root A/B comparison; never attribute v1/v2 label effects to model improvement."}
    payloads["release.json"] = json.dumps(release, indent=2).encode()
    payloads["dataset_audit.json"] = json.dumps(audit, indent=2).encode()
    payloads["corrections.csv"] = pd.DataFrame(changes).to_csv(index=False).encode()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    checksums.update({name: digest(path) for name, path in images.items()})
    payloads["checksums.json"] = json.dumps(checksums, indent=2).encode()
    output.mkdir(parents=True, exist_ok=False)
    local = output / "validation_manifests"
    local.mkdir()
    for task, manifest in manifests.items():
        manifest.to_csv(local / f"{task}_manifest.csv", index=False)
        manifest[manifest.status.eq("usable")].to_csv(local / f"{task}_usable.csv", index=False)
        manifest[manifest.status.eq("excluded")].to_csv(local / f"{task}_excluded.csv", index=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    final = output / f"{VERSION}.zip"
    partial = final.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in images.items():
            archive.write(path, name)
        for name, data in payloads.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(partial) as archive:
        for name, expected in checksums.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive verification failed: {name}")
    partial.replace(final)
    final.with_suffix(".zip.sha256").write_text(f"{digest(final)}  {final.name}\n")
    if any(digest(ROOT / name) != value for name, value in protected.items()):
        raise RuntimeError("Protected Experiment 1 inputs or workbook changed")
    (output / "preservation.json").write_text(json.dumps({"all_protected_inputs_unchanged": True, "sha256": protected}, indent=2))
    print(json.dumps({"dataset_version": VERSION, "zip": str(final), "tasks": audit["tasks"], "image_issues": audit["image_issue_counts"]}, indent=2))


if __name__ == "__main__":
    main()
