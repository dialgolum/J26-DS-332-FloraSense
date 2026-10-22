"""Read-only analysis of saved C3 out-of-fold predictions; no torch or training.

python -m services.c3_quality_assessment.analyze_feature_errors
Positive classes: root Not Acceptable; leaf One Side. Resolve local images only
through verified manifests joined on (Plant_ID, Sample_ID), never Colab paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .data_loader import EXCEL_PATH, ROOT
from .prepare_feature_data import KEY, clean, normalize_label, sample_key, prepare


CLASSES = {"root": ["Acceptable", "Not Acceptable"], "leaf": ["Both Sides", "One Side"]}
MANIFEST_DIR = ROOT / "data/quality_assessment/processed/feature_classification"
ARTIFACT_DIR = ROOT / "models/checkpoints/c3_features/baseline_v1"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classify_error(truth, prediction, task):
    negative, positive = CLASSES[task]
    if truth not in (negative, positive) or prediction not in (negative, positive):
        raise ValueError("Unknown feature label")
    return { (negative, negative): "TN", (negative, positive): "FP",
             (positive, negative): "FN", (positive, positive): "TP" }[truth, prediction]


def analyze_predictions(predictions, verified, task, root=ROOT):
    """Return one review row per prediction, retaining invalid/unmatched rows.

    Ambiguous composite keys fail closed. Probabilities are model scores, not
    calibrated certainty. An unverified match never enters a contact sheet.
    """
    required = {*KEY, "sample_key", "Image_ID", "view_type", "image_path", "label", "sha256",
                "fold", "true_index", "predicted_index", "prediction", "minority_probability"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Prediction columns missing: {sorted(required - set(predictions.columns))}")
    source_required = {*KEY, "sample_key", "Image_ID", "view_type", "image_path", "label", "sha256", "status", "exclusion_reason"}
    if not source_required.issubset(verified.columns):
        raise ValueError("Verified manifest columns missing")
    predictions, verified = predictions.copy(), verified.copy()
    for frame in (predictions, verified):
        for column in KEY:
            frame[column] = frame[column].map(clean)
        if (frame[KEY] == "").any().any() or frame.duplicated(KEY).any():
            raise ValueError("Blank or ambiguous composite sample keys")
    source = verified.rename(columns={column: "verified_" + column for column in verified.columns if column not in KEY})
    merged = predictions.merge(source, on=KEY, how="left", validate="one_to_one", indicator=True)
    records = []
    checked = {}
    for _, row in merged.iterrows():
        issues = []
        key = sample_key(row.Plant_ID, row.Sample_ID)
        truth, prediction = normalize_label(row.label, task), normalize_label(row.prediction, task)
        if row.sample_key != key:
            issues.append("prediction_sample_key_mismatch")
        if truth is None or prediction is None:
            issues.append("unknown_prediction_label")
        true_index = pd.to_numeric(row.true_index, errors="coerce")
        predicted_index = pd.to_numeric(row.predicted_index, errors="coerce")
        if truth in CLASSES[task] and true_index != CLASSES[task].index(truth):
            issues.append("true_index_label_mismatch")
        if prediction in CLASSES[task] and predicted_index != CLASSES[task].index(prediction):
            issues.append("predicted_index_label_mismatch")
        probability = pd.to_numeric(row.minority_probability, errors="coerce")
        if pd.isna(probability) or not 0 <= probability <= 1:
            issues.append("invalid_probability")
        elif predicted_index in (0, 1) and probability != .5 and int(probability > .5) != predicted_index:
            issues.append("probability_prediction_mismatch")
        fold = pd.to_numeric(row.fold, errors="coerce")
        if pd.isna(fold) or fold < 1 or fold != int(fold):
            issues.append("invalid_fold")
        local_path = ""
        if row["_merge"] != "both":
            issues.append("missing_verified_source")
        else:
            local_path = clean(row.verified_image_path)
            if row.verified_sample_key != key:
                issues.append("verified_sample_key_mismatch")
            if row.verified_status != "usable" or clean(row.verified_exclusion_reason):
                issues.append("source_not_usable")
            expected_view = "Root" if task == "root" else "Front"
            if row.view_type != expected_view or row.verified_view_type != expected_view:
                issues.append("wrong_task_view")
            if row.Image_ID != row.verified_Image_ID:
                issues.append("image_id_mismatch")
            if truth != normalize_label(row.verified_label, task):
                issues.append("expert_label_changed")
            if row.sha256 != row.verified_sha256:
                issues.append("source_hash_changed_from_experiment")
            if local_path not in checked:
                path = Path(local_path).resolve()
                file_issues = []
                if not local_path or not path.is_relative_to(root.resolve()):
                    file_issues.append("source_outside_repository")
                elif not path.is_file():
                    file_issues.append("source_file_missing")
                else:
                    try:
                        if digest(path) != row.verified_sha256:
                            file_issues.append("source_hash_changed_from_manifest")
                        with Image.open(path) as image:
                            image.verify()
                    except Exception:
                        file_issues.append("unreadable_source")
                checked[local_path] = file_issues
            issues.extend(checked[local_path])
        error_type = classify_error(truth, prediction, task) if truth and prediction else "UNKNOWN"
        verified_match = not issues
        priority = 1 if error_type == "FN" else 2 if error_type == "FP" else 3
        if not verified_match:
            priority = 0  # integrity issues need resolution before visual review
        record = {"task": task, "Plant_ID": row.Plant_ID, "Sample_ID": row.Sample_ID, "sample_key": key,
            "Image_ID": row.Image_ID, "fold": row.fold, "expert_label": truth or clean(row.label),
            "predicted_label": prediction or clean(row.prediction), "error_type": error_type,
            "correct": error_type in {"TN", "TP"}, "review_priority": priority,
            "minority_probability": probability,
            "predicted_label_score": (probability if predicted_index == 1 else 1 - probability) if pd.notna(probability) else None,
            "source_image_path": local_path, "experiment_image_path": row.image_path,
            "source_sha256": clean(row.get("verified_sha256")), "experiment_sha256": row.sha256,
            "match_status": "verified" if verified_match else "unverified",
            "validation_issues": ";".join(dict.fromkeys(issues))}
        records.append(record)
    return pd.DataFrame(records).sort_values(["review_priority", "predicted_label_score", "sample_key"],
                                             ascending=[True, False, True]).reset_index(drop=True)


def summarize(frame):
    valid = frame[frame.match_status.eq("verified")]
    plant = valid.groupby("Plant_ID").error_type.value_counts().unstack(fill_value=0).reindex(columns=["TN", "FP", "FN", "TP"], fill_value=0)
    plant["samples"] = plant.sum(axis=1)
    plant["correct"] = plant.TN + plant.TP
    plant["errors"] = plant.FP + plant.FN
    plant["error_rate"] = plant.errors / plant.samples
    labels = valid.groupby(["Plant_ID", "expert_label", "predicted_label", "error_type"]).size().rename("samples").reset_index()
    pooled = valid.groupby(["expert_label", "predicted_label", "error_type"]).size().rename("samples").reset_index()
    counts = {kind: int(valid.error_type.eq(kind).sum()) for kind in ["TN", "FP", "FN", "TP"]}
    return plant.reset_index(), labels, pooled, {"predictions": len(frame), "verified": len(valid),
        "unverified": len(frame) - len(valid), "correct": counts["TN"] + counts["TP"],
        "errors": counts["FP"] + counts["FN"], "counts": counts}


def font(size):
    for name in ["DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def contact_sheets(rows, directory, prefix, per_page=12):
    """Only verified images; aspect preserved, no original images are changed."""
    directory.mkdir(parents=True, exist_ok=True)
    pages, associations = [], {}
    rows = rows.reset_index(drop=True)
    title_font, label_font = font(20), font(17)
    for start in range(0, len(rows), per_page):
        batch = rows.iloc[start:start + per_page]
        sheet = Image.new("RGB", (1200, ((len(batch) + 2) // 3) * 460 + 50), "#e7ece9")
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 12), f"{prefix} | positive = {CLASSES[batch.iloc[0].task][1]} | original views", fill="black", font=title_font)
        filename = f"{prefix}_{start // per_page + 1:03d}.jpg"
        for offset, row in enumerate(batch.itertuples()):
            if row.match_status != "verified":
                raise ValueError("Unverified image cannot enter a contact sheet")
            x, y = (offset % 3) * 400, (offset // 3) * 460 + 50
            draw.rectangle((x + 5, y + 5, x + 395, y + 455), fill="white", outline="#bb2424" if row.error_type == "FN" else "#a78319", width=3)
            with Image.open(row.source_image_path) as source:
                source.draft("RGB", (1024, 1024))
                thumbnail = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (380, 330))
            sheet.paste(thumbnail, (x + (400 - thumbnail.width) // 2, y + 8 + (330 - thumbnail.height) // 2))
            lines = [f"{row.sample_key} | {row.error_type} | fold {row.fold}",
                     f"Expert: {row.expert_label}", f"Predicted: {row.predicted_label}",
                     f"P(minority): {row.minority_probability:.3f}"]
            for line_index, line in enumerate(lines):
                draw.text((x + 12, y + 341 + line_index * 25), line, fill="black", font=label_font)
            associations[row.sample_key] = str(Path("contact_sheets") / filename)
        sheet.save(directory / filename, quality=92)
        pages.append(filename)
    return pages, associations


def run_analysis(artifacts, manifests, output, root=ROOT):
    audit = json.loads((manifests / "dataset_audit.json").read_text(encoding="utf-8"))
    workbook = root / EXCEL_PATH.relative_to(ROOT)
    current_hash = digest(workbook)
    refresh = current_hash != audit["workbook_sha256"]
    refreshed, current_audit = None, audit
    if refresh:
        print("Workbook hash differs from source audit; verifying an isolated current-source snapshot.", flush=True)
        refreshed, current_audit = prepare(pd.read_excel(workbook, sheet_name=None), root)
        current_audit["workbook_sha256"] = current_hash
        if digest(workbook) != current_hash:
            raise RuntimeError("Workbook changed while verifying current sources")
    # Protect the full artifact tree, including checkpoints, without loading models.
    protected = {path: digest(path) for path in artifacts.rglob("*") if path.is_file()}
    protected[workbook] = digest(workbook)
    for path in [manifests / "dataset_audit.json", manifests / "root_usable.csv", manifests / "leaf_usable.csv"]:
        protected[path] = digest(path)
    output.mkdir(parents=True, exist_ok=False)
    if refresh:
        snapshot = output / "verified_source_snapshot"
        snapshot.mkdir()
        for task, frame in refreshed.items():
            frame.to_csv(snapshot / f"{task}_manifest.csv", index=False)
            frame[frame.status.eq("usable")].to_csv(snapshot / f"{task}_usable.csv", index=False)
            frame[frame.status.eq("excluded")].to_csv(snapshot / f"{task}_excluded.csv", index=False)
        (snapshot / "dataset_audit.json").write_text(json.dumps(current_audit, indent=2), encoding="utf-8")
    summary = {"positive_classes": {task: labels[1] for task, labels in CLASSES.items()},
               "workbook_sha256": current_hash, "original_source_audit_workbook_sha256": audit["workbook_sha256"],
               "source_snapshot_regenerated": refresh, "tasks": {},
               "provenance": {str(path.relative_to(root)): value for path, value in protected.items()},
               "score_note": "Model scores are not calibrated probabilities of correctness."}
    for task in CLASSES:
        task_output = output / task
        task_output.mkdir()
        prediction_path = artifacts / task / "out_of_fold_predictions.csv"
        evaluation = json.loads((artifacts / task / "evaluation.json").read_text())
        if evaluation["class_names"] != CLASSES[task]:
            raise ValueError("Experiment class mapping differs from supported task classes")
        predictions = pd.read_csv(prediction_path, keep_default_na=False)
        source = (refreshed[task][refreshed[task].status.eq("usable")].copy() if refresh
                  else pd.read_csv(manifests / f"{task}_usable.csv", keep_default_na=False))
        frame = analyze_predictions(predictions, source, task, root)
        plant, labels, pooled, counts = summarize(frame)
        source_keys = set(source.sample_key)
        counts["verified_samples_without_predictions"] = sorted(source_keys - set(frame.sample_key))
        counts["baseline_metrics"] = evaluation["oof_metrics"]
        counts["experiment_workbook_sha256"] = evaluation["workbook_sha256"]
        counts["experiment_workbook_matches_current"] = evaluation["workbook_sha256"] == current_hash
        counts["input_columns"] = list(predictions.columns)
        observed_matrix = [[counts["counts"]["TN"], counts["counts"]["FP"]], [counts["counts"]["FN"], counts["counts"]["TP"]]]
        counts["confusion_matches_saved_evaluation"] = observed_matrix == evaluation["oof_metrics"]["confusion_matrix"]
        verified_errors = frame[frame.match_status.eq("verified") & ~frame.correct]
        error_pages, error_links = contact_sheets(verified_errors, task_output / "contact_sheets", "misclassified")
        priority_pages, priority_links = contact_sheets(verified_errors[verified_errors.error_type.eq("FN")], task_output / "contact_sheets", "priority_false_negatives")
        frame["error_contact_sheet"] = frame.sample_key.map(error_links).fillna("")
        frame["priority_contact_sheet"] = frame.sample_key.map(priority_links).fillna("")
        frame.to_csv(task_output / "error_analysis.csv", index=False)
        frame[frame.review_priority.le(2)].to_csv(task_output / "review_queue.csv", index=False)
        frame[frame.match_status.ne("verified")].to_csv(task_output / "validation_issues.csv", index=False)
        plant.to_csv(task_output / "summary_by_plant.csv", index=False)
        labels.to_csv(task_output / "summary_by_plant_and_labels.csv", index=False)
        pooled.to_csv(task_output / "summary_by_labels.csv", index=False)
        notes = frame[["sample_key", "Image_ID", "expert_label", "predicted_label", "error_type"]].copy()
        for column in ["visual_observation", "annotation_ambiguity", "hypothesis", "suggested_action", "reviewer"]:
            notes[column] = ""
        notes.to_csv(task_output / "review_notes.csv", index=False)
        counts["contact_sheets"] = {"errors": error_pages, "priority_false_negatives": priority_pages}
        summary["tasks"][task] = counts
    if any(digest(path) != value for path, value in protected.items()):
        raise RuntimeError("A protected input changed during analysis")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ["# FloraSense C3 baseline v1 error analysis", "", "## Observations from saved records", "",
              "Positive class = Not Acceptable (root), One Side (leaf). FN means the expert marked the positive class but the model predicted the negative class.", ""]
    for task, counts in summary["tasks"].items():
        report.append(f"- {task}: {counts['verified']} verified predictions, {counts['correct']} correct, {counts['counts']['FN']} FN and {counts['counts']['FP']} FP; {counts['unverified']} unverified. Saved confusion matrix matches: {counts['confusion_matches_saved_evaluation']}.")
    report.append(f"\nCurrent workbook file hash differs from the original source audit: {refresh}. An isolated verified_source_snapshot was regenerated when needed; original manifests were preserved. Experiment-to-current labels, image IDs, views, and content hashes were checked per row; changed records are unverified, not silently reconciled.")
    report += ["", "Images were joined on Plant_ID + Sample_ID; image IDs, task views, expert labels and SHA-256 content were checked. Colab image_path values were retained only for provenance. No checkpoint was loaded or inference run.",
        "", "## Priority review", "", "Open each task's contact_sheets/priority_false_negatives_001.jpg first. Then use review_queue.csv and the remaining misclassified sheets. Rows are ordered by FN before FP, then by descending predicted-label score. A wrong confident prediction is not proof of a wrong annotation. Correct samples remain in error_analysis.csv.",
        "", "## Hypotheses to investigate (not established findings)", "",
        "- A single root view may conceal root coverage behind the pot, medium, or other roots; confirm whether the expert label describes visible roots or the entire root system.",
        "- Leaf distribution may depend on the stem centerline, viewpoint, occlusion, and the definition of a side. Review the expert convention for borderline or rotated plants.",
        "- Background/plant-type cues and loss of fine structure at 224 pixels may contribute to errors. These require controlled follow-up, not conclusions from prediction disagreement alone.",
        "", "## Visual observations and annotation review", "",
        "Use review_notes.csv to record visible facts separately from annotation ambiguity and causal hypotheses. Optional inspection notes are in visual_review.md when supplied. No annotation is automatically corrected. Error contact sheets show full, aspect-preserved views rather than unmarked crops.",
        "", "## Limits", "", "These are out-of-fold errors from Experiment 1, not final-model inference or an independent external test set. Reviewing them informs development and must not be presented as a new untouched evaluation. Minority support is small. No P08 evidence exists; P05 has no expert One Side examples. Exact hashes do not detect every near-duplicate or cross-ID recapture. Checkpoint files, workbook, API, frontend and rule engine are not modified.", ""]
    (output / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/quality_assessment/processed/error_analysis/baseline_v1" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args(argv)
    if not args.output_dir.resolve().is_relative_to((ROOT / "data/quality_assessment/processed").resolve()):
        parser.error("Analysis output must stay in the ignored processed-data directory")
    summary = run_analysis(args.artifacts.resolve(), args.manifest_dir.resolve(), args.output_dir.resolve())
    print(json.dumps({task: {key: value[key] for key in ["verified", "correct", "errors", "counts", "unverified"]} for task, value in summary["tasks"].items()}, indent=2))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
