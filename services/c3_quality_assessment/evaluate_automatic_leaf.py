"""Reproducible descriptive evaluation; workbook values never enter image inference."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from .automatic_leaf import ALGORITHM_VERSION, analyze_image, decode_image
from .data_loader import ROOT, load_dataset, resolve_image_path


def valid_count(value):
    try:
        number = float(value)
        return math.isfinite(number) and number >= 0 and number.is_integer()
    except (TypeError, ValueError):
        return False


def evaluate(output_dir: Path, report_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sheets = load_dataset()
    images = sheets["Images"]
    p01 = images[images.Plant_ID.astype(str).str.strip() == "P01"]
    annotations = sheets["Quality_Annotations"]
    inventory = []
    for row in p01.itertuples():
        path = resolve_image_path(row.Relative_Path)
        with Image.open(path) as image:
            inventory.append({"sample_id": row.Sample_ID, "image_id": row.Image_ID,
                              "view": row.View_Type, "path": str(row.Relative_Path),
                              "stored_dimensions": list(image.size), "bytes": path.stat().st_size})
    (report_dir / "p01_image_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    results = []
    for row in p01[p01.View_Type.astype(str).str.strip() == "Front"].itertuples():
        path = resolve_image_path(row.Relative_Path)
        payload = path.read_bytes()
        image, size = decode_image(payload, "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png")
        prediction = analyze_image(image, size, grid_profile_confirmed=True)  # No ground-truth count or requirement passed here.
        (output_dir / f"{row.Sample_ID}_{row.Image_ID}_preview.jpg").write_bytes(
            base64.b64decode(prediction.pop("annotated_preview").split(",", 1)[1]))
        (output_dir / f"{row.Sample_ID}_{row.Image_ID}.json").write_text(json.dumps(prediction, indent=2), encoding="utf-8")
        ground_truth_rows = annotations[(annotations.Sample_ID == row.Sample_ID) & (annotations.Plant_ID == "P01")]
        ground_truth = ground_truth_rows.iloc[0].Total_Leaf_Count if len(ground_truth_rows) == 1 else None
        ground_truth = int(ground_truth) if valid_count(ground_truth) else None
        count = prediction["estimated_count"]
        other_samples = [str(other.Sample_ID) for other in p01.itertuples()
                         if other.Sample_ID != row.Sample_ID
                         and resolve_image_path(other.Relative_Path).resolve() == path.resolve()]
        issues = []
        if other_samples:
            issues.append("shared image path with " + ",".join(sorted(set(other_samples))))
        if "root" in path.stem.lower():
            issues.append("front record references root-view filename")
        results.append({"sample_id": row.Sample_ID, "image_id": row.Image_ID,
                        "predicted_count": count, "ground_truth_count": ground_truth,
                        "absolute_error": abs(count - ground_truth) if count is not None and ground_truth is not None else None,
                        "review_status": prediction["status"], "count_reliable": prediction["count_reliable"],
                        "calibration_status": prediction["calibration"]["status"],
                        "linkage_issue": "; ".join(issues), "sha256": hashlib.sha256(payload).hexdigest()})
    csv_path = report_dir / "p01_leaf_count_evaluation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]) if results else [])
        writer.writeheader()
        writer.writerows(results)
    scored = [row for row in results if row["absolute_error"] is not None]
    clean = [row for row in scored if not row["linkage_issue"]]
    summary = {"algorithm_version": ALGORITHM_VERSION, "front_records": len(results),
               "valid_ground_truth_records": sum(row["ground_truth_count"] is not None for row in results),
               "numeric_predictions_scored": len(scored),
               "mae_all_workbook_front_records": sum(row["absolute_error"] for row in scored) / len(scored) if scored else None,
               "unflagged_front_records_scored": len(clean),
               "mae_excluding_flagged_linkages": sum(row["absolute_error"] for row in clean) / len(clean) if clean else None,
               "reliable_counts": sum(row["count_reliable"] for row in results),
               "manual_review_records": sum(row["review_status"] == "MANUAL REVIEW" for row in results),
               "grid_scale_candidates": sum(row["calibration_status"] == "grid_scale_candidate" for row in results),
               "physical_grid_spacing_supplied": True, "major_grid_spacing_cm": 1.0, "minor_grid_spacing_cm": 0.2, "trained_model_used": False}
    (report_dir / "p01_leaf_count_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# P01 Phase 1 leaf-count baseline evaluation", "",
             "Algorithm: `" + ALGORITHM_VERSION + "`. Deterministic, untrained OpenCV baseline; parameters were not fitted to the workbook counts.", "",
             "Ground truth: unique finite non-negative integer `Quality_Annotations.Total_Leaf_Count` for each P01 Sample_ID. These are sample-wide manual counts, not verified front-view visible-leaf instance labels. Height_Assessments totals are buyer-specific assessment denominators, not leaf-count labels. Existing S005 sessions contain manually selected endpoint measurements and do not label every visible leaf. No masks, boxes or leaf endpoints are available as automatic-detection ground truth.", "",
             "All available workbook-linked Front records are evaluated once. Side/Focus/Root views are inventoried, not counted as independent plants. No training or random image split occurred; any future trained-model evaluation must group all views, crops and augmentations by plant/Sample_ID before splitting. New captures of the same physical plant must remain in that group.", "",
             "## Results", "", "```json", json.dumps(summary, indent=2), "```", "",
             "MAE is the mean absolute count error over numeric predictions with valid ground-truth counts only. Abstentions are excluded and coverage is reported above. The all-record metric retains the workbook's S007 Front linkage; the separate unflagged metric excludes known linkage issues. Neither metric measures reliable/accepted count accuracy: all counts require review.", "",
             "| Sample | Predicted candidate count | Manual count | Absolute error | Status | Linkage issue |",
             "|---|---:|---:|---:|---|---|"]
    for row in results:
        lines.append("| " + " | ".join(str(row[key]) if row[key] is not None else "unavailable" for key in
                     ["sample_id", "predicted_count", "ground_truth_count", "absolute_error", "review_status", "linkage_issue"]) + " |")
    lines.extend(["", "## Interpretation and failure cases", "",
                  "The baseline is not suitable for accepted all-visible-leaf counting. Overlapping leaves merge into connected vegetation; distance maxima split elongated or variegated leaves into duplicate fragments. Low-contrast and small/partial leaves can disappear at the 1400 px processing resolution. Green-tinted background, grid/ruler edges and stems can generate extra regions. Region labels partition pixels without duplicate boxes from overlapping proposals, but this does not establish one region per biological leaf. Border, small-region, irregularity and threshold-sensitivity flags are heuristic warnings, not calibrated confidence.", "",
                  "S007 is a known invalid front-image association: its Front record references P01_S006_root.jpg, also linked to S006 Root. This was not repaired or silently excluded from the raw table. Human source-data correction and front-view relabeling are needed before model evaluation.", "",
                  "Graph periodicity is sought independently in both axes across multiple background tiles. Physical spacing is not in the workbook; the user confirmed 1 cm major lines with five 0.2 cm minor intervals. The evaluation uses this confirmed profile. A five-to-one thickness/contrast hierarchy must identify the major lines before assigning scale; equal periodic lines alone are rejected. Rotation is estimated from background line orientations. Inconsistent spacing across tiles or axes is rejected as possible perspective distortion rather than corrected with an unverified homography. Plant/background coplanarity remains unverified. No automatic leaf heights or buyer quality decisions are generated.", "",
                  "## Next development step", "",
                  "Collect front-view per-leaf instance masks, including small/partially visible leaves, with consistent occlusion rules; annotate emergence/tip keypoints separately and verify physical graph spacing/capture geometry. Use annotation assistance followed by human review. Train/fine-tune an instance-segmentation model (for example Mask R-CNN), keeping every view of one plant in the same train/validation/test group. Compare per-plant count MAE, detection precision/recall and small/occluded-leaf recall on held-out plants; calibrate abstention thresholds on validation plants only. Nineteen plant records are insufficient for a convincing generalization claim. Retain manual review until this evidence supports accepted counts.", "",
                  "Mask R-CNN primary paper: https://arxiv.org/abs/1703.06870 . This is a recommendation; no detector was trained or downloaded in Phase 1.", "",
                  "Generated inventory and CSV are alongside this report. Annotated previews and complete per-image diagnostic JSON are under `data/quality_assessment/processed/p01_leaf_prototype/` by default. Original images and workbook are read only."])
    (report_dir / "phase1_leaf_count_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/quality_assessment/processed/p01_leaf_prototype")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/quality_assessment")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output_dir, args.report_dir), indent=2))
