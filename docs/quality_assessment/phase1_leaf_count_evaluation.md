# P01 Phase 1 leaf-count baseline evaluation

Algorithm: `p01-opencv-baseline-v1`. Deterministic, untrained OpenCV baseline; parameters were not fitted to the workbook counts.

Ground truth: unique finite non-negative integer `Quality_Annotations.Total_Leaf_Count` for each P01 Sample_ID. These are sample-wide manual counts, not verified front-view visible-leaf instance labels. Height_Assessments totals are buyer-specific assessment denominators, not leaf-count labels. Existing S005 sessions contain manually selected endpoint measurements and do not label every visible leaf. No masks, boxes or leaf endpoints are available as automatic-detection ground truth.

All available workbook-linked Front records are evaluated once. Side/Focus/Root views are inventoried, not counted as independent plants. No training or random image split occurred; any future trained-model evaluation must group all views, crops and augmentations by plant/Sample_ID before splitting. New captures of the same physical plant must remain in that group.

## Results

```json
{
  "algorithm_version": "p01-opencv-baseline-v1",
  "front_records": 19,
  "valid_ground_truth_records": 19,
  "numeric_predictions_scored": 16,
  "mae_all_workbook_front_records": 94.1875,
  "unflagged_front_records_scored": 16,
  "mae_excluding_flagged_linkages": 94.1875,
  "reliable_counts": 0,
  "manual_review_records": 19,
  "grid_scale_candidates": 0,
  "physical_grid_spacing_supplied": true,
  "major_grid_spacing_cm": 1.0,
  "minor_grid_spacing_cm": 0.2,
  "trained_model_used": false
}
```

MAE is the mean absolute count error over numeric predictions with valid ground-truth counts only. Abstentions are excluded and coverage is reported above. The all-record metric retains the workbook's S007 Front linkage; the separate unflagged metric excludes known linkage issues. Neither metric measures reliable/accepted count accuracy: all counts require review.

| Sample | Predicted candidate count | Manual count | Absolute error | Status | Linkage issue |
|---|---:|---:|---:|---|---|
| S001 | 88 | 8 | 80 | MANUAL REVIEW |  |
| S002 | 89 | 9 | 80 | MANUAL REVIEW |  |
| S003 | 161 | 9 | 152 | MANUAL REVIEW |  |
| S004 | unavailable | 8 | unavailable | MANUAL REVIEW |  |
| S005 | 38 | 11 | 27 | MANUAL REVIEW |  |
| S006 | 92 | 8 | 84 | MANUAL REVIEW |  |
| S007 | unavailable | 8 | unavailable | MANUAL REVIEW | shared image path with S006; front record references root-view filename |
| S008 | 144 | 9 | 135 | MANUAL REVIEW |  |
| S009 | 86 | 9 | 77 | MANUAL REVIEW |  |
| S010 | 86 | 8 | 78 | MANUAL REVIEW |  |
| S011 | 89 | 10 | 79 | MANUAL REVIEW |  |
| S012 | 100 | 7 | 93 | MANUAL REVIEW |  |
| S013 | 65 | 9 | 56 | MANUAL REVIEW |  |
| S014 | 158 | 10 | 148 | MANUAL REVIEW |  |
| S015 | unavailable | 8 | unavailable | MANUAL REVIEW |  |
| S016 | 100 | 8 | 92 | MANUAL REVIEW |  |
| S017 | 70 | 6 | 64 | MANUAL REVIEW |  |
| S018 | 154 | 8 | 146 | MANUAL REVIEW |  |
| S019 | 122 | 6 | 116 | MANUAL REVIEW |  |

## Interpretation and failure cases

The baseline is not suitable for accepted all-visible-leaf counting. Overlapping leaves merge into connected vegetation; distance maxima split elongated or variegated leaves into duplicate fragments. Low-contrast and small/partial leaves can disappear at the 1400 px processing resolution. Green-tinted background, grid/ruler edges and stems can generate extra regions. Region labels partition pixels without duplicate boxes from overlapping proposals, but this does not establish one region per biological leaf. Border, small-region, irregularity and threshold-sensitivity flags are heuristic warnings, not calibrated confidence.

S007 is a known invalid front-image association: its Front record references P01_S006_root.jpg, also linked to S006 Root. This was not repaired or silently excluded from the raw table. Human source-data correction and front-view relabeling are needed before model evaluation.

Graph periodicity is sought independently in both axes across multiple background tiles. Physical spacing is not in the workbook; the user confirmed 1 cm major lines with five 0.2 cm minor intervals. The evaluation uses this confirmed profile. A five-to-one thickness/contrast hierarchy must identify the major lines before assigning scale; equal periodic lines alone are rejected. Rotation is estimated from background line orientations. Inconsistent spacing across tiles or axes is rejected as possible perspective distortion rather than corrected with an unverified homography. Plant/background coplanarity remains unverified. No automatic leaf heights or buyer quality decisions are generated.

## Next development step

Collect front-view per-leaf instance masks, including small/partially visible leaves, with consistent occlusion rules; annotate emergence/tip keypoints separately and verify physical graph spacing/capture geometry. Use annotation assistance followed by human review. Train/fine-tune an instance-segmentation model (for example Mask R-CNN), keeping every view of one plant in the same train/validation/test group. Compare per-plant count MAE, detection precision/recall and small/occluded-leaf recall on held-out plants; calibrate abstention thresholds on validation plants only. Nineteen plant records are insufficient for a convincing generalization claim. Retain manual review until this evidence supports accepted counts.

Mask R-CNN primary paper: https://arxiv.org/abs/1703.06870 . This is a recommendation; no detector was trained or downloaded in Phase 1.

Generated inventory and CSV are alongside this report. Annotated previews and complete per-image diagnostic JSON are under `data/quality_assessment/processed/p01_leaf_prototype/` by default. Original images and workbook are read only.
