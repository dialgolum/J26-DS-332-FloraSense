# C3 saved-prediction error analysis

## Current annotation interpretation and continued analysis

Leaf distribution is governed by [annotation_guidelines.md](annotation_guidelines.md):
**branch origins relative to the main stem**, not foliage balance. Earlier visual
reviews asking whether the silhouette defines sides are historical; this is now
clarified. A photograph may conceal junctions without invalidating its expert
sample label. No learned-feature attribution follows from model disagreement.

The plant-wise/branch-origin continuation of Experiment 1 is saved under
`data/quality_assessment/processed/error_analysis/baseline_v1/review_20261021/plant_branch_review_20261021/`.
It reuses existing OOF predictions and saved folds; it does not rerun the original
workflow. Read findings.md, workbook_comparability.md, and experiment2_proposal.md
there before planning any training. Root labels have changed in the working
workbook since Experiment 1, so frozen-v1 metrics and current-label sensitivity
must remain separate. Nothing in this supplement authorizes training or changes
Excel, model checkpoints, API, rule engine or frontend.

## Original workflow

Run from the repository root:

```powershell
python -B -m services.c3_quality_assessment.analyze_feature_errors
python -B -m unittest services.c3_quality_assessment.tests.test_feature_error_analysis -v
```

The default inputs are `models/checkpoints/c3_features/baseline_v1/{root,leaf}/`
and `data/quality_assessment/processed/feature_classification/`. Every run creates
a fresh timestamped directory under the already ignored
`data/quality_assessment/processed/error_analysis/baseline_v1/` directory. Override
with `--artifacts`, `--manifest-dir`, or `--output-dir`; CLI outputs must stay
inside ignored processed data and the output directory must not already exist.
Original artifacts are preserved, including their ZIP, reports and checkpoints.

No PyTorch import, checkpoint loading, inference, optimizer step or training is
involved. The workflow checks the audit against the workbook hash. If the hash
differs, it reruns existing preparation into an isolated `verified_source_snapshot`
inside the new output directory, preserving the original manifests. It records
experiment/current workbook hashes, validates class order, and joins predictions to local
verified image records using **Plant_ID + Sample_ID**. It checks sample_key,
Image_ID, view, labels, indices, score ranges/decisions, source hashes, existence,
and image headers. Existing source manifests were generated with full decoding;
hash equality verifies unchanged content. Contact-sheet images are decoded for
display. Colab image paths are recorded for provenance but never used for lookup.
The experiment's portable-manifest checksum need not equal the Windows-manifest
checksum because paths differ; per-record image hashes and labels are compared.
File-level workbook drift is not assumed to mean semantic label changes. Any
actual per-record label/content change is explicitly excluded from analysis.

Missing/changed sources, inconsistent indices or labels, and invalid predictions
are retained with `match_status=unverified` and explicit validation issues, excluded
from summary counts and sheets, and placed first in the review queue. Duplicate
composite keys fail closed. `summary.json` reports missing prediction coverage and
whether verified counts reproduce the saved confusion matrix; inspect these
before interpreting error totals. It also records protected input SHA-256 values.

Positive means **Not Acceptable** for root and **One Side** for leaf:

| Code | Expert label | Prediction |
|---|---|---|
| FN | Positive | Negative |
| FP | Negative | Positive |
| TP | Positive | Positive |
| TN | Negative | Negative |

Review each task's outputs in this order:

1. `contact_sheets/priority_false_negatives_001.jpg`: root unacceptable samples
   predicted acceptable, or leaf One Side samples predicted Both Sides. Expert
   label, predicted label, fold, sample key, and minority score appear on each tile.
2. `review_queue.csv`: FN first, FP second; descending wrong-label model score
   within each group. Integrity problems have priority 0. Scores are uncalibrated.
3. Other `contact_sheets/misclassified_*.jpg` pages: all verified errors, including
   priority FNs. The CSV's contact-sheet columns link samples to pages.
4. `error_analysis.csv`: every prediction, including correctly classified TP/TN.
5. `summary_by_plant.csv`, `summary_by_plant_and_labels.csv`, and
   `summary_by_labels.csv`: counts/rates by plant and label pair.
6. `validation_issues.csv`, `summary.json`, and `analysis_report.md`: linkage,
   coverage, aggregate findings, review questions and limitations.
7. `review_notes.csv`: record directly visible facts in `visual_observation`,
   expert-policy uncertainty in `annotation_ambiguity`, and untested explanations
   in `hypothesis`. Record a reviewer and suggested action. Optional manually
   inspected examples are documented in `visual_review.md` alongside the report.

Do not change Excel because the model disagrees. An image can be difficult while
the expert label is correct. Confirm the root-coverage criterion and leaf side/
centerline convention, particularly when parts are obscured. Do not infer these
feature labels from overall plant acceptance.

The OOF CSV format currently includes Plant_ID, Sample_ID, sample_key, Image_ID,
view_type, image_path, label, raw_label, sha256, status, exclusion_reason, fold,
true_index, predicted_index, prediction, and minority_probability. The associated
evaluation.json declares class order and workbook provenance. A format change
requires explicit validation rather than positional column interpretation.

These reports analyze Experiment 1's out-of-fold predictions; they do not run the
full-data refit model. After reviewing OOF errors, the same data is development
evidence, not a fresh untouched test set. P08 has no evidence; P05 has no One Side
ground truth. No changes to annotations, models, rule engine, P01 workflows, API,
or frontend are made. Generated photos and reports stay outside version control.
