# C3 feature dataset version history

## c3_features_v2_20261022 — corrected Experiment 2 ground truth

Released 2026-10-22 from the current Excel annotations. The researcher confirmed
that the three previously identified root revisions intentionally correct
annotation mistakes. Composite keys resolve the P06/P07/P07 confirmation to:

| Sample | Feature | Experiment 1 label | Corrected label | Status |
|---|---|---|---|---|
| P06:S007 | Root_Distribution | Acceptable | Not Acceptable | Researcher-confirmed correction |
| P07:S002 | Root_Distribution | Not Acceptable | Acceptable | Researcher-confirmed correction |
| P07:S013 | Root_Distribution | Not Acceptable | Acceptable | Researcher-confirmed correction |

The current workbook is ground truth for v2. It is copied byte-for-byte into the
new snapshot for provenance; the working workbook is not edited. See release.json
for the actual workbook hash, audit, class support and split-source hashes.
Other workbook fields are preserved as supplied; general-quality labels are
not used to infer root/leaf distribution.

- Root: 104 samples, 90 Acceptable / 14 Not Acceptable.
- Leaf: 90 samples, 74 Both Sides / 16 One Side; all feature targets unchanged
  from Experiment 1.
- Image/sample population, image IDs and image content are unchanged from v1.
- Exact saved train/validation/outer-test assignments are reused with v2 targets;
  all six source CSVs remain untouched.
- Release directory:
  `data/quality_assessment/processed/experiment2/v2_20261022/`.
- ZIP: `c3_features_v2_20261022.zip`. Images and input CSV paths are portable
  relative to the extracted ZIP root. No training code/notebook is included.
- Builder: `python -B -m services.c3_quality_assessment.prepare_experiment2_snapshot`.
  It refuses to overwrite an existing release or accept unexpected target changes.

Corrected P07 has no Not Acceptable roots. P05 has no One Side leaf examples;
P08 has no verified feature dataset. Six existing One Side Front samples retain
their expert labels while branch-origin visibility/qualification needs review:
P02:S012, P04:S003, P06:S004, P06:S013, P07:S002, P07:S013. These flags are not
automatic relabeling or post-hoc exclusion instructions.

For a controlled corrected-root comparison, establish A-v2 and B-v2 using the
same v2 targets and saved groups. Stored A-v1 metrics remain historical. Merely
re-scoring A-v1 on revised targets is a sensitivity analysis, not model gain.
Future training must explicitly load the supplied split CSVs; the previous
generic training entry point's split generation is not this v2 protocol.

## Experiment 1 original feature snapshot — preserved

- Ground-truth workbook hash in the original snapshot:
  `e72dfe03687cc9795b114be49ceaba2751525e9fe3c0569d3bc5fab3ea77e6c1`.
- Original ZIP:
  `data/quality_assessment/processed/colab/florasense_c3_training.zip`.
- Original ZIP hash:
  `4cae788b6d48b52d6203d85b46ed59328de7544e997d72376124d927a8f95015`.
- Root: 104 samples, 89 Acceptable / 15 Not Acceptable.
- Leaf: 90 samples, 74 Both Sides / 16 One Side.
- Metrics, OOF predictions, saved splits and checkpoints:
  `models/checkpoints/c3_features/baseline_v1/{root,leaf}/`.
- Root macro-F1 0.4793993, balanced accuracy 0.5037453; leaf macro-F1
  0.4916655, balanced accuracy 0.5760135. These were not replaced or relabeled.

The prior analysis/HEAD workbook hash `1cadfcbe…` differed from the original
snapshot hash, but its feature targets matched the original release. The current
three root corrections are semantic revisions, not merely a file-format hash
change. Original raw Experiment 1 workbook bytes are not available as a separate
inspected file; its frozen feature manifests and image hashes remain available
in the original ZIP. No attempt was made to reconstruct or overwrite v1.
