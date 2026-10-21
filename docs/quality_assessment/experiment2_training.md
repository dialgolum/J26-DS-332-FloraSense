# Experiment 2 Colab training support

No local training is permitted. The runner rejects Windows, non-Colab and
non-CUDA execution before creating models, downloading weights or writing runs.
Local tests instantiate random untrained ResNet18s only to check parameter
configuration and checkpoint serialization; they perform no optimizer steps.

## Files

- Runner: `services/c3_quality_assessment/experiment2_training.py`.
- Notebook: `services/c3_quality_assessment/notebooks/c3_experiment2_colab.ipynb`.
- Support builder: `python -B -m services.c3_quality_assessment.build_experiment2_colab`.
- Generated support ZIP:
  `data/quality_assessment/processed/experiment2/colab_support/c3_experiment2_support.zip`.
- Immutable dataset ZIP:
  `data/quality_assessment/processed/experiment2/v2_20261022/c3_features_v2_20261022.zip`.

Rebuild the support ZIP/notebook together after changing the runner: their hashes
are pinned in the notebook. The builder never rewrites the dataset ZIP. Dataset
validation verifies every checksum, composite identity, view, applicability,
release supports, saved split/manifest row equality, exact image/sample
disjointness and once-only outer test coverage. It never generates a fold.

## Exact execution order

1. Upload both ZIPs to **My Drive / FloraSense_C3 / experiment2** without renaming.
2. Open Colab, choose **File > Upload notebook**, and upload the notebook above.
3. Choose **Runtime > Change runtime type > T4 GPU**, then connect to a hosted
   runtime. Another available CUDA GPU is accepted and recorded; not a local runtime.
4. Run cell **1 Mount Drive** and authorize access. Adjust DRIVE_DIR if needed.
5. Run **2 Validate archives and extract**. Both ZIP hashes must match the
   notebook; files extract to fresh `/content` directories.
6. Run **3 Dependencies and hosted GPU**. Colab's matched torch/torchvision CUDA
   packages are preserved; evaluation dependencies are installed separately.
7. Run **4 Validate all V2 data and six saved partitions**. Check the printed
   corrected-root and unchanged-leaf supports and output RUN_ROOT.
8. Run **5 Architecture A**: frozen ImageNet ResNet18 plus head, root then leaf.
9. Run **6 Architecture B**: ImageNet ResNet18 layer4 plus head, root then leaf.
10. Run **7 Compare**, then **8 Export**. A/B comparison rejects inconsistent
    target versions, preprocessing/config, split hashes and OOF sample identities.

Results persist to `My Drive/FloraSense_C3/experiment2/runs/<timestamp>/`:

```text
A/root/  A/leaf/  B/root/  B/leaf/
    fold_1.pth, fold_2.pth, fold_3.pth
    fold_1_split.csv, fold_2_split.csv, fold_3_split.csv
    out_of_fold_predictions.csv, evaluation.json, per_plant.csv
    confusion_matrix.png
model_comparison.csv, comparison_report.md
root_paired_comparison.csv, leaf_paired_comparison.csv
```

The export cell downloads `c3_experiment2_models.zip` and backs it up to Drive.
Existing arms refuse overwrite. There is no optimizer-state resume; if interrupted,
retain the partial run for inspection and restart with a new timestamp. No fold
is selected by test scores and no full-data deployment refit is produced.

## Controlled protocol

A trains fc only; B trains layer4 and fc from the same ImageNet initialization,
not from Experiment 1's full-data model. BatchNorm running statistics stay frozen.
Both use the exact exported train/validation/test rows for every task/fold; no
StratifiedGroupKFold or train_test_split is called. Plant_ID namespaces Sample_ID
but is not supplied as a learned feature. Varieties can occur across partitions,
as in the original protocol: this is not a held-out-variety test. Root and leaf
have independent weights and assignments; no shared multi-task fitting occurs.
Cross-ID physical recaptures/near-duplicates cannot be ruled out by exact hashes.

Both arms use identical cached image tensors: original + two fixed augmentations,
seeds 42 + image index. EXIF orientation, JPEG draft decode, aspect-preserving
208-pixel contain, centered gray padding to 224, ImageNet normalization. Training
views use horizontal flips, +/-4 degree rotation and .08 brightness/contrast/
saturation jitter. No crops or vertical flips; evaluation uses only originals.
Unlike frozen feature extraction, B never caches learned encoder outputs across
epochs/folds. Every fold starts afresh from ImageNet. No test images supply loss,
weights, BatchNorm statistics or checkpoint selection.

Training-only inverse-frequency weights; AdamW, batch 16, head LR .001, layer4
LR .00001, weight decay .01, max 40 epochs, patience 6 and min_delta .00001.
B starts layer4 training at epoch 1, without the optional proposal warm-up; this
is explicit in the notebook. Validation weighted loss is aggregated by sum of
training-derived class weights, selects checkpoints, and triggers early stopping.
Test evaluation follows checkpoint selection with fixed argmax. Seeds, versions,
GPU, split and manifest hashes are saved. GPU numerical equality across versions
is not guaranteed; fixed partitions and paired predictions support comparison.

Outputs include fold/pooled confusion matrices in JSON, pooled PNGs, class
precision/recall/support, macro-F1, balanced accuracy, minority precision/recall,
majority reference, per-plant metrics and paired A/B differences. Missing-class
per-plant recall/balanced accuracy is unavailable rather than a fabricated zero.
Corrected P07 has no unacceptable roots, P05 no One Side leaves, and P08 no data.
Six branch-origin visibility/qualification flags remain; no targets are changed
or difficult outer samples excluded. Previously reviewed outer data is a
comparative benchmark, not a new untouched external evaluation. A-v2 is trained
on corrected root targets; old A-v1 metrics must not substitute for this control.

## Checkpoint compatibility and future inference

Each checkpoint contains a full ordinary ResNet18 `model_state_dict`, two-class
`class_names`, `class_to_idx`, image size, preprocessing, task, A/B identity,
dataset/workbook/split/manifest provenance and selected epoch. Serialization uses
CPU tensors and weights_only-compatible metadata. A synthetic roundtrip test
loads it through the existing `model_inference.load_quality_model()` with only
the test's checkpoint path/device patched. No existing real model is changed.

The current inference module still hardcodes the **general-quality** path and
224x224 direct resizing, so it is **not ready to serve these feature checkpoints
unchanged**. Future integration must accept explicit feature model paths, read
task class order, and reproduce EXIF + contain/pad + normalization (including
documented JPEG decode) from metadata. Keep root/leaf result semantics separate
from general quality and buyer rules. Do not replace quality_resnet18.pth. API,
frontend and rule-engine changes remain outside this implementation.

## Local checks (no training)

```powershell
python -B -m unittest services.c3_quality_assessment.tests.test_experiment2_training -v
```

Notebook code cells are syntax/schema validated without execution. Synthetic
fixtures exercise fixed partitions, tampering, leakage and seed handling. Actual
V2 archives are validated without a model forward pass. The real CUDA training
path remains unexecuted until the researcher runs Colab.
