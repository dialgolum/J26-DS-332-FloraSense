# C3 Colab training and interruption recovery

## Recovery findings — 2026-10-21

Branch: `feature/c3-leaf-detection`. No reset, revert, commit, push, or merge was
performed. The workbook and deleted Excel lock file were pre-existing user
changes; original images and the workbook were preserved.

No Python process remained from the interrupted attempt. The only local attempt
artifact was `models/checkpoints/c3_features/initial_20261021/error.json`, recording
a refused ImageNet weight-download connection. No feature extraction, optimizer
training, fold checkpoint, final checkpoint, or partial weight download was found.
The error log is retained as useful recovery evidence.

The original general-quality checkpoint SHA-256 still equals:

```text
85de24117754379b133e8fccbdbea787600756fcd5df301b8334e556a430ba42
```

The rule engine, P01 implementation, FastAPI
directory, and frontend match their tracked versions. This does not claim that
pre-existing globally ambiguous Sample_ID API lookups have been fixed.

All 15 corrected P02 S012–S016 Front/Stem/Root references pass path, plant,
sample, view, existence, readability, and duplicate checks. The added
`P02_S002_front.jpg` and `P05_S015_stem.jpg` exist and decode. The preparation
implementation releases corrected P02 links conditionally; invalid paths,
duplicates, unknown labels, and conflicting annotations remain excluded.

Current prepared counts:

| Task | Usable samples/images | Classes | Excluded records |
|---|---:|---|---:|
| Root | 104 | Acceptable 89; Not Acceptable 15 | 14 |
| Leaf | 90 | Both Sides 74; One Side 16 | 9 |

Root exclusions are five P01 samples without Root records plus nine P08 samples
without images/annotations. Leaf exclusions are the nine P08 samples. There are
no missing referenced image files, filename mismatches, shared references, or
conflicting task labels in the current audit. P05 has no One Side examples.

## Reused and changed files

- Reused `prepare_feature_data.py`, its tests, the data loader, source images,
  workbook, and generated feature manifests/audit. Regeneration remains local,
  read-only with respect to the sources.
- Adapted the interrupted `train_feature_classifier.py` for portable manifests
  and Colab CUDA execution. Windows training fails before weights are downloaded
  or output directories are created. No local training is needed or permitted.
- Added `build_colab_bundle.py`, packaging/notebook tests, this guide, and
  `notebooks/c3_feature_training_colab.ipynb`.
- Generated the ZIP and its `.sha256` file under
  `data/quality_assessment/processed/colab/` (ignored by Git).

## Rebuild the handoff locally (preparation only)

```powershell
python -B -m services.c3_quality_assessment.prepare_feature_data
python -B -m services.c3_quality_assessment.build_colab_bundle
python -B -m unittest services.c3_quality_assessment.tests.test_prepare_feature_data services.c3_quality_assessment.tests.test_p01_height_measurement services.c3_quality_assessment.tests.test_colab_preparation -v
```

These commands do not instantiate or train models. Rebuilding creates a matching
notebook with the ZIP checksum embedded. Transfer the ZIP and notebook together.

## Start training in hosted Google Colab

1. In Google Drive create **My Drive / FloraSense_C3**.
2. Upload this approximately 917 MiB ZIP without renaming it:
   `data/quality_assessment/processed/colab/florasense_c3_training.zip`.
3. Open https://colab.research.google.com/, select **File → Upload notebook**, and
   choose `services/c3_quality_assessment/notebooks/c3_feature_training_colab.ipynb`.
   Save a copy to Drive if desired.
4. Select **Runtime → Change runtime type → T4 GPU → Save**, then connect to a
   hosted runtime. If a T4 is unavailable, another CUDA GPU is acceptable; the
   notebook stops if CUDA is unavailable. Do not connect a local runtime.
5. Run cells from the top. Authorize the Drive mount. If you used a different
   folder, edit `DRIVE_DIR` in the setup cell. The notebook copies the ZIP to
   `/content`, checks its SHA-256, extracts it safely, checks every packaged file,
   installs evaluation dependencies, and checks grouped splits before training.
6. Run the **Train on the GPU** cell. It trains both tasks sequentially and saves
   after each outer fold to `My Drive/FloraSense_C3/runs/<UTC timestamp>/`.
7. Run the metrics cell for pooled/fold confusion matrices, per-class precision,
   recall/F1/support, minority precision/recall, macro-F1, balanced accuracy,
   majority baselines in JSON, and per-plant support and metrics.
8. Run the download cell. It downloads `florasense_c3_models.zip` and saves a
   timestamped backup ZIP in Drive. The separate final checkpoints are:
   `root/final_model.pth` and `leaf/final_model.pth`. Preserve each task's
   `evaluation.json`, `fold_*_split.csv`, and `out_of_fold_predictions.csv` with
   the models. Class mappings and preprocessing metadata are embedded in every
   checkpoint and report.

## Training design and limits

The initial baseline freezes ImageNet-pretrained ResNet18, including BatchNorm,
and trains a separate binary linear head for each task. Fixed encoder features
are computed with the GPU. Only training groups use augmentation views; original
deterministic features are used for validation/test. No dataset statistics or
encoder parameters are fitted on the evaluation groups.

Defaults: seed 42, three outer stratified group folds, an inner three-way group
split using one validation fold, at most 40 epochs, patience 6, AdamW 0.001,
weight decay 0.01, batch size 16, training-only inverse-frequency weights.
Split candidates are selected by class-support balance only; saved split CSVs
record the exact allocation. All photos and derived views of a `sample_key`
remain together, with path/hash-disjointness assertions on every split.

Preprocessing preserves aspect ratio, applies EXIF orientation, fits the image
within 208×208 and centers it on a gray 224×224 canvas, then applies ImageNet
normalization. Augmentation is horizontal flip, ±4° rotation, and up to 8%
brightness/contrast/saturation changes. It never crops or vertically flips.

Checkpoints minimize inner-validation weighted cross-entropy with early stopping.
Outer folds are evaluated only after checkpoint selection. Final full-data refits
use the median validation-selected epoch count, with no test-based model choice.
No independent external test score is claimed for the final refit. Models are
ordinary full ResNet18 state dictionaries despite the frozen-backbone training.

With only 15/16 minority samples, results remain preliminary. No P08 claim is
supported; P05 lacks leaf-distribution negatives. Exact hashes cannot identify
different photographs of the same plant under different IDs, so physical-plant
identity and near-duplicates still require human verification. Single-class
plant subsets report balanced accuracy as unavailable. Seeds and split hashes
aid reproducibility; floating-point results can differ across GPU/library versions.

No Colab training was executed during preparation. The GPU path and real training
results can only be validated when the user runs the notebook. Colab availability
and runtime duration vary. On disconnection, Drive checkpoints survive; the
initial pipeline starts a fresh timestamped run rather than resuming optimizer
state. API/frontend integration and stem-height assessment remain outside scope.

Official Colab references:
https://research.google.com/colaboratory/faq.html and
https://colab.research.google.com/notebooks/io.ipynb.
