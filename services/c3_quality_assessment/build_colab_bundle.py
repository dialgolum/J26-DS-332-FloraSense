"""Build a verified portable ZIP and an unexecuted Colab notebook; never train.

python -m services.c3_quality_assessment.build_colab_bundle
Original image bytes are preserved. The workbook and legacy model are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pandas as pd
from PIL import Image

from .data_loader import ROOT, EXCEL_PATH
from .prepare_feature_data import sample_key


CLASSES = {"root": ["Acceptable", "Not Acceptable"], "leaf": ["Both Sides", "One Side"]}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def notebook_document(zip_sha256):
    """All cells are plain Python so syntax can be validated without execution."""
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": text.splitlines(keepends=True)})

    md("""# FloraSense C3: root and leaf distribution — Colab GPU

Upload `florasense_c3_training.zip` to **My Drive / FloraSense_C3**. Open this
notebook with Colab, then select **Runtime → Change runtime type → T4 GPU**.
Run the cells in order. Training runs only in hosted Colab with CUDA; do not use
a local runtime. The ZIP contains original verified image bytes, relative-path
manifests, audit provenance, and a snapshot of the training implementation.

The baseline trains two independent two-class ResNet18 heads on frozen ImageNet
features. Three outer grouped folds are evaluated once; separate inner validation
groups select checkpoints and early stopping. No test-fold model selection.
P08 and stem height are outside this experiment. No existing model is loaded.
""")
    code('''import os, platform, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
assert platform.system() != "Windows", "Use a hosted Google Colab runtime"
from google.colab import drive
import torch
assert torch.cuda.is_available(), "Select Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__)
drive.mount("/content/drive")
''')
    code('''from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib, json, shutil, stat, zipfile

DRIVE_DIR = Path("/content/drive/MyDrive/FloraSense_C3")
DATA_ZIP = DRIVE_DIR / "florasense_c3_training.zip"
EXPECTED_ZIP_SHA256 = "''' + zip_sha256 + '''"
assert DATA_ZIP.is_file(), f"Upload the prepared ZIP to {DATA_ZIP}"
LOCAL_ZIP = Path("/content/florasense_c3_training.zip")
shutil.copyfile(DATA_ZIP, LOCAL_ZIP)
def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
assert sha256(LOCAL_ZIP) == EXPECTED_ZIP_SHA256, "Wrong or incomplete ZIP; use the matching notebook"
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
DATA_ROOT = Path("/content") / ("florasense_c3_" + stamp)
DATA_ROOT.mkdir(exist_ok=False)
with zipfile.ZipFile(LOCAL_ZIP) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)), "Duplicate archive members"
    for member in archive.infolist():
        name = PurePosixPath(member.filename)
        assert not name.is_absolute() and ".." not in name.parts
        assert ":" not in member.filename and "\\\\" not in member.filename
        assert not stat.S_ISLNK(member.external_attr >> 16), "Symlinks are not accepted"
    archive.extractall(DATA_ROOT)
bundle = json.loads((DATA_ROOT / "bundle.json").read_text())
assert set(names) == set(bundle["file_sha256"]) | {"bundle.json"}
for name, expected in bundle["file_sha256"].items():
    assert sha256(DATA_ROOT / name) == expected, f"Checksum mismatch: {name}"
print("Verified", bundle["image_files"], "image files; source workbook:", bundle["workbook_sha256"])
''')
    code('''import subprocess
# Colab supplies the CUDA-compatible torch/torchvision pair. Do not replace it
# with local CPU wheels. Install only the data/evaluation dependencies.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(DATA_ROOT / "requirements_colab.txt")])
sys.path.insert(0, str(DATA_ROOT))
''')
    code('''import numpy as np
import pandas as pd
from services.c3_quality_assessment.train_feature_classifier import (
    CLASSES, load_verified_manifest, grouped_folds, main,
)
from services.c3_quality_assessment.prepare_feature_data import validate_split_groups

for task in ("root", "leaf"):
    frame = load_verified_manifest(DATA_ROOT / f"{task}_usable.csv", task)
    labels = frame.label.map({name: i for i, name in enumerate(CLASSES[task])}).to_numpy()
    print(task, "samples:", len(frame), "labels:", frame.label.value_counts().to_dict())
    for fold, (development, test) in enumerate(grouped_folds(frame, labels, 3, 42, minimum=3), 1):
        train_rel, val_rel = grouped_folds(frame.iloc[development].reset_index(drop=True), labels[development], 3, 42 + fold, minimum=2)[0]
        train, validation = development[train_rel], development[val_rel]
        for a, b in ((train, validation), (train, test), (validation, test)):
            validate_split_groups(frame.iloc[a], frame.iloc[b])
        print(" fold", fold, {name: np.bincount(labels[idx], minlength=2).tolist()
                              for name, idx in (("train", train), ("validation", validation), ("test", test))})
print("Preflight passed. No model has been trained by these setup cells.")
''')
    md("""## Train on the GPU

The next cell starts training. Defaults: seed 42, 3 outer folds, 3-way inner
split (one held-out validation fold), up to 40 epochs, patience 6, AdamW 0.001,
weight decay 0.01, batch size 16. Only the final linear layer is trained; the
ImageNet backbone and BatchNorm stay frozen in evaluation mode. Two fixed
augmentation views plus the original are used for training only. Frozen encoder
features are cached in memory; no encoder is fitted on validation/test data.

Images are EXIF-oriented, resized to fit 208×208, and centered on a 224×224 gray
canvas. Training uses horizontal flips, ±4° rotation, and mild color changes;
no crops or vertical flips. Class weights use training groups only. Argmax is
fixed; no test-based threshold tuning. All views/crops share `sample_key`.

Checkpoints are saved to Drive after each fold. The final model is refit on all
usable data for the median validation-selected epoch count. CV estimates the
training procedure; the final refit has no independent external test score.
""")
    code('''RUN_DIR = DRIVE_DIR / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
print("Saving models and reports to", RUN_DIR)
main(["--task", "both", "--manifest-dir", str(DATA_ROOT), "--output-dir", str(RUN_DIR),
      "--folds", "3", "--seed", "42", "--max-epochs", "40", "--patience", "6",
      "--learning-rate", "0.001", "--augmentation-views", "2"])
''')
    code('''import matplotlib.pyplot as plt
summary_rows = []
for task in ("root", "leaf"):
    report = json.loads((RUN_DIR / task / "evaluation.json").read_text())
    metrics = report["oof_metrics"]
    summary_rows.append({"task": task, **{k: metrics[k] for k in
        ("macro_f1", "balanced_accuracy", "minority_precision", "minority_recall")}})
    print(task, "per-class precision/recall/F1/support:")
    display(pd.DataFrame(metrics["classification_report"]).T)
    print("Outer-fold variation (each test fold evaluated once):")
    display(pd.DataFrame([{"fold": item["fold"], "best_epoch": item["best_epoch"],
        **{key: item["metrics"][key] for key in ("macro_f1", "balanced_accuracy", "minority_precision", "minority_recall")}}
        for item in report["folds"]]))
    print("Per plant (both-class metrics unavailable where a class is absent):")
    display(pd.DataFrame([{"plant": plant, "samples": value["samples"],
        "support": value["class_support"], "macro_f1": value["metrics"]["macro_f1"],
        "balanced_accuracy": value["metrics"]["balanced_accuracy"]}
        for plant, value in report["per_plant"].items()]))
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES[task], yticklabels=CLASSES[task],
           xlabel="Predicted", ylabel="True", title=f"{task}: out-of-fold confusion matrix")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(RUN_DIR / task / "confusion_matrix.png", dpi=160)
    plt.show()
display(pd.DataFrame(summary_rows))
''')
    code('''from google.colab import files
artifact_zip = shutil.make_archive("/content/florasense_c3_models", "zip", root_dir=RUN_DIR)
drive_copy = RUN_DIR.parent / (RUN_DIR.name + "_models.zip")
shutil.copyfile(artifact_zip, drive_copy)
print("Drive backup:", drive_copy)
print("Final checkpoints:", RUN_DIR / "root/final_model.pth", RUN_DIR / "leaf/final_model.pth")
files.download(artifact_zip)
''')
    md("""## Interpretation and next handoff

Download includes separate root/leaf `final_model.pth`, three validation-selected
fold checkpoints per task, split CSVs, out-of-fold predictions, evaluation JSON,
class mappings, preprocessing metadata, and confusion-matrix PNGs. Keep the
metadata and reports with the weights. No general-quality checkpoint is present.

Minority counts are only 15 root and 16 leaf samples. Report fold variability and
majority baselines, not just pooled accuracy. There is no P08 evidence and no
P05 One Side examples. Composite keys assume each record identifies one physical
plant; exact hashes prevent byte-identical leakage but cannot establish that
different photos show different plants. Review cross-ID recaptures/near-duplicates
before making generalization claims. Do not tune on outer test predictions.

If Colab disconnects, saved Drive fold checkpoints remain. Restart with a new
output directory; this initial pipeline reruns rather than resuming optimizer
state. GPU availability, including T4 availability, depends on Colab allocation.
API/frontend integration and stem-height assessment are separate future phases.
""")
    for index, cell in enumerate(cells):
        cell["id"] = f"florasense-c3-{index:02d}"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "c3_feature_training_colab.ipynb", "provenance": [], "gpuType": "T4"}}, "cells": cells}


def build_bundle(manifest_dir, output_zip, notebook_path, root=ROOT):
    root = root.resolve()
    audit_path = manifest_dir / "dataset_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["workbook_sha256"] != digest(root / EXCEL_PATH.relative_to(ROOT)):
        raise ValueError("Dataset audit is stale; rerun prepare_feature_data")
    payloads, image_sources = {}, {}
    for task, classes in CLASSES.items():
        frame = pd.read_csv(manifest_dir / f"{task}_usable.csv", keep_default_na=False)
        if frame.empty or not frame.status.eq("usable").all() or not frame.exclusion_reason.eq("").all():
            raise ValueError("Only nonempty usable manifests may be packaged")
        if not frame.label.isin(classes).all() or not frame.view_type.eq("Root" if task == "root" else "Front").all():
            raise ValueError("Invalid task labels or views")
        if len(frame) != audit["tasks"][task]["usable_images"] or frame.label.value_counts().to_dict() != audit["tasks"][task]["class_counts"]:
            raise ValueError("Manifest disagrees with audited counts")
        for field in ["image_path", "sha256"]:
            if frame.groupby(field).sample_key.nunique().gt(1).any():
                raise ValueError("Ambiguous duplicate-image ownership")
        portable_paths = []
        for row in frame.itertuples():
            path = Path(row.image_path).resolve()
            if not path.is_relative_to(root) or row.sample_key != sample_key(row.Plant_ID, row.Sample_ID):
                raise ValueError("Invalid image path or sample key")
            if digest(path) != row.sha256:
                raise ValueError(f"Image differs from audited content: {path}")
            with Image.open(path) as image:
                image.verify()
            relative = f"images/{row.Plant_ID}/{path.name}"
            if relative in image_sources and image_sources[relative] != path:
                raise ValueError("Archive filename collision")
            image_sources[relative] = path
            portable_paths.append(relative)
        frame["image_path"] = portable_paths
        payloads[f"{task}_usable.csv"] = frame.to_csv(index=False).encode("utf-8")

    def portable_audit(value):
        if isinstance(value, dict):
            return {key: portable_audit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable_audit(item) for item in value]
        if isinstance(value, str) and value.startswith(str(root)):
            return Path(value).relative_to(root).as_posix()
        return value

    payloads["dataset_audit.json"] = json.dumps(portable_audit(audit), indent=2).encode("utf-8")
    payloads["requirements_colab.txt"] = b"pandas>=2.2,<3\nscikit-learn==1.7.2\nPillow>=10\nmatplotlib>=3.8\n"
    payloads["README.md"] = b"FloraSense C3 verified training bundle. Open the matching Colab notebook. No workbook, stem-height data, legacy model, or secrets included. See dataset_audit.json for provenance and exclusions.\n"
    payloads["services/__init__.py"] = b""
    for filename in ["__init__.py", "data_loader.py", "prepare_feature_data.py", "train_feature_classifier.py"]:
        payloads[f"services/c3_quality_assessment/{filename}"] = (root / "services/c3_quality_assessment" / filename).read_bytes()
    checksums = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    checksums.update({name: digest(path) for name, path in image_sources.items()})
    bundle = {"format_version": 1, "workbook_sha256": audit["workbook_sha256"],
              "source_audit_sha256": digest(audit_path), "image_files": len(image_sources),
              "class_names": CLASSES, "file_sha256": checksums,
              "note": "Original images copied byte-for-byte. CSV paths are relative to this bundle root."}
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    partial = output_zip.with_suffix(output_zip.suffix + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, path in sorted(image_sources.items()):
            archive.write(path, name)
        for name, data in payloads.items():
            archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("bundle.json", json.dumps(bundle, indent=2))
    # Verify every archived byte before atomically publishing the generated ZIP.
    with zipfile.ZipFile(partial) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate ZIP entries")
        for name, expected in checksums.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive checksum mismatch: {name}")
    if digest(root / EXCEL_PATH.relative_to(ROOT)) != audit["workbook_sha256"]:
        raise ValueError("Workbook changed during packaging")
    os.replace(partial, output_zip)
    zip_hash = digest(output_zip)
    output_zip.with_suffix(".zip.sha256").write_text(f"{zip_hash}  {output_zip.name}\n", encoding="utf-8")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = notebook_document(zip_hash)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(notebook_path), "exec")
    notebook_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"zip": str(output_zip), "notebook": str(notebook_path), "image_files": len(image_sources),
            "zip_sha256": zip_hash, "zip_bytes": output_zip.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/quality_assessment/processed/feature_classification")
    parser.add_argument("--output-zip", type=Path, default=ROOT / "data/quality_assessment/processed/colab/florasense_c3_training.zip")
    parser.add_argument("--notebook", type=Path, default=ROOT / "services/c3_quality_assessment/notebooks/c3_feature_training_colab.ipynb")
    args = parser.parse_args(argv)
    print(json.dumps(build_bundle(args.manifest_dir, args.output_zip, args.notebook), indent=2))


if __name__ == "__main__":
    main()
