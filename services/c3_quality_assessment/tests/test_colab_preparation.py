"""Validate packaging/splits without instantiating or training any model."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from services.c3_quality_assessment.build_colab_bundle import build_bundle, digest, notebook_document
from services.c3_quality_assessment.data_loader import ROOT, EXCEL_PATH
from services.c3_quality_assessment.train_feature_classifier import (
    evaluation_metrics, grouped_folds, load_verified_manifest,
)
from services.c3_quality_assessment.prepare_feature_data import validate_split_groups


class ColabPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_notebook_is_unexecuted_and_every_code_cell_compiles(self):
        notebook = notebook_document("a" * 64)
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), "colab_cell", "exec")

    def test_grouped_stratification_and_split_disjointness(self):
        rows = []
        for index in range(24):
            for view in range(2):
                rows.append({"sample_key": f"P02:S{index:03d}", "image_path": f"/{index}_{view}.png",
                             "sha256": f"digest-{index}-{view}", "label": index % 2})
        frame = pd.DataFrame(rows)
        coverage = []
        for train, test in grouped_folds(frame, frame.label.to_numpy(), 3, 42):
            validate_split_groups(frame.iloc[train], frame.iloc[test])
            self.assertEqual(set(frame.iloc[test].label), {0, 1})
            coverage.extend(test)
        self.assertEqual(sorted(coverage), list(range(len(frame))))

    def test_split_planner_rejects_cross_sample_duplicate_content(self):
        frame = pd.DataFrame([{"sample_key": f"P02:S{i}", "image_path": f"/{i}.png",
                               "sha256": "shared", "label": i % 2} for i in range(18)])
        with self.assertRaisesRegex(ValueError, "leakage"):
            grouped_folds(frame, frame.label.to_numpy(), 3, 42)

    def test_single_class_plant_balanced_accuracy_is_unavailable(self):
        metrics = evaluation_metrics([0, 0], [0, 1], ["Both Sides", "One Side"])
        self.assertIsNone(metrics["balanced_accuracy"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 0]])

    def fixture(self):
        workbook = self.root / EXCEL_PATH.relative_to(ROOT)
        workbook.parent.mkdir(parents=True)
        workbook.write_bytes(b"synthetic workbook provenance; never opened or modified")
        manifests = self.root / "manifests"
        manifests.mkdir()
        audit = {"workbook_sha256": digest(workbook), "tasks": {}}
        for task, view, labels in [("root", "Root", ["Acceptable", "Not Acceptable"]),
                                   ("leaf", "Front", ["Both Sides", "One Side"])]:
            records = []
            for index, label in enumerate(labels, 1):
                path = self.root / f"P02_S{index:03d}_{view.lower()}.png"
                Image.new("RGB", (10, 20), (index * 50, 0 if task == "root" else 50, 0)).save(path)
                records.append({"Plant_ID": "P02", "Sample_ID": f"S{index:03d}",
                    "sample_key": f"P02:S{index:03d}", "image_path": str(path), "sha256": digest(path),
                    "label": label, "view_type": view, "status": "usable", "exclusion_reason": ""})
            pd.DataFrame(records).to_csv(manifests / f"{task}_usable.csv", index=False)
            audit["tasks"][task] = {"usable_images": 2, "class_counts": {label: 1 for label in labels}}
        (manifests / "dataset_audit.json").write_text(json.dumps(audit))
        source = self.root / "services/c3_quality_assessment"
        source.mkdir(parents=True)
        for filename in ["__init__.py", "data_loader.py", "prepare_feature_data.py", "train_feature_classifier.py"]:
            (source / filename).write_text("# synthetic source snapshot\n")
        return manifests

    def test_bundle_roundtrip_relative_paths_checksums_and_source_preservation(self):
        manifests = self.fixture()
        source_before = {path: digest(path) for path in self.root.glob("*.png")}
        output = self.root / "out/dataset.zip"
        notebook = self.root / "out/training.ipynb"
        result = build_bundle(manifests, output, notebook, self.root)
        self.assertEqual(result["image_files"], 4)
        extracted = self.root / "extracted"
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            archive.extractall(extracted)
        bundle = json.loads((extracted / "bundle.json").read_text())
        for name, expected in bundle["file_sha256"].items():
            self.assertEqual(digest(extracted / name), expected)
        for task in ["root", "leaf"]:
            raw = pd.read_csv(extracted / f"{task}_usable.csv")
            self.assertTrue(raw.image_path.str.startswith("images/").all())
            loaded = load_verified_manifest(extracted / f"{task}_usable.csv", task)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(loaded.image_path.map(lambda p: Path(p).is_file()).all())
        self.assertEqual(source_before, {path: digest(path) for path in source_before})
        self.assertFalse(list(extracted.rglob("*.xlsx")))
        self.assertFalse(list(extracted.rglob("*.pth")))

    def test_changed_source_image_and_stale_audit_are_rejected(self):
        manifests = self.fixture()
        next(self.root.glob("*.png")).write_bytes(b"changed content")
        with self.assertRaisesRegex(ValueError, "differs from audited"):
            build_bundle(manifests, self.root / "bad.zip", self.root / "bad.ipynb", self.root)
        workbook = self.root / EXCEL_PATH.relative_to(ROOT)
        workbook.write_bytes(b"changed workbook")
        with self.assertRaisesRegex(ValueError, "stale"):
            build_bundle(manifests, self.root / "bad.zip", self.root / "bad.ipynb", self.root)

    def test_portable_manifest_tampering_is_rejected(self):
        manifests = self.fixture()
        output = self.root / "bundle.zip"
        build_bundle(manifests, output, self.root / "training.ipynb", self.root)
        extracted = self.root / "extracted"
        with zipfile.ZipFile(output) as archive:
            archive.extractall(extracted)
        path = extracted / "root_usable.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            load_verified_manifest(path, "root")


if __name__ == "__main__":
    unittest.main()
