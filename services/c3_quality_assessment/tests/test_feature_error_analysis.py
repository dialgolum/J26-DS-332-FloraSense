"""Synthetic saved predictions only: no checkpoint loading or training."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from services.c3_quality_assessment.analyze_feature_errors import (
    analyze_predictions, classify_error, contact_sheets, digest, summarize,
)


class FeatureErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self):
        source, predictions = [], []
        for index, (plant, truth, prediction) in enumerate([
            ("P01", "Acceptable", "Acceptable"),
            ("P02", "Not Acceptable", "Acceptable"),
            ("P03", "Acceptable", "Not Acceptable"),
            ("P04", "Not Acceptable", "Not Acceptable"),
        ]):
            path = self.root / f"{plant}_S001_root.png"
            Image.new("RGB", (25, 70), (index * 40, 90, 50)).save(path)
            row = {"Plant_ID": plant, "Sample_ID": "S001", "sample_key": f"{plant}:S001",
                   "Image_ID": f"IMG{index}", "view_type": "Root", "image_path": str(path),
                   "label": truth, "sha256": digest(path), "status": "usable", "exclusion_reason": ""}
            source.append(row)
            predictions.append({**row, "image_path": "/content/obsolete/" + path.name,
                "fold": 1, "true_index": int(truth == "Not Acceptable"),
                "predicted_index": int(prediction == "Not Acceptable"), "prediction": prediction,
                "minority_probability": .8 if prediction == "Not Acceptable" else .2})
        return pd.DataFrame(predictions), pd.DataFrame(source)

    def test_all_error_types_and_minority_positive_convention(self):
        self.assertEqual(classify_error("Not Acceptable", "Acceptable", "root"), "FN")
        self.assertEqual(classify_error("One Side", "Both Sides", "leaf"), "FN")
        with self.assertRaises(ValueError):
            classify_error("unclear", "Both Sides", "leaf")
        predictions, source = self.fixture()
        result = analyze_predictions(predictions, source, "root", self.root)
        self.assertEqual(set(result.error_type), {"TN", "FP", "FN", "TP"})
        self.assertTrue(result.match_status.eq("verified").all())
        self.assertEqual(result.iloc[0].error_type, "FN")
        plant, labels, pooled, summary = summarize(result)
        self.assertEqual(summary["correct"], 2)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["counts"], {"TN": 1, "FP": 1, "FN": 1, "TP": 1})
        self.assertEqual(int(plant.samples.sum()), 4)
        self.assertEqual(int(pooled.samples.sum()), 4)

    def test_composite_key_matches_local_source_not_colab_path(self):
        predictions, source = self.fixture()
        result = analyze_predictions(predictions, source, "root", self.root).set_index("Plant_ID")
        self.assertEqual(result.loc["P02", "expert_label"], "Not Acceptable")
        self.assertEqual(result.loc["P02", "source_image_path"], source.iloc[1].image_path)
        self.assertTrue(result.experiment_image_path.str.startswith("/content/").all())

    def test_missing_source_and_changed_labels_are_not_silently_matched(self):
        predictions, source = self.fixture()
        source.loc[1, "label"] = "Acceptable"
        result = analyze_predictions(predictions, source.iloc[:3], "root", self.root).set_index("Plant_ID")
        self.assertIn("expert_label_changed", result.loc["P02", "validation_issues"])
        self.assertIn("missing_verified_source", result.loc["P04", "validation_issues"])
        self.assertEqual(summarize(result.reset_index())[3]["unverified"], 2)

    def test_changed_image_content_and_invalid_probability_are_reported(self):
        predictions, source = self.fixture()
        Path(source.iloc[0].image_path).write_bytes(b"corrupt image")
        predictions.loc[1, "minority_probability"] = 1.5
        predictions.loc[2, "true_index"] = 1
        result = analyze_predictions(predictions, source, "root", self.root).set_index("Plant_ID")
        self.assertIn("source_hash_changed_from_manifest", result.loc["P01", "validation_issues"])
        self.assertIn("unreadable_source", result.loc["P01", "validation_issues"])
        self.assertIn("invalid_probability", result.loc["P02", "validation_issues"])
        self.assertIn("true_index_label_mismatch", result.loc["P03", "validation_issues"])

    def test_ambiguous_composite_keys_fail_closed(self):
        predictions, source = self.fixture()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            analyze_predictions(pd.concat([predictions, predictions.iloc[:1]]), source, "root", self.root)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            analyze_predictions(predictions, pd.concat([source, source.iloc[:1]]), "root", self.root)

    def test_contact_sheet_pagination_and_source_preservation(self):
        predictions, source = self.fixture()
        before = {path: digest(path) for path in source.image_path}
        result = analyze_predictions(predictions, source, "root", self.root)
        errors = result[~result.correct]
        pages, links = contact_sheets(errors, self.root / "sheets", "errors", per_page=1)
        self.assertEqual(len(pages), 2)
        self.assertEqual(set(links), set(errors.sample_key))
        for page in pages:
            with Image.open(self.root / "sheets" / page) as image:
                image.verify()
        self.assertEqual(before, {path: digest(path) for path in source.image_path})
        result.loc[:, "match_status"] = "unverified"
        with self.assertRaisesRegex(ValueError, "Unverified"):
            contact_sheets(result, self.root / "bad", "errors")


if __name__ == "__main__":
    unittest.main()
