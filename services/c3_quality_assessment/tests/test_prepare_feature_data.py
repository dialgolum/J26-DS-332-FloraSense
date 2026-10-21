"""Feature preparation tests use synthetic images, never the production workbook."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from services.c3_quality_assessment.prepare_feature_data import (
    normalize_label, prepare, sample_key, validate_split_groups,
)


class FeaturePreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def image(self, name, color="green"):
        (self.root / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), color).save(self.root / name)
        return name

    def sheets(self, records):
        """Records: plant, sample, view, path, root label, leaf label."""
        samples, annotations, images = [], [], []
        for index, (plant, sample, view, path, root_label, leaf_label) in enumerate(records):
            key = {"Plant_ID": plant, "Sample_ID": sample}
            samples.append(key)
            annotations.append({**key, "Root_Distribution": root_label, "Leaf_Distribution": leaf_label})
            images.append({**key, "Image_ID": f"IMG{index}", "View_Type": view, "Relative_Path": path})
        return {"Samples": pd.DataFrame(samples).drop_duplicates(),
                "Quality_Annotations": pd.DataFrame(annotations).drop_duplicates(),
                "Images": pd.DataFrame(images)}

    def test_composite_keys_join_same_local_id_to_correct_plant(self):
        records = [
            ("P01", "S001", "Root", self.image("P01_S001_root.png"), "Acceptable", None),
            ("P02", "S001", "Root", self.image("P02_S001_root.png", "red"), "Not Acceptable", "Both Sides"),
        ]
        manifests, _ = prepare(self.sheets(records), self.root)
        root = manifests["root"].set_index("sample_key")
        self.assertEqual(len(root), 2)
        self.assertEqual(root.loc["P01:S001", "label"], "Acceptable")
        self.assertEqual(root.loc["P02:S001", "label"], "Not Acceptable")
        self.assertTrue(root.status.eq("usable").all())
        self.assertEqual(sample_key(" P02 ", " S001 "), "P02:S001")

    def test_explicit_label_normalization_and_unknown_rejection(self):
        for raw in [" not  ACCEPTABLE ", "not_acceptable", "Not-Acceptable", "Unacceptable"]:
            self.assertEqual(normalize_label(raw, "root"), "Not Acceptable")
        for raw in [" Both  Sides ", "both_side", "BOTH-SIDES"]:
            self.assertEqual(normalize_label(raw, "leaf"), "Both Sides")
        self.assertEqual(normalize_label("one_side", "leaf"), "One Side")
        for raw in [None, float("nan"), "", "PASS", "Yes", "unclear", "Both Sides"]:
            self.assertIsNone(normalize_label(raw, "root"))

    def test_duplicate_conflicting_front_reference_is_quarantined(self):
        path = self.image("P03_S001_front.png")
        sheets = self.sheets([
            ("P03", "S001", "Front", path, "Acceptable", "One Side"),
            ("P03", "S002", "Front", "./" + path, "Acceptable", "Both Sides"),
        ])
        manifests, audit = prepare(sheets, self.root)
        leaf = manifests["leaf"]
        self.assertTrue(leaf.status.eq("excluded").all())
        self.assertTrue(leaf.exclusion_reason.str.contains("duplicate_path_reference").all())
        self.assertTrue(leaf.exclusion_reason.str.contains("conflicting_labels").all())
        self.assertEqual(len(audit["duplicate_references"]), 1)

    def test_shared_content_under_different_paths_cannot_leak(self):
        a = self.image("P03_S001_root.png")
        b = self.image("P04_S001_root.png")
        manifests, _ = prepare(self.sheets([
            ("P03", "S001", "Root", a, "Acceptable", "One Side"),
            ("P04", "S001", "Root", b, "Not Acceptable", "Both Sides"),
        ]), self.root)
        root = manifests["root"]
        self.assertTrue(root.status.eq("excluded").all())
        self.assertTrue(root.exclusion_reason.str.contains("shared_image_content_across_samples").all())

    def test_cross_view_duplicate_is_also_excluded(self):
        path = self.image("P01_S001_root.png")
        manifests, _ = prepare(self.sheets([
            ("P01", "S001", "Root", path, "Acceptable", None),
            ("P01", "S002", "Front", path, "Acceptable", None),
        ]), self.root)
        self.assertIn("duplicate_path_reference", manifests["root"].iloc[0].exclusion_reason)
        self.assertEqual(manifests["root"].iloc[0].status, "excluded")

    def test_quarantine_missing_corrupt_mismatch_and_unknown_labels(self):
        (self.root / "P03_S003_root.png").write_text("not an image")
        manifests, _ = prepare(self.sheets([
            ("P02", "S012", "Root", self.image("P02_S012_root.png"), "Acceptable", "Both Sides"),
            ("P03", "S002", "Root", "P03_S002_root.png", "Acceptable", "One Side"),
            ("P03", "S003", "Root", "P03_S003_root.png", "Acceptable", "One Side"),
            ("P03", "S004", "Root", self.image("P03_S099_root.png", "blue"), "unclear", "One Side"),
        ]), self.root)
        root = manifests["root"].set_index("sample_key")
        self.assertIn("p02_corrected_linkage_not_verified", root.loc["P02:S012", "exclusion_reason"])
        self.assertIn("missing_file", root.loc["P03:S002", "exclusion_reason"])
        self.assertIn("unreadable_image", root.loc["P03:S003", "exclusion_reason"])
        self.assertIn("filename_sample_mismatch", root.loc["P03:S004", "exclusion_reason"])
        self.assertIn("missing_or_unsupported_label", root.loc["P03:S004", "exclusion_reason"])

    def test_p08_without_annotations_or_images_is_reported_not_inferred(self):
        sheets = self.sheets([("P01", "S001", "Root", self.image("P01_S001_root.png"), "Acceptable", None)])
        sheets["Samples"] = pd.concat([sheets["Samples"], pd.DataFrame([{"Plant_ID": "P08", "Sample_ID": "S001"}])])
        manifests, _ = prepare(sheets, self.root)
        for manifest in manifests.values():
            row = manifest[manifest.Plant_ID.eq("P08")].iloc[0]
            self.assertEqual(row.status, "excluded")
            self.assertIn("missing_annotation", row.exclusion_reason)
            self.assertIn("missing_view_image", row.exclusion_reason)
            self.assertEqual(row.label, "")

    def test_split_guard_checks_all_three_leakage_channels(self):
        train = pd.DataFrame([{"sample_key": "P01:S001", "image_path": "/a.png", "sha256": "aaa"}])
        evaluation = pd.DataFrame([{"sample_key": "P02:S001", "image_path": "/b.png", "sha256": "bbb"}])
        validate_split_groups(train, evaluation)
        for column in ["sample_key", "image_path", "sha256"]:
            overlapping = evaluation.copy()
            overlapping[column] = train[column]
            with self.assertRaisesRegex(ValueError, column):
                validate_split_groups(train, overlapping)

    def test_verified_p08_is_allowed_and_sources_are_unchanged(self):
        sheets = self.sheets([("P08", "S001", "Front", self.image("P08_S001_front.png"), "Acceptable", " Both  Sides ")])
        originals = {name: frame.copy(deep=True) for name, frame in sheets.items()}
        manifests, _ = prepare(sheets, self.root)
        row = manifests["leaf"].iloc[0]
        self.assertEqual(row.status, "usable")
        self.assertEqual(row.label, "Both Sides")
        self.assertEqual(row.raw_label, " Both  Sides ")
        for name, original in originals.items():
            pd.testing.assert_frame_equal(sheets[name], original)

    def test_empty_annotations_are_safe_and_duplicate_samples_fail_closed(self):
        sheets = self.sheets([("P03", "S001", "Root", self.image("P03_S001_root.png"), "Acceptable", "One Side")])
        sheets["Quality_Annotations"] = sheets["Quality_Annotations"].iloc[:0]
        manifests, _ = prepare(sheets, self.root)
        self.assertIn("missing_annotation", manifests["root"].iloc[0].exclusion_reason)
        sheets["Samples"] = pd.concat([sheets["Samples"], sheets["Samples"]])
        with self.assertRaisesRegex(ValueError, "duplicate composite keys"):
            prepare(sheets, self.root)

    def test_duplicate_annotations_do_not_multiply_training_records(self):
        sheets = self.sheets([("P03", "S001", "Root", self.image("P03_S001_root.png"), "Acceptable", "One Side")])
        other = sheets["Quality_Annotations"].copy()
        other["Root_Distribution"] = "Not Acceptable"
        sheets["Quality_Annotations"] = pd.concat([sheets["Quality_Annotations"], other])
        manifests, _ = prepare(sheets, self.root)
        root = manifests["root"]
        self.assertEqual(len(root), 1)
        self.assertIn("duplicate_annotation", root.iloc[0].exclusion_reason)
        self.assertIn("conflicting_labels", root.iloc[0].exclusion_reason)

    def test_corrected_p02_links_release_only_valid_records(self):
        sheets = self.sheets([
            ("P02", "S012", "Root", self.image("P02/P02_S012_root.png"), "Acceptable", "One Side"),
            ("P02", "S013", "Root", self.image("P02/P02_S012_root_bad.png", "red"), "Acceptable", "Both Sides"),
            ("P02", "S014", "Root", "P02/P02_S014_root.png", "Acceptable", "Both Sides"),
            ("P02", "S015", "Root", self.image("P02/P02_S015_root.png", "blue"), "unclear", "Both Sides"),
        ])
        manifests, audit = prepare(sheets, self.root)
        root = manifests["root"].set_index("sample_key")
        self.assertEqual(root.loc["P02:S012", "status"], "usable")
        self.assertTrue(root.loc[["P02:S013", "P02:S014", "P02:S015"], "status"].eq("excluded").all())
        self.assertIn("missing_file", root.loc["P02:S014", "exclusion_reason"])
        self.assertIn("missing_or_unsupported_label", root.loc["P02:S015", "exclusion_reason"])
        self.assertEqual(len(audit["p02_corrected_reference_validation"]), 4)


if __name__ == "__main__":
    unittest.main()
