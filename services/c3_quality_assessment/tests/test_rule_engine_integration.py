
"""Integration test for rule-based and AI visual assessment."""

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = PROJECT_ROOT / "services" / "c3_quality_assessment"

sys.path.insert(0, str(COMPONENT_DIR))

from rule_engine import assess_sample


IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "quality_assessment"
    / "original"
    / "individual"
    / "P01"
    / "P01_S001_front.jpg"
)


def build_test_sheets():
    """Create minimal test data matching the rule engine's schema."""

    return {
        "Samples": pd.DataFrame([
            {
                "Sample_ID": "TEST_S001",
                "Plant_ID": "P01",
                "Observed_Root_Status": "Rooted",
                "Observed_Packing_Mode": "Sleeve",
            }
        ]),
        "Plant_Types": pd.DataFrame([
            {
                "Plant_ID": "P01",
                "Height_Check": "No",
                "Leaf_Count_Check": "No",
                "Leaf_Distribution_Check": "No",
                "Root_Distribution": "No",
                "Oasis_Condition": "No",
                "Overall_Quality": "Yes",
            }
        ]),
        "Requirement_Profiles": pd.DataFrame([
            {
                "Requirement_ID": "TEST_R001",
                "Plant_ID": "P01",
                "Buyer_ID": "TEST_BUYER",
                "Root_Status": "Rooted",
                "Packing_Mode": "Sleeve",
                "Min_Total_Leaf_Count": None,
            }
        ]),
        "Height_Assessments": pd.DataFrame(columns=[
            "Sample_ID", "Requirement_ID", "Height_Result"
        ]),
        "Quality_Annotations": pd.DataFrame([
            {
                "Sample_ID": "TEST_S001",
                "Plant_ID": "P01",
                "Total_Leaf_Count": None,
                "Leaf_Distribution": None,
                "Root_Distribution": None,
                "Oasis_Damaged": None,
                "General_visual_Quality": "Acceptable",
            }
        ]),
    }


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    sheets = build_test_sheets()

    # Test 1: Existing rule-based behavior without AI.
    rule_result = assess_sample(
        sheets,
        sample_id="TEST_S001",
        requirement_id="TEST_R001",
    )

    assert rule_result["Overall_Result"] == "PASS"
    assert rule_result["Visual_Model"] is None

    # Test 2: AI result is returned separately.
    combined_result = assess_sample(
        sheets,
        sample_id="TEST_S001",
        requirement_id="TEST_R001",
        image_path=IMAGE_PATH,
    )

    visual = combined_result["Visual_Model"]

    assert visual is not None

    print("\n--- RULE ENGINE + AI INTEGRATION TEST ---")
    print("Rule-based result:", combined_result["Overall_Result"])
    print("AI inference status:", visual["Status"])

    if visual["Status"] == "SUCCESS":
        print("Image:", visual["Image"])
        print("Prediction:", visual["Predicted_Quality"])
        print(f"Confidence: {visual['Confidence']:.2%}")
        print("Probabilities:", visual["Probabilities"])
        print("Device:", visual["Device"])

        assert visual["Predicted_Quality"] in {
            "Acceptable", "Not Acceptable"
        }
        assert 0.0 <= visual["Confidence"] <= 1.0
        assert combined_result["Overall_Result"] == "PASS"

        print("\nPASS: Both assessment paths completed.")
    else:
        print("Inference error:", visual["Error"])
        print(
            "\nThe rule-based assessment still completed, "
            "but AI inference needs investigation."
        )
        raise RuntimeError("AI inference failed during integration test")



class TestAssessmentOrchestration(unittest.TestCase):
    def setUp(self):
        from services.c3_quality_assessment.run_assessment import run_assessment
        self.run = run_assessment
        self.sheets = build_test_sheets()
        for name in ("Samples", "Plant_Types", "Requirement_Profiles", "Quality_Annotations"):
            self.sheets[name]["Plant_ID"] = "P02"
        self.sheets["Plant_Types"]["Root_Distribution"] = "Yes"
        self.sheets["Quality_Annotations"]["Root_Distribution"] = "Acceptable"
        self.sheets["Quality_Annotations"]["Leaf_Distribution"] = "Both Sides"
        self.sheets["Images"] = pd.DataFrame([
            {"Plant_ID": "P02", "Sample_ID": "TEST_S001", "View_Type": view,
             "Relative_Path": str(IMAGE_PATH), "Image_ID": view}
            for view in ("Root", "Front")])
        self.predict = patch("services.c3_quality_assessment.run_assessment.predict_feature").start()
        self.addCleanup(patch.stopall)
        self.predict.side_effect = lambda image, checkpoint, task, **kwargs: {
            "status": "success", "task": task,
            "predicted_label": "Not Acceptable" if task == "root" else "One Side",
            "confidence": .8, "class_probabilities": {}, "dataset_version": "c3_features_v2_20261022"}
        self.path = patch("services.c3_quality_assessment.run_assessment.resolve_image_path").start()
        self.path.return_value.is_file.return_value = True

    def assess(self, **kwargs):
        return self.run(self.sheets, "TEST_S001", "TEST_R001", **kwargs)

    def test_disabled_preserves_legacy_fields_and_inputs(self):
        before = {k: v.copy(deep=True) for k, v in self.sheets.items()}
        expected = assess_sample(self.sheets, "TEST_S001", "TEST_R001")
        result = self.assess()
        for key, value in expected.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result["Rule_Assessment"], expected)
        self.predict.assert_not_called()
        for key in before:
            pd.testing.assert_frame_equal(before[key], self.sheets[key])

    def test_composite_identity_and_disagreement_do_not_flip_pass(self):
        other = self.sheets["Samples"].copy()
        other["Plant_ID"] = "P01"
        self.sheets["Samples"] = pd.concat([other, self.sheets["Samples"]], ignore_index=True)
        other_images = self.sheets["Images"].copy()
        other_images["Plant_ID"] = "P01"
        self.sheets["Images"] = pd.concat([other_images, self.sheets["Images"]], ignore_index=True)
        result = self.assess(ai_enabled=True, root_checkpoint="explicit-root.pth")
        self.assertEqual(result["sample_key"], "P02:TEST_S001")
        self.assertEqual(result["Buyer_Rule_Result"], "PASS")
        self.assertEqual(result["Disagreements"][0]["expert_label"], "Acceptable")
        self.assertEqual(result["AI_Observations"]["root_B"]["image"]["view"], "Root")
        self.assertTrue(result["Manual_Review"]["required"])
        self.assertFalse(result["Manual_Review"]["affects_buyer_rule_result"])
        self.assertEqual(self.predict.call_args.kwargs["variant"], "B")

    def test_ai_agreement_or_opposition_cannot_flip_fail(self):
        self.sheets["Quality_Annotations"]["Root_Distribution"] = "Not Acceptable"
        self.predict.side_effect = None
        self.predict.return_value = {"status": "success", "predicted_label": "Acceptable"}
        self.assertEqual(self.assess(ai_enabled=True, root_checkpoint="root.pth")["Overall_Result"], "FAIL")

    def test_missing_unknown_and_duplicate_expert_annotations(self):
        original = self.sheets["Quality_Annotations"].copy()
        for value in (None, "Unknown", ""):
            with self.subTest(value=value):
                self.sheets["Quality_Annotations"] = original.copy()
                self.sheets["Quality_Annotations"]["Root_Distribution"] = value
                result = self.assess(ai_enabled=True, root_checkpoint="root.pth")
                self.assertEqual(result["Overall_Result"], "MANUAL REVIEW")
                self.assertEqual(result["Disagreements"], [])
                self.assertIsNone(result["AI_Observations"]["root_B"]["expert_annotation"]["label"])
        for frame in (original.iloc[:0], pd.concat([original, original])):
            self.sheets["Quality_Annotations"] = frame
            result = self.assess(ai_enabled=True, root_checkpoint="root.pth")
            self.assertEqual(result["Overall_Result"], "MANUAL REVIEW")
            self.assertEqual(result["Disagreements"], [])

    def test_leaf_branch_origin_and_explicit_comparison(self):
        self.sheets["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
        result = self.assess(ai_enabled=True, root_checkpoint="root.pth",
                             leaf_checkpoint="leaf-b.pth", leaf_a_checkpoint="leaf-a.pth")
        self.assertEqual(result["Overall_Result"], "PASS")
        observations = result["AI_Observations"]
        self.assertEqual(observations["leaf_B"]["observation_type"], "branch_origin_distribution")
        self.assertEqual(observations["leaf_B"]["image"]["view"], "Front")
        self.assertEqual(observations["leaf_A_comparison"]["comparison_status"], "disagreement")
        self.assertEqual([call.args[1] for call in self.predict.call_args_list],
                         ["root.pth", "leaf-b.pth", "leaf-a.pth"])
        self.assertEqual([call.kwargs["variant"] for call in self.predict.call_args_list], ["B", "B", "A"])

    def test_leaf_ai_never_replaces_expert_decision(self):
        self.sheets["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
        for expert, ai, expected in (("Both Sides", "One Side", "PASS"),
                                     ("One Side", "Both Sides", "FAIL"),
                                     (None, "Both Sides", "MANUAL REVIEW")):
            with self.subTest(expert=expert, ai=ai):
                self.sheets["Quality_Annotations"]["Leaf_Distribution"] = expert
                self.predict.side_effect = lambda image, checkpoint, task, **kwargs: {
                    "status": "success", "predicted_label": ai if task == "leaf" else "Acceptable"}
                without_ai = self.assess()
                result = self.assess(ai_enabled=True, root_checkpoint="root.pth", leaf_checkpoint="leaf.pth")
                self.assertEqual(result["Buyer_Rule_Result"], expected)
                self.assertEqual(result["Checks"], without_ai["Checks"])
                self.assertEqual(result["AI_Observations"]["leaf_B"]["comparison_status"],
                                 "disagreement" if expert is not None else "expert_annotation_missing")

    def test_inference_errors_do_not_break_buyer_assessment(self):
        for failure in (RuntimeError("unavailable checkpoint"), {"status": "error", "error": {"code": "invalid_metadata"}},
                        {"status": "success", "predicted_label": "invalid"}):
            with self.subTest(failure=failure):
                self.predict.side_effect = failure if isinstance(failure, Exception) else None
                self.predict.return_value = failure
                result = self.assess(ai_enabled=True, root_checkpoint="missing.pth")
                self.assertEqual(result["Overall_Result"], "PASS")
                self.assertEqual(result["AI_Observations"]["root_B"]["status"], "error")

    def test_missing_checkpoint_images_and_duplicate_images(self):
        self.assertEqual(self.assess(ai_enabled=True)["AI_Observations"]["root_B"]["status"], "unavailable")
        images = self.sheets["Images"]
        for frame in (images.iloc[:0], pd.concat([images, images]), pd.DataFrame({"bad": [1]})):
            self.sheets["Images"] = frame
            result = self.assess(ai_enabled=True, root_checkpoint="root.pth")
            self.assertEqual(result["Overall_Result"], "PASS")
            self.assertEqual(result["AI_Observations"]["root_B"]["status"], "unavailable")
        self.sheets["Images"] = images
        self.path.return_value.is_file.return_value = False
        self.assertEqual(self.assess(ai_enabled=True, root_checkpoint="root.pth")["AI_Observations"]["root_B"]["status"], "unavailable")
        self.predict.assert_not_called()

    def test_unsupported_plant_and_missing_root_status(self):
        self.sheets["Samples"]["Observed_Root_Status"] = None
        self.assertEqual(self.assess(ai_enabled=True, root_checkpoint="root.pth")["AI_Observations"]["root_B"]["status"], "unavailable")
        self.sheets["Samples"]["Observed_Root_Status"] = "Rooted"
        for frame in self.sheets.values():
            if "Plant_ID" in frame:
                frame["Plant_ID"] = "P08"
        result = self.assess(ai_enabled=True, root_checkpoint="root.pth")
        self.assertEqual(result["AI_Observations"]["root_B"]["status"], "unsupported")
        self.assertEqual(result["Overall_Result"], "PASS")
        self.predict.assert_not_called()


if __name__ == "__main__":
    main()
