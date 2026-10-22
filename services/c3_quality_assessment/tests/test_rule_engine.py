
import unittest

from services.c3_quality_assessment.rule_engine import (
    assess_sample,
)


def make_sheets(
    height_result="PASS",
    root_distribution="Acceptable",
    oasis_damaged="No",
    leaf_count=8,
    overall_quality="Acceptable",
):
    """Build a small in-memory dataset for rule-engine tests."""
    import pandas as pd

    return {
        "Samples": pd.DataFrame([{
            "Sample_ID": "S001",
            "Plant_ID": "P01",
            "Observed_Root_Status": "Rooted",
            "Observed_Packing_Mode": "2Oasis",
        }]),
        "Plant_Types": pd.DataFrame([{
            "Plant_ID": "P01",
            "Height_Check": "Yes",
            "Leaf_Count_Check": "Yes",
            "Leaf_Distribution_Check": "No",
            "Root_Distribution": "Yes",
            "Oasis_Condition": "Yes",
            "Overall_Quality": "Yes",
        }]),
        "Requirement_Profiles": pd.DataFrame([{
            "Requirement_ID": "R001",
            "Plant_ID": "P01",
            "Buyer_ID": "B001",
            "Root_Status": "Rooted",
            "Packing_Mode": "2Oasis",
            "Min_Total_Leaf_Count": 8,
        }]),
        "Height_Assessments": pd.DataFrame([{
            "Sample_ID": "S001",
            "Requirement_ID": "R001",
            "Height_Result": height_result,
            "Leaves_In_Range": {"PASS": 4, "FAIL": 1, "MANUAL REVIEW": 3}[height_result],
            "Total_Leaves_Assessed": 8,
        }]),
        "Quality_Annotations": pd.DataFrame([{
            "Sample_ID": "S001",
            "Plant_ID": "P01",
            "Total_Leaf_Count": leaf_count,
            "Root_Distribution": root_distribution,
            "Oasis_Damaged": oasis_damaged,
            "General_visual_Quality": overall_quality,
        }]),
    }


class TestQualityRuleEngine(unittest.TestCase):

    def test_all_checks_pass(self):
        result = assess_sample(make_sheets(), "S001", "R001")
        self.assertEqual(result["Overall_Result"], "PASS")

    def test_height_failure(self):
        result = assess_sample(
            make_sheets(height_result="FAIL"), "S001", "R001"
        )
        self.assertEqual(result["Overall_Result"], "FAIL")

    def test_manual_review(self):
        result = assess_sample(
            make_sheets(height_result="MANUAL REVIEW"),
            "S001",
            "R001",
        )
        self.assertEqual(result["Overall_Result"], "MANUAL REVIEW")

    def test_low_leaf_count(self):
        result = assess_sample(
            make_sheets(leaf_count=6), "S001", "R001"
        )
        self.assertEqual(result["Overall_Result"], "FAIL")

    def test_root_distribution_failure(self):
        result = assess_sample(
            make_sheets(root_distribution="Not Acceptable"),
            "S001",
            "R001",
        )
        self.assertEqual(result["Overall_Result"], "FAIL")

    def test_oasis_damage(self):
        result = assess_sample(
            make_sheets(oasis_damaged="Yes"), "S001", "R001"
        )
        self.assertEqual(result["Overall_Result"], "FAIL")


    def test_leaf_distribution_policy_for_p02_through_p07(self):
        for plant_id in ("P02", "P03", "P04", "P05", "P06", "P07"):
            for label, expected in (("Both Sides", "PASS"), ("One Side", "FAIL"),
                                    ("  bOtH  SiDeS  ", "PASS"), (" one SIDE ", "FAIL"),
                                    (None, "MANUAL REVIEW"), (float("nan"), "MANUAL REVIEW"),
                                    ("", "MANUAL REVIEW"), ("  ", "MANUAL REVIEW"),
                                    ("Unknown", "MANUAL REVIEW"), ("Acceptable", "MANUAL REVIEW")):
                with self.subTest(plant=plant_id, label=label):
                    sheets = make_sheets()
                    for frame in sheets.values():
                        if "Plant_ID" in frame:
                            frame["Plant_ID"] = plant_id
                    sheets["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
                    sheets["Quality_Annotations"]["Leaf_Distribution"] = label
                    result = assess_sample(sheets, "S001", "R001")
                    leaf = next(check for check in result["Checks"] if check["Check"] == "Leaf distribution")
                    self.assertEqual(leaf["Status"], expected)
                    self.assertEqual(leaf["Reason"], f"Observed: {label}")
                    self.assertEqual(result["Overall_Result"], expected)
                    self.assertTrue(all(check["Status"] == "PASS" for check in result["Checks"] if check != leaf))

    def test_p01_and_other_plant_leaf_behavior_unchanged(self):
        for plant_id in ("P01", "P08"):
            for enabled in ("No", "Yes"):
                for label, legacy_status in (("Both Sides", "MANUAL REVIEW"), ("One Side", "MANUAL REVIEW"),
                                             (None, "MANUAL REVIEW"), ("Acceptable", "PASS"), ("Not Acceptable", "FAIL")):
                    with self.subTest(plant=plant_id, enabled=enabled, label=label):
                        sheets = make_sheets()
                        for frame in sheets.values():
                            if "Plant_ID" in frame:
                                frame["Plant_ID"] = plant_id
                        sheets["Plant_Types"]["Leaf_Distribution_Check"] = enabled
                        sheets["Quality_Annotations"]["Leaf_Distribution"] = label
                        result = assess_sample(sheets, "S001", "R001")
                        leaf = [check for check in result["Checks"] if check["Check"] == "Leaf distribution"]
                        self.assertEqual(len(leaf), int(enabled == "Yes"))
                        self.assertEqual(result["Overall_Result"], legacy_status if enabled == "Yes" else "PASS")

    def test_leaf_pass_does_not_mask_other_failures_or_missing_height(self):
        for height, root, expected in (("FAIL", "Acceptable", "FAIL"),
                                       ("MANUAL REVIEW", "Acceptable", "MANUAL REVIEW"),
                                       ("PASS", "Not Acceptable", "FAIL")):
            sheets = make_sheets(height_result=height, root_distribution=root)
            for frame in sheets.values():
                if "Plant_ID" in frame:
                    frame["Plant_ID"] = "P02"
            sheets["Plant_Types"]["Leaf_Distribution_Check"] = "Yes"
            sheets["Quality_Annotations"]["Leaf_Distribution"] = "Both Sides"
            self.assertEqual(assess_sample(sheets, "S001", "R001")["Overall_Result"], expected)


if __name__ == "__main__":
    unittest.main()
