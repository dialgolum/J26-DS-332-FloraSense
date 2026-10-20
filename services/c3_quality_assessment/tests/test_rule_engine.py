
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


if __name__ == "__main__":
    unittest.main()
