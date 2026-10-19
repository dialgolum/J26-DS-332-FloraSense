
"""Integration test for rule-based and AI visual assessment."""

import sys
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


if __name__ == "__main__":
    main()
