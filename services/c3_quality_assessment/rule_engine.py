
import pandas as pd
if __package__:
    from .model_inference import predict_visual_quality
else:
    from model_inference import predict_visual_quality

def is_yes(value):
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def normalize(value):
    if pd.isna(value):
        return ""
    return "".join(str(value).lower().split())


def assessment_status(value):
    text = normalize(value)

    if text in {"acceptable", "pass", "passed", "yes", "ok"}:
        return "PASS"

    if text in {
        "notacceptable", "fail", "failed", "no", "unacceptable"
    }:
        return "FAIL"

    return "MANUAL REVIEW"


def assess_sample(sheets, sample_id, requirement_id,image_path=None):
    samples = sheets["Samples"]
    plants = sheets["Plant_Types"]
    profiles = sheets["Requirement_Profiles"]
    heights = sheets["Height_Assessments"]
    annotations = sheets["Quality_Annotations"]

    sample_rows = samples[samples["Sample_ID"] == sample_id]
    profile_rows = profiles[
        profiles["Requirement_ID"] == requirement_id
    ]

    if len(sample_rows) != 1:
        raise ValueError(f"Sample not found or duplicated: {sample_id}")

    if len(profile_rows) != 1:
        raise ValueError(
            f"Requirement not found or duplicated: {requirement_id}"
        )

    sample = sample_rows.iloc[0]
    profile = profile_rows.iloc[0]
    plant_rows = plants[plants["Plant_ID"] == sample["Plant_ID"]]

    if len(plant_rows) != 1:
        raise ValueError(f"Plant type not found: {sample['Plant_ID']}")

    plant = plant_rows.iloc[0]

    if profile["Plant_ID"] != sample["Plant_ID"]:
        raise ValueError(
            "The selected requirement belongs to a different plant type."
        )

    results = []

    def add_result(check, status, reason):
        results.append({
            "Check": check,
            "Status": status,
            "Reason": reason,
        })

    # 1. Root status
    expected_root = normalize(profile.get("Root_Status"))
    observed_root = normalize(sample.get("Observed_Root_Status"))

    if not expected_root or not observed_root:
        add_result("Root status", "MANUAL REVIEW",
                   "Root status or requirement is missing.")
    elif expected_root == observed_root:
        add_result("Root status", "PASS", "Root status matches.")
    else:
        add_result("Root status", "FAIL",
                   "Observed root status differs from the requirement.")

    is_rooted = observed_root == "rooted"

    # 2. Packing mode
    expected_mode = normalize(profile.get("Packing_Mode"))
    observed_mode = normalize(sample.get("Observed_Packing_Mode"))

    if not expected_mode or not observed_mode:
        add_result("Packing mode", "MANUAL REVIEW",
                   "Packing mode information is incomplete.")
    elif expected_mode == observed_mode:
        add_result("Packing mode", "PASS", "Packing mode matches.")
    else:
        add_result("Packing mode", "FAIL",
                   "Observed packing mode differs from the requirement.")

    
    # 3. Height result based on the percentage of leaves in range
    height_rows = heights[
        (heights["Sample_ID"] == sample_id)
        & (heights["Requirement_ID"] == requirement_id)
    ]

    if is_yes(plant.get("Height_Check")):
        if len(height_rows) != 1:
            add_result(
                "Height",
                "MANUAL REVIEW",
                "No unique height assessment exists for this profile."
            )
        else:
            height_row = height_rows.iloc[0]
            leaves_in_range = height_row["Leaves_In_Range"]
            total_leaves = height_row["Total_Leaves_Assessed"]

            # Validate the recorded counts before calculating the percentage.
            if (
                pd.isna(leaves_in_range)
                or pd.isna(total_leaves)
                or total_leaves <= 0
                or leaves_in_range < 0
                or leaves_in_range > total_leaves
            ):
                add_result(
                    "Height",
                    "MANUAL REVIEW",
                    "Missing or invalid leaf height assessment counts."
                )
            else:
                percentage = (leaves_in_range / total_leaves) * 100

                if percentage >= 50:
                    status = "PASS"
                elif percentage < 30:
                    status = "FAIL"
                else:
                    status = "MANUAL REVIEW"

                add_result(
                    "Height",
                    status,
                    f"{leaves_in_range}/{total_leaves} leaves "
                    f"({percentage:.2f}%) are within the buyer's "
                    f"height range."
                )


    # 4. Visual annotations
    annotation_rows = annotations[
        (annotations["Sample_ID"] == sample_id)
        & (annotations["Plant_ID"] == sample["Plant_ID"])
    ]

    if len(annotation_rows) != 1:
        add_result("Visual quality", "MANUAL REVIEW",
                   "Missing or duplicated quality annotations.")
    else:
        annotation = annotation_rows.iloc[0]

        # Leaf count: requires a configured minimum in the profile.
        if is_yes(plant.get("Leaf_Count_Check")):
            minimum = profile.get("Min_Total_Leaf_Count")
            count = annotation.get("Total_Leaf_Count")

            if pd.isna(minimum) or pd.isna(count):
                add_result(
                    "Leaf count", "MANUAL REVIEW",
                    "Add Min_Total_Leaf_Count to the requirement profile "
                    "and enter a valid minimum."
                )
            elif count >= minimum:
                add_result("Leaf count", "PASS",
                           f"{count} leaves; minimum is {minimum}.")
            else:
                add_result("Leaf count", "FAIL",
                           f"{count} leaves; minimum is {minimum}.")

        # Leaf distribution applies only when enabled for the plant type.
        if is_yes(plant.get("Leaf_Distribution_Check")):
            add_result(
                "Leaf distribution",
                assessment_status(annotation.get("Leaf_Distribution")),
                f"Observed: {annotation.get('Leaf_Distribution')}"
            )

        # Root distribution applies only to rooted plants.
        if is_rooted and is_yes(plant.get("Root_Distribution")):
            add_result(
                "Root distribution",
                assessment_status(annotation.get("Root_Distribution")),
                f"Observed: {annotation.get('Root_Distribution')}"
            )

        # Oasis condition is relevant only when the packing mode uses oasis.
        if is_rooted and "oasis" in observed_mode:
            if is_yes(plant.get("Oasis_Condition")):
                damaged = normalize(annotation.get("Oasis_Damaged"))

                if damaged in {"no", "false", "0"}:
                    add_result("Oasis condition", "PASS",
                               "No oasis damage recorded.")
                elif damaged in {"yes", "true", "1"}:
                    add_result("Oasis condition", "FAIL",
                               "Oasis damage recorded.")
                else:
                    add_result("Oasis condition", "MANUAL REVIEW",
                               "Oasis condition is missing or unclear.")

        # Overall visual quality
        if is_yes(plant.get("Overall_Quality")):
            add_result(
                "Overall visual quality",
                assessment_status(annotation.get("General_visual_Quality")),
                f"Observed: {annotation.get('General_visual_Quality')}"
            )

    statuses = [item["Status"] for item in results]

    if "FAIL" in statuses:
        overall = "FAIL"
    elif "MANUAL REVIEW" in statuses:
        overall = "MANUAL REVIEW"
    else:
        overall = "PASS"

    
    # 5. Optional AI visual-quality assessment.
    # This result does not change the existing rule-based decision.
    visual_model_result = None

    if image_path is not None:
        try:
            prediction = predict_visual_quality(image_path)

            visual_model_result = {
                "Image": str(image_path),
                "Predicted_Quality": prediction["Predicted_Quality"],
                "Confidence": prediction["Confidence"],
                "Probabilities": prediction["Probabilities"],
                "Device": prediction["Device"],
                "Status": "SUCCESS",
            }

        except Exception as exc:
            # Keep the rule-based assessment available if inference fails.
            visual_model_result = {
                "Image": str(image_path),
                "Status": "ERROR",
                "Error": str(exc),
            }



    return {
        "Sample_ID": sample_id,
        "Plant_ID": sample["Plant_ID"],
        "Requirement_ID": requirement_id,
        "Buyer_ID": profile.get("Buyer_ID"),
        "Overall_Result": overall,
        "Checks": results,
        "Visual_Model": visual_model_result,
    }
