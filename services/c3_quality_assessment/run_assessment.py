
from pathlib import Path

import argparse
import json
import pandas as pd

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.c3_quality_assessment.data_loader import load_dataset, validate_dataset, resolve_image_path
from services.c3_quality_assessment.rule_engine import assess_sample, is_yes, normalize
from services.c3_quality_assessment.model_inference import predict_feature
from services.c3_quality_assessment.prepare_feature_data import normalize_label


def run_assessment(sheets, sample_id, requirement_id, *, ai_enabled=False,
                   root_checkpoint=None, leaf_checkpoint=None, leaf_a_checkpoint=None,
                   device="cpu", include_visual_quality=False):
    """Attach advisory observations to the unchanged expert/buyer-rule result.

    Requirement_ID resolves plant identity. Scope Samples only to disambiguate
    local Sample_IDs; leave annotations, height records and rule logic intact.
    Missing/unknown expert values are never replaced by predictions.
    """
    profiles = sheets["Requirement_Profiles"]
    rows = profiles[profiles.Requirement_ID.eq(requirement_id)]
    if len(rows) != 1:
        raise ValueError(f"Requirement not found or duplicated: {requirement_id}")
    profile = rows.iloc[0]
    plant_id = profile["Plant_ID"]
    scoped = dict(sheets)
    scoped["Samples"] = sheets["Samples"][sheets["Samples"].Plant_ID.eq(plant_id)]

    def image_record(view):
        images = sheets.get("Images")
        if images is None or images.empty:
            return None, "No image records available."
        if not {"Plant_ID", "Sample_ID", "View_Type", "Relative_Path"}.issubset(images.columns):
            return None, "Image records lack required linkage columns."
        matches = images[images.Plant_ID.eq(plant_id) & images.Sample_ID.eq(sample_id)
                         & images.View_Type.astype(str).str.strip().str.casefold().eq(view.casefold())]
        if len(matches) != 1:
            return None, f"Expected one {view} image for {plant_id}:{sample_id}; found {len(matches)}."
        row = matches.iloc[0]
        if pd.isna(row.get("Relative_Path")) or not str(row.Relative_Path).strip():
            return None, f"{view} image path is missing."
        try:
            path = resolve_image_path(row.Relative_Path)
            exists = path.is_file()
        except (OSError, ValueError, TypeError) as exc:
            return None, f"Invalid {view} image path: {exc}"
        if not exists:
            return None, f"{view} image file is missing: {path}"
        return {"image_id": row.get("Image_ID"), "image_path": str(path), "view": view}, None

    visual_path = None
    if include_visual_quality:
        image, _ = image_record("Front")
        if image:
            visual_path = image["image_path"]
    rule = assess_sample(scoped, sample_id, requirement_id, image_path=visual_path)
    sample = scoped["Samples"][scoped["Samples"].Sample_ID.eq(sample_id)].iloc[0]
    plant = sheets["Plant_Types"][sheets["Plant_Types"].Plant_ID.eq(plant_id)].iloc[0]
    annotations = sheets["Quality_Annotations"]
    expert_rows = annotations[annotations.Plant_ID.eq(plant_id) & annotations.Sample_ID.eq(sample_id)]
    disagreements, reasons = [], []
    for check in rule["Checks"]:
        if check["Status"] == "MANUAL REVIEW":
            reasons.append(f"Rule {check['Check']}: {check['Reason']}")

    def observe(task, checkpoint, variant):
        column = "Root_Distribution" if task == "root" else "Leaf_Distribution"
        raw = expert_rows.iloc[0].get(column) if len(expert_rows) == 1 else None
        expert = normalize_label(raw, task)
        expert_status = ("ambiguous" if len(expert_rows) > 1 else "missing" if len(expert_rows) == 0 or pd.isna(raw) or not str(raw).strip()
                         else "available" if expert is not None else "unsupported")
        result = {"task": task, "variant": variant, "advisory": True,
            "observation_type": "root_distribution" if task == "root" else "branch_origin_distribution",
            "definition": ("Root distribution expert feature." if task == "root" else
                "Qualifying branch origins on opposite sides of the main stem; not balanced foliage or equal leaf counts."),
            "expert_annotation": {"status": expert_status, "raw_label": None if pd.isna(raw) else str(raw), "label": expert},
            "comparison_status": "not_compared", "prediction": None, "requires_manual_review": False}

        def unavailable(status, reason):
            result.update(status=status, reason=reason, requires_manual_review=True)
            reasons.append(f"{task} {variant}: {reason}")
            return result

        if not ai_enabled:
            result.update(status="disabled", reason="Feature AI was not requested.")
            return result
        enabled = is_yes(plant.get("Root_Distribution" if task == "root" else "Leaf_Distribution_Check"))
        if not enabled or (task == "root" and normalize(sample.get("Observed_Root_Status")) not in {"", "rooted"}):
            result.update(status="not_applicable", reason="Feature is not applicable to this plant/sample configuration.")
            return result
        supported = {f"P{i:02d}" for i in range(1 if task == "root" else 2, 8)}
        if plant_id not in supported:
            return unavailable("unsupported", "Plant type has no supported Experiment 2 evidence.")
        if task == "root" and not normalize(sample.get("Observed_Root_Status")):
            return unavailable("unavailable", "Observed root status is missing; root inference was not run.")
        if checkpoint is None:
            return unavailable("unavailable", "No explicit checkpoint configured.")
        image, problem = image_record("Root" if task == "root" else "Front")
        if problem:
            return unavailable("unavailable", problem)
        result["image"] = image
        try:
            prediction = predict_feature(image["image_path"], checkpoint, task, variant=variant, device=device)
            result["prediction"] = prediction
            if prediction.get("status") != "success":
                return unavailable("error", str(prediction.get("error", "Inference failed.")))
            label = normalize_label(prediction.get("predicted_label"), task)
            if label is None:
                return unavailable("error", "Inference returned an unsupported feature label.")
        except Exception as exc:
            return unavailable("error", f"Inference failed: {exc}")
        result.update(status="success", requires_manual_review=True,
                      reason="Advisory AI observation; uncalibrated softmax score cannot establish buyer approval.")
        if expert_status == "available":
            result["comparison_status"] = "agreement" if label == expert else "disagreement"
            if label != expert:
                disagreements.append({"task": task, "variant": variant, "expert_label": expert, "ai_label": label})
                reasons.append(f"{task} {variant}: AI disagrees with expert annotation; retain expert/buyer decision.")
        else:
            result["comparison_status"] = "expert_annotation_" + expert_status
            reasons.append(f"{task} {variant}: expert annotation is {expert_status}; AI does not fill the gap.")
        reasons.append(f"{task} {variant}: AI observation requires human review; score is uncalibrated.")
        return result

    observations = {"root_B": observe("root", root_checkpoint, "B"),
                    "leaf_B": observe("leaf", leaf_checkpoint, "B")}
    if leaf_a_checkpoint is not None:
        observations["leaf_A_comparison"] = observe("leaf", leaf_a_checkpoint, "A")
    visual = rule.get("Visual_Model")
    if visual and visual.get("Status") == "ERROR":
        reasons.append("Legacy visual-quality inference failed; buyer-rule decision retained.")
    # Keep every legacy field and add a separate, immutable-in-meaning rule result.
    return {**rule, "Rule_Assessment": rule, "Buyer_Rule_Result": rule["Overall_Result"],
        "Requirement_Profile": {key: None if pd.isna(value) else value.item() if hasattr(value, "item") else value
                                for key, value in profile.to_dict().items()},
        "sample_key": f"{plant_id}:{sample_id}", "AI_Observations": observations,
        "Disagreements": disagreements,
        "Manual_Review": {"required": bool(reasons), "reasons": list(dict.fromkeys(reasons)),
                          "affects_buyer_rule_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Buyer-rule assessment with optional advisory feature AI.")
    parser.add_argument("--sample")
    parser.add_argument("--requirement")
    parser.add_argument("--ai", action="store_true")
    parser.add_argument("--root-checkpoint", type=Path)
    parser.add_argument("--leaf-checkpoint", type=Path)
    parser.add_argument("--leaf-a-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--visual-quality", action="store_true", help="Optional legacy general-quality observation")
    args = parser.parse_args(argv)
    if bool(args.sample) != bool(args.requirement):
        parser.error("Supply both --sample and --requirement, or neither for interactive use.")
    sheets = load_dataset()
    if args.sample:
        result = run_assessment(sheets, args.sample, args.requirement, ai_enabled=args.ai,
            root_checkpoint=args.root_checkpoint, leaf_checkpoint=args.leaf_checkpoint,
            leaf_a_checkpoint=args.leaf_a_checkpoint, device=args.device, include_visual_quality=args.visual_quality)
        print(json.dumps(result, indent=2, default=str))
        return
    validate_dataset(sheets)

    print("\nAvailable samples:")
    print(
        sheets["Samples"][["Sample_ID", "Plant_ID"]]
        .to_string(index=False)
    )

    sample_id = input("\nEnter Sample_ID (e.g. S001): ").strip()
    requirement_id = input(
        "Enter Requirement_ID (e.g. R001): "
    ).strip()

    # Run the existing rule-based assessment and optional AI inference.
    result = run_assessment(
        sheets,
        sample_id,
        requirement_id,
        include_visual_quality=True, ai_enabled=args.ai,
        root_checkpoint=args.root_checkpoint, leaf_checkpoint=args.leaf_checkpoint,
        leaf_a_checkpoint=args.leaf_a_checkpoint, device=args.device,
    )

    print("\n--- ASSESSMENT RESULT ---")
    print(f"Sample: {result['Sample_ID']}")
    print(f"Plant type: {result['Plant_ID']}")
    print(f"Buyer: {result['Buyer_ID']}")
    print(f"Requirement: {result['Requirement_ID']}")

    for check in result["Checks"]:
        print(
            f"{check['Check']}: {check['Status']} "
            f"- {check['Reason']}"
        )

    print(f"\nRULE-BASED OVERALL RESULT: {result['Overall_Result']}")

    # Display the AI result separately.
    visual = result.get("Visual_Model")

    print("\n--- AI VISUAL QUALITY RESULT ---")

    if visual is None:
        print("AI inference was not requested.")
    elif visual["Status"] == "SUCCESS":
        print(f"Image: {visual['Image']}")
        print(f"Prediction: {visual['Predicted_Quality']}")
        print(f"Confidence: {visual['Confidence']:.2%}")
        print(f"Probabilities: {visual['Probabilities']}")
        print(f"Device: {visual['Device']}")
    else:
        print(f"AI inference failed: {visual.get('Error', 'Unknown error')}")

    if args.ai:
        print(json.dumps({key: result[key] for key in ("AI_Observations", "Disagreements", "Manual_Review")}, indent=2, default=str))
    print("\nNote: The AI prediction does not change the rule-based result.")


if __name__ == "__main__":
    main()
