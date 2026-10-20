
from pathlib import Path

from data_loader import load_dataset, validate_dataset, resolve_image_path
from rule_engine import assess_sample


def main():
    sheets = load_dataset()
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

    # Find the image associated with the selected sample.
    image_records = sheets["Images"]

    matching_images = image_records[
        (image_records["Sample_ID"].astype(str).str.strip() == sample_id)
        & (image_records["View_Type"].astype(str).str.strip().str.lower() == "front")
    ]

    image_path = None

    if not matching_images.empty:
        relative_path = matching_images.iloc[0]["Relative_Path"]
        image_path = resolve_image_path(relative_path)

        if not image_path.is_file():
            print(f"\nWarning: Image file not found: {image_path}")
            image_path = None
    else:
        print("\nWarning: No front image found for this sample.")

    # Run the existing rule-based assessment and optional AI inference.
    result = assess_sample(
        sheets,
        sample_id,
        requirement_id,
        image_path=image_path,
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

    print("\nNote: The AI prediction does not change the rule-based result.")


if __name__ == "__main__":
    main()
