
from data_loader import load_dataset, validate_dataset
from rule_engine import assess_sample


def main():
    sheets = load_dataset()
    validate_dataset(sheets)

    print("\nAvailable samples:")
    print(
        sheets["Samples"][
            ["Sample_ID", "Plant_ID"]
        ].to_string(index=False)
    )

    sample_id = input("\nEnter Sample_ID (e.g. S001): ").strip()
    requirement_id = input(
        "Enter Requirement_ID (e.g. R001): "
    ).strip()

    result = assess_sample(sheets, sample_id, requirement_id)

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

    print(f"\nOVERALL RESULT: {result['Overall_Result']}")


if __name__ == "__main__":
    main()
