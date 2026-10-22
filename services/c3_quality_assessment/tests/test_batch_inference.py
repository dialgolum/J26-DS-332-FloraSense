
"""Batch smoke test for the trained visual quality model."""

import sys
from pathlib import Path

COMPONENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_DIR))

from model_inference import predict_visual_quality

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "quality_assessment"
    / "original"
    / "individual"
    / "P01"
)

# Test a few existing images.
IMAGE_NAMES = [
    "P01_S001_front.jpg",
    "P01_S002_front.jpg",
    "P01_S005_front.jpg",
    "P01_S007_front.jpg",
]


def main():
    print("\n--- BATCH VISUAL QUALITY TEST ---")

    successful = 0
    failed = 0

    for image_name in IMAGE_NAMES:
        image_path = IMAGE_DIR / image_name

        if not image_path.is_file():
            print(f"\n[SKIPPED] {image_name}: Image not found")
            failed += 1
            continue

        try:
            result = predict_visual_quality(str(image_path))

            print(f"\nImage: {image_name}")
            print(f"Prediction: {result['Predicted_Quality']}")
            print(f"Confidence: {result['Confidence']:.2%}")
            print(f"Device: {result['Device']}")

            successful += 1

        except Exception as exc:
            print(f"\n[ERROR] {image_name}: {exc}")
            failed += 1

    print("\n--- SUMMARY ---")
    print(f"Successful: {successful}")
    print(f"Failed or skipped: {failed}")


if __name__ == "__main__":
    main()
