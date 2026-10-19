
"""Basic smoke test for the trained visual quality model."""

import sys
from pathlib import Path

COMPONENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_DIR))

from model_inference import predict_visual_quality


# Sample image in the project dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "quality_assessment"
    / "original"
    / "individual"
    / "P01"
    / "P01_S005_front.jpg"
)


if __name__ == "__main__":
    result = predict_visual_quality(IMAGE_PATH)

    print("\n--- VISUAL QUALITY PREDICTION ---")
    print(f"Image: {IMAGE_PATH.name}")
    print(f"Prediction: {result['Predicted_Quality']}")
    print(f"Confidence: {result['Confidence']:.2%}")
    print("Class probabilities:")

    for label, probability in result["Probabilities"].items():
        print(f"  {label}: {probability:.2%}")

    print(f"Device: {result['Device']}")
