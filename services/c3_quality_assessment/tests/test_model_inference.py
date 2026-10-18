
"""Basic smoke test for the trained visual quality model."""

from pathlib import Path

from model_inference import predict_visual_quality


# This image is expected to exist in the project's image dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "quality_assessment"
    / "original"
    / "individual"
    / "P01_S010_root.jpg"
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
