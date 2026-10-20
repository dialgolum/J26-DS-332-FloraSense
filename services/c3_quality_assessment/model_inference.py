
"""Load the trained ResNet18 model and predict plant visual quality."""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Resolve model files relative to this Python file.
COMPONENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = COMPONENT_DIR / "notebooks" / "quality_resnet18.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep these settings consistent with the training notebook.
IMAGE_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_quality_model():
    """Load the saved ResNet18 checkpoint for inference."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint.get(
        "class_names",
        ["Acceptable", "Not Acceptable"],
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def predict_visual_quality(image_path):
    """
    Predict general visual quality from one image.

    Returns the predicted class, confidence, and class probabilities.
    Confidence is a model score, not a guarantee of correctness.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, class_names = load_quality_model()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "Predicted_Quality": class_names[predicted_index],
        "Confidence": round(
            float(probabilities[predicted_index].item()), 4
        ),
        "Probabilities": {
            class_name: round(float(probability.item()), 4)
            for class_name, probability in zip(
                class_names, probabilities
            )
        },
        "Device": str(DEVICE),
    }
