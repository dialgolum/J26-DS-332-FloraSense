
"""Load the trained ResNet18 model and predict plant visual quality."""

from pathlib import Path
import argparse
import hashlib
import json

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
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


class FeatureInferenceError(ValueError):
    """Actionable feature-inference error; legacy inference remains independent."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def validate_feature_metadata(checkpoint, task, variant=None):
    """Accept only the inspected Experiment 2 recipe; never guess class order."""
    labels = {"root": {"Acceptable", "Not Acceptable"}, "leaf": {"Both Sides", "One Side"}}
    if task not in labels:
        raise FeatureInferenceError("INVALID_TASK", "Task must be root or leaf.")
    if not isinstance(checkpoint, dict):
        raise FeatureInferenceError("INVALID_METADATA", "Checkpoint must contain metadata and model_state_dict.")
    names = checkpoint.get("class_names")
    expected = {"image_size": 224, "content_size": 208, "exif_transpose": True,
                "jpeg_draft_size": [1024, 1024], "resize": "aspect-preserving contain; centered padding",
                "fill_rgb": [127, 127, 127], "mean": [.485, .456, .406], "std": [.229, .224, .225]}
    preprocessing = checkpoint.get("preprocessing")
    valid = (checkpoint.get("architecture") == "resnet18" and checkpoint.get("task") == task
             and checkpoint.get("dataset_version") == "c3_features_v2_20261022"
             and checkpoint.get("image_size") == 224
             and checkpoint.get("experiment_architecture") in {"A", "B"}
             and (variant is None or checkpoint.get("experiment_architecture") == variant)
             and isinstance(names, list) and len(names) == 2 and all(isinstance(n, str) for n in names)
             and set(names) == labels[task]
             and checkpoint.get("class_to_idx") == {name: index for index, name in enumerate(names)}
             and isinstance(preprocessing, dict) and all(preprocessing.get(k) == v for k, v in expected.items())
             and checkpoint.get("fold") in (1, 2, 3)
             and isinstance(checkpoint.get("model_state_dict"), dict))
    if not valid:
        raise FeatureInferenceError("INVALID_METADATA", "Checkpoint task, variant, class mapping, V2 version, fold or preprocessing is incompatible.")
    return names, preprocessing


def feature_image_tensor(image_path, preprocessing):
    """Exact Experiment 2 deterministic evaluation view, including JPEG draft."""
    path = Path(image_path)
    if not path.is_file():
        raise FeatureInferenceError("IMAGE_NOT_FOUND", f"Image not found: {path}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise FeatureInferenceError("UNSUPPORTED_IMAGE", "Only JPEG and PNG images are supported.")
    try:
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG"} or getattr(source, "n_frames", 1) != 1:
                raise FeatureInferenceError("UNSUPPORTED_IMAGE", "Expected a single-frame JPEG or PNG.")
            source.draft("RGB", tuple(preprocessing["jpeg_draft_size"]))
            image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"),
                (preprocessing["content_size"],) * 2, Image.Resampling.BILINEAR)
            size = preprocessing["image_size"]
            canvas = Image.new("RGB", (size, size), tuple(preprocessing["fill_rgb"]))
            canvas.paste(image, ((size-image.width)//2, (size-image.height)//2))
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            mean=preprocessing["mean"], std=preprocessing["std"])])(canvas)
    except FeatureInferenceError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise FeatureInferenceError("INVALID_IMAGE", f"Image could not be decoded: {path}") from exc

#loads the trained ResNet18 model and predicts plant visual quality. 
class FeatureClassifier:
    """Reusable advisory classifier. Explicitly choose a fold, task and A/B arm.

    No best-fold selection, ensemble or inferred file view is performed. Supply
    Root images for root and Front images for leaf. Reuse this instance to avoid
    reloading weights. CPU is the default for local inference.
    """

    def __init__(self, checkpoint_path, task, variant=None, device="cpu"):
        self.task = task
        self.path = Path(checkpoint_path).expanduser().resolve()
        if not self.path.is_file():
            raise FeatureInferenceError("CHECKPOINT_UNAVAILABLE", f"Checkpoint not found: {self.path}")
        try:
            self.device = torch.device(device)
            if self.device.type not in {"cpu", "cuda"} or (self.device.type == "cuda" and not torch.cuda.is_available()):
                raise ValueError("Requested device is unavailable")
            checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise FeatureInferenceError("CHECKPOINT_UNAVAILABLE", f"Cannot load checkpoint on {device}: {exc}") from exc
        self.class_names, self.preprocessing = validate_feature_metadata(checkpoint, task, variant)
        self.variant = checkpoint["experiment_architecture"]
        self.dataset_version = checkpoint["dataset_version"]
        self.fold = checkpoint["fold"]
        try:
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.to(self.device).eval()
        except Exception as exc:
            raise FeatureInferenceError("INVALID_WEIGHTS", f"Weights do not match a two-class ResNet18: {exc}") from exc
        with self.path.open("rb") as stream:
            self.checkpoint_id = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()

    def predict(self, image_path):
        tensor = feature_image_tensor(image_path, self.preprocessing).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probability = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        if probability.shape != (2,) or not torch.isfinite(probability).all():
            raise FeatureInferenceError("INVALID_PREDICTION", "Model returned invalid class probabilities.")
        index = int(probability.argmax())
        return {"status": "success", "task": self.task, "predicted_label": self.class_names[index],
            "confidence": float(probability[index]),
            "class_probabilities": {name: float(probability[i]) for i, name in enumerate(self.class_names)},
            "architecture": "resnet18", "model_variant": self.variant, "dataset_version": self.dataset_version,
            "checkpoint_id": self.checkpoint_id, "checkpoint_path": str(self.path), "fold": self.fold,
            "device": str(self.device), "advisory": True, "score_note": "Uncalibrated softmax model score; expert annotations and buyer decisions are not changed."}


def predict_feature(image_path, checkpoint_path, task, variant=None, device="cpu"):
    """Convenience entry point with structured errors; reuse FeatureClassifier for batches."""
    try:
        return FeatureClassifier(checkpoint_path, task, variant, device).predict(image_path)
    except FeatureInferenceError as exc:
        return {"status": "error", "task": task, "advisory": True,
                "error": {"code": exc.code, "message": str(exc)}}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {"status": "error", "task": task, "advisory": True,
                "error": {"code": "INFERENCE_FAILED", "message": str(exc)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Advisory C3 feature inference on one Root or Front image.")
    parser.add_argument("--task", required=True, choices=["root", "leaf", "visual"])
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit feature fold .pth; never selected by test score")
    parser.add_argument("--variant", choices=["A", "B"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.task == "visual":
        result = predict_visual_quality(args.image)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for root/leaf")
        result = predict_feature(args.image, args.checkpoint, args.task, args.variant, args.device)
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
