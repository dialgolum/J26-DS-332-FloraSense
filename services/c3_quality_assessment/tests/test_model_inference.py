
"""Basic smoke test for the trained visual quality model."""

import sys
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image
from torchvision import models

COMPONENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_DIR))

from model_inference import predict_visual_quality
import model_inference as inference


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


class FeatureInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Synthetic untrained weights only; no download or optimization.
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
        cls.state = model.state_dict()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = self.root / "image.png"
        Image.new("RGB", (31, 97), "green").save(self.image)
        from services.c3_quality_assessment.experiment2_training import PREPROCESSING
        self.metadata = {"architecture": "resnet18", "experiment_architecture": "B", "task": "leaf",
            "dataset_version": "c3_features_v2_20261022", "image_size": 224, "fold": 1,
            "class_names": ["One Side", "Both Sides"], "class_to_idx": {"One Side": 0, "Both Sides": 1},
            "preprocessing": copy.deepcopy(PREPROCESSING), "model_state_dict": self.state}

    def checkpoint(self):
        path = self.root / "fold_1.pth"
        torch.save(self.metadata, path)
        return path

    def test_preprocessing_matches_training_original_view_exactly(self):
        from services.c3_quality_assessment.experiment2_training import image_views
        for name, size, orientation in [("portrait.png", (31, 97), None),
                                         ("landscape.jpg", (1900, 500), None),
                                         ("rotated.jpg", (500, 1900), 6)]:
            path = self.root / name
            image = Image.new("RGB", size, "green")
            if orientation:
                exif = image.getexif(); exif[274] = orientation
                image.save(path, exif=exif)
            else:
                image.save(path)
            actual = inference.feature_image_tensor(path, self.metadata["preprocessing"])
            self.assertEqual(tuple(actual.shape), (3, 224, 224))
            self.assertTrue(torch.equal(actual, image_views(path, 42)[0]))

    def test_loading_class_order_and_prediction_structure(self):
        service = inference.FeatureClassifier(self.checkpoint(), "leaf", "B")
        self.assertFalse(service.model.training)
        class FixedLogits(nn.Module):
            def forward(self, tensor):
                return torch.tensor([[2., -1.]], device=tensor.device)
        service.model = FixedLogits()
        result = service.predict(self.image)
        self.assertEqual(result["predicted_label"], "One Side")
        self.assertEqual(list(result["class_probabilities"]), ["One Side", "Both Sides"])
        self.assertAlmostEqual(sum(result["class_probabilities"].values()), 1., places=6)
        self.assertEqual(result["confidence"], result["class_probabilities"]["One Side"])
        self.assertTrue(result["advisory"])
        self.assertTrue(result["checkpoint_id"].startswith("sha256:"))
        self.assertEqual(result["dataset_version"], "c3_features_v2_20261022")
        self.assertEqual(result["architecture"], "resnet18")

    def test_metadata_rejects_wrong_task_variant_mapping_and_recipe(self):
        for key, value in [("task", "root"), ("experiment_architecture", "A"),
                           ("class_to_idx", {"One Side": 1, "Both Sides": 0}),
                           ("dataset_version", "unknown"), ("preprocessing", {}), ("architecture", "resnet50")]:
            metadata = {**self.metadata, key: value}
            with self.subTest(key=key), self.assertRaises(inference.FeatureInferenceError):
                inference.validate_feature_metadata(metadata, "leaf", "B")

    def test_missing_and_invalid_checkpoint_errors(self):
        result = inference.predict_feature(self.image, self.root / "absent.pth", "leaf")
        self.assertEqual(result["error"]["code"], "CHECKPOINT_UNAVAILABLE")
        path = self.root / "broken.pth"; path.write_bytes(b"not a checkpoint")
        self.assertEqual(inference.predict_feature(self.image, path, "leaf")["status"], "error")
        self.metadata["model_state_dict"] = {"fc.weight": torch.zeros(3, 512)}
        result = inference.predict_feature(self.image, self.checkpoint(), "leaf")
        self.assertEqual(result["error"]["code"], "INVALID_WEIGHTS")

    def test_missing_unsupported_corrupt_and_animated_images(self):
        preprocessing = self.metadata["preprocessing"]
        bad = self.root / "bad.png"; bad.write_bytes(b"bad image")
        gif = self.root / "image.gif"; Image.new("RGB", (20, 20)).save(gif)
        animated = self.root / "animated.png"
        Image.new("RGB", (20, 20), "red").save(animated, save_all=True,
            append_images=[Image.new("RGB", (20, 20), "blue")], duration=100)
        for path, code in [(self.root / "missing.jpg", "IMAGE_NOT_FOUND"),
                           (bad, "INVALID_IMAGE"), (gif, "UNSUPPORTED_IMAGE"), (animated, "UNSUPPORTED_IMAGE")]:
            with self.subTest(path=path), self.assertRaises(inference.FeatureInferenceError) as context:
                inference.feature_image_tensor(path, preprocessing)
            self.assertEqual(context.exception.code, code)


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
