"""
Image preprocessing and greenhouse-aware augmentation for Component 2.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import torch
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE: int = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ImageSource = Union[str, Path, bytes, Image.Image]

def get_inference_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def get_train_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size,
                scale=(0.6, 1.0),
                ratio=(0.85, 1.15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=25),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

def load_image(source: ImageSource) -> Image.Image:
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(Path(source))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

def preprocess_image(source: ImageSource, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    tensor = get_inference_transform(image_size)(load_image(source))
    return tensor.unsqueeze(0)

if __name__ == "__main__":
    dummy = Image.new("RGB", (640, 480), color=(34, 139, 34))
    x = preprocess_image(dummy)
    assert x.shape == (1, 3, 224, 224), x.shape
    aug = get_train_transform()(dummy)
    assert aug.shape == (3, 224, 224), aug.shape
    print("augmentations OK ->", tuple(x.shape), tuple(aug.shape))