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
    # Aggressive close-up cropping forces focus onto leaf lesions rather than pots or greenhouse benches
    return T.Compose(
        [
            T.RandomResizedCrop(
                image_size,
                scale=(0.35, 0.85),
                ratio=(0.8, 1.25),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=45),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.03),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.25),
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