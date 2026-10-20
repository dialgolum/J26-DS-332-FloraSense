"""
Preprocessing step 2: preview augmentations and save a figure for slides.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from services.c2_disease_detection import config
from services.c2_disease_detection.training.scripts.nursery_data import load_metadata
from services.c2_disease_detection.vision.preprocessing.augmentations import (
    IMAGENET_MEAN, IMAGENET_STD, get_inference_transform, get_train_transform, load_image)

MEAN, STD = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)

def to_img(t: torch.Tensor) -> np.ndarray:
    return (t * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=config.REPORTS_DIR / "augmentation_preview.png")
    args = ap.parse_args()

    df, _ = load_metadata(args.metadata)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    clean, aug = get_inference_transform(), get_train_transform()
    picks = []
    for c in list(config.CORE_CLASSES) + [config.OOD_FOLDER]:
        sub = df[df["label"] == c]
        if len(sub):
            picks += [(c, p) for p in sub["path"].sample(min(args.per_class, len(sub)), random_state=args.seed)]
    cols = 2 + args.views
    fig, axes = plt.subplots(len(picks), cols, figsize=(1.9 * cols, 1.9 * len(picks)), squeeze=False)
    for r, (label, path) in enumerate(picks):
        img = load_image(path)
        tiles = [("original", np.asarray(img.copy().resize((224, 224)))), ("preprocessed", to_img(clean(img)))]
        tiles += [(f"aug {i + 1}", to_img(aug(img))) for i in range(args.views)]
        for c, (title, arr) in enumerate(tiles):
            ax = axes[r][c]
            ax.imshow(arr)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("Preprocessing (224x224, ImageNet normalisation) and greenhouse augmentation", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()