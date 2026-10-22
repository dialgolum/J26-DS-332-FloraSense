"""
Build the initial prototype store (the disease knowledge base) from data/c2_disease/raw.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from services.c2_disease_detection import config
from services.c2_disease_detection.training.scripts.nursery_data import load_splits
from services.c2_disease_detection.training.scripts.train_nursery import extract_features
from services.c2_disease_detection.vision.backbone.mobilenet_loader import get_device, load_checkpoint
from services.c2_disease_detection.vision.preprocessing.augmentations import get_inference_transform
from services.c2_disease_detection.vision.prototypes.prototype_store import PrototypeStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH)
    ap.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.PROTOTYPE_PATH)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--ignore-splits", action="store_true")
    args = ap.parse_args()

    splits_path = args.splits or (config.SPLITS_PATH if config.SPLITS_PATH.exists() and not args.ignore_splits else None)
    split_paths = {}
    if splits_path:
        sdf = load_splits(splits_path)
        sdf = sdf[sdf["split"].isin(["train", "val"])]
        for c in config.CORE_CLASSES:
            split_paths[c] = [Path(x) for x in sdf[sdf["label"] == c]["path"]]
        print(f"Using {splits_path}: TEST photos are excluded from prototypes.")

    device = get_device()
    model, extra = load_checkpoint(args.checkpoint, device)
    store = PrototypeStore(dim=model.embedding_dim)
    rng = random.Random(args.seed)
    used = {}

    for cls in config.CORE_CLASSES:
        imgs = split_paths.get(cls, []) if splits_path else config.list_images(args.raw_dir / cls)
        if not imgs:
            print(f"  [skip] {cls}: no images")
            continue
        if args.k > 0:
            rng.shuffle(imgs)
            imgs = imgs[: args.k]
        feats, ok = extract_features(model, imgs, get_inference_transform(), device, desc=cls)
        feats = feats[ok]
        with torch.no_grad():
            emb = model.project(torch.from_numpy(feats).to(device)).cpu().numpy()
        store.set_class(cls, emb, kind="core", display_name=cls.capitalize())
        used[cls] = [p.name for p, good in zip(imgs, ok) if good]
        print(f"  {cls:10s} prototype from {emb.shape[0]} images")

    if len(store) == 0:
        raise SystemExit("No prototypes built. Check image paths.")
    store.save(args.out)
    (args.out.parent / "prototype_support.json").write_text(json.dumps(used, indent=2))
    print(f"\nSaved {len(store)} prototypes -> {args.out}")


if __name__ == "__main__":
    main()