"""
Regularized Prototypical Network training with Metric Regularization and Early Stopping.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from services.c2_disease_detection import config
from services.c2_disease_detection.evaluation.metrics.calibration_metrics import (
    expected_calibration_error,
    fit_temperature,
)
from services.c2_disease_detection.training.scripts.nursery_data import (
    load_metadata,
    make_splits,
    summarise,
)
from services.c2_disease_detection.vision.backbone.mobilenet_loader import (
    BACKBONE_FEATURE_DIMS,
    EmbeddingNet,
    get_device,
    save_checkpoint,
)
from services.c2_disease_detection.vision.inference.classifier import prototype_logits
from services.c2_disease_detection.vision.preprocessing.augmentations import (
    get_inference_transform,
    get_train_transform,
    load_image,
)


class _ImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform) -> None:
        self.paths, self.transform = list(paths), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            return self.transform(load_image(self.paths[i])), True
        except Exception:
            return torch.zeros(3, 224, 224), False


@torch.no_grad()
def extract_images(paths: Sequence[Path], transform, batch_size: int = 32) -> Tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(_ImageDataset(paths, transform), batch_size=batch_size, num_workers=0)
    imgs, oks = [], []
    for x, ok in loader:
        imgs.append(x)
        oks.append(ok.numpy())
    if not imgs:
        return torch.zeros(0, 3, 224, 224), np.zeros(0, bool)
    return torch.cat(imgs, dim=0), np.concatenate(oks).astype(bool)


@torch.no_grad()
def extract_features(
    model: EmbeddingNet,
    paths: Sequence[Path],
    transform,
    device: torch.device,
    batch_size: int = 32,
    desc: str = "extracting features",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compatibility helper used by build_prototypes and run_nursery_eval."""
    loader = DataLoader(_ImageDataset(paths, transform), batch_size=batch_size, num_workers=0)
    feats, oks = [], []
    model.eval()
    for x, ok in loader:
        x = x.to(device)
        f = model.extract_features(x).cpu().numpy()
        feats.append(f)
        oks.append(ok.numpy())
    if not feats:
        return np.zeros((0, model.feature_dim), np.float32), np.zeros(0, bool)
    return np.concatenate(feats, axis=0), np.concatenate(oks, axis=0)


def sample_episode(
    images_dict: Dict[str, torch.Tensor],
    classes: Sequence[str],
    n_way: int,
    k: int,
    q: int,
    rng: np.random.Generator,
):
    chosen = list(rng.choice(classes, size=n_way, replace=False))
    sup, qry = [], []
    for c in chosen:
        arr = images_dict[c]  # (n_images, views, 3, 224, 224)
        idx = rng.choice(arr.shape[0], size=k + q, replace=False)
        views = rng.integers(0, arr.shape[1], size=k + q)
        x = arr[idx, views]
        sup.append(x[:k])
        qry.append(x[k:])
    support = torch.stack(sup)
    query = torch.cat(qry, dim=0)
    labels = torch.arange(n_way).repeat_interleave(q)
    return support, query, labels


def episode_forward(
    model: EmbeddingNet,
    support: torch.Tensor,
    query: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
    k: int,
    temperature: float,
    lambda_compact: float,
    device: torch.device,
):
    d = model.embedding_dim
    support_flat = support.view(-1, 3, 224, 224).to(device)
    emb_s = model(support_flat).view(n_way, k, d)
    protos = F.normalize(emb_s.mean(dim=1), dim=1)

    query = query.to(device)
    emb_q = model(query)
    logits = prototype_logits(emb_q, protos, temperature, "cosine")
    labels = labels.to(device)
    ce_loss = F.cross_entropy(logits, labels)

    # Intra-class Centroid Compactness Regularizer
    protos_expanded = protos.unsqueeze(1).expand(-1, k, -1)
    compactness_loss = (1.0 - (emb_s * protos_expanded).sum(dim=-1)).mean()

    total_loss = ce_loss + lambda_compact * compactness_loss
    acc = (logits.argmax(1) == labels).float().mean().item()
    return total_loss, acc


@torch.no_grad()
def validate(model, clean_train_imgs, val_imgs, classes, k, draws, device, seed, temperature):
    model.eval()
    rng = np.random.default_rng(seed)
    emb_tr = {c: model(clean_train_imgs[c].to(device)) for c in classes}
    xs, ys = [], []
    for i, c in enumerate(classes):
        if len(val_imgs.get(c, [])):
            xs.append(model(val_imgs[c].to(device)))
            ys += [i] * len(val_imgs[c])
    if not xs:
        return float("nan"), np.zeros((0, len(classes))), np.zeros(0, int), float("nan")
    q = torch.cat(xs)
    y = torch.tensor(ys, device=device)
    accs, all_cos, all_y = [], [], []
    for _ in range(draws):
        protos = torch.stack([
            emb_tr[c][torch.as_tensor(rng.choice(len(emb_tr[c]), min(k, len(emb_tr[c])), replace=False),
                                      device=device)].mean(0) for c in classes])
        cos = q @ F.normalize(protos, dim=1).T
        accs.append((cos.argmax(1) == y).float().mean().item())
        all_cos.append(cos.cpu().numpy())
        all_y.append(y.cpu().numpy())
    cos_all, y_all = np.concatenate(all_cos), np.concatenate(all_y)
    p = F.softmax(torch.as_tensor(cos_all) / temperature, dim=1).numpy()
    ece = expected_calibration_error(p.max(1), (p.argmax(1) == y_all).astype(float))
    return float(np.mean(accs)), cos_all, y_all, ece


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    ap.add_argument("--out", type=Path, default=config.CHECKPOINT_PATH)
    ap.add_argument("--splits-out", type=Path, default=config.SPLITS_PATH)
    ap.add_argument("--backbone", default="mobilenet_v3_small", choices=list(BACKBONE_FEATURE_DIMS))
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--unfreeze-blocks", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--episodes-per-step", type=int, default=4)
    ap.add_argument("--n-way", type=int, default=4)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--q-query", type=int, default=3)
    ap.add_argument("--aug-views", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.25)
    ap.add_argument("--val-every", type=int, default=15)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--lambda-compact", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)

    df, problems = load_metadata(args.metadata)
    for p in problems:
        print("WARNING:", p)
    df, notes = make_splits(df, args.val_fraction, args.test_fraction, args.seed)
    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["group"]).to_csv(args.splits_out, index=False)
    print("\nImages per class and split (plant-level):\n" + summarise(df))
    for n in notes:
        print("NOTE:", n)

    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    classes = [c for c in config.CORE_CLASSES if (train_df["label"] == c).any()]

    model = EmbeddingNet(args.backbone, pretrained=not args.no_pretrained, freeze_backbone=True, dropout=0.35).to(device)
    if args.unfreeze_blocks > 0:
        model.unfreeze_top_blocks(args.unfreeze_blocks)
        print(f"Fine-tuning enabled: last {args.unfreeze_blocks} backbone blocks unfrozen.")
    else:
        print("Backbone frozen. Training regularized projection head.")

    t0 = time.time()
    train_tf, clean_tf = get_train_transform(), get_inference_transform()
    train_imgs: Dict[str, torch.Tensor] = {}
    clean_train_imgs: Dict[str, torch.Tensor] = {}
    for c in tqdm(classes, desc="caching views"):
        paths = [Path(p) for p in train_df[train_df["label"] == c]["path"]]
        views, keep = [], None
        for v in range(1 + args.aug_views):
            imgs, ok = extract_images(paths, clean_tf if v == 0 else train_tf, args.batch_size)
            keep = ok if keep is None else (keep & ok)
            views.append(imgs)
        arr = torch.stack(views, dim=1)[keep]
        train_imgs[c] = arr
        clean_train_imgs[c] = arr[:, 0]

    val_imgs: Dict[str, torch.Tensor] = {}
    for c in classes:
        paths = [Path(p) for p in val_df[val_df["label"] == c]["path"]]
        if paths:
            imgs, ok = extract_images(paths, clean_tf, args.batch_size)
            val_imgs[c] = imgs[ok]
    print(f"Dataset cached in RAM in {time.time() - t0:.1f}s")

    param_groups = [
        {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": args.lr_head, "weight_decay": args.weight_decay},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr_backbone, "weight_decay": args.weight_decay},
    ]
    optim = torch.optim.AdamW(param_groups)
    steps = max(1, args.episodes // args.episodes_per_step)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)
    rng = np.random.default_rng(args.seed)

    log: List[dict] = []
    best_acc, best_state = -1.0, copy.deepcopy(model.state_dict())
    run_loss, run_acc = [], []
    k_val = min(args.k_shot, min(train_imgs[c].shape[0] for c in classes))
    patience_counter = 0

    print(f"\nBeginning training ({steps} steps, early stopping patience={args.patience})...")
    for step in range(1, steps + 1):
        model.train()
        optim.zero_grad()
        ls = 0.0
        for _ in range(args.episodes_per_step):
            s, qu, y = sample_episode(train_imgs, classes, args.n_way, args.k_shot, args.q_query, rng)
            loss, acc = episode_forward(
                model, s, qu, y, args.n_way, args.k_shot, args.temperature, args.lambda_compact, device
            )
            (loss / args.episodes_per_step).backward()
            ls += loss.item()
            run_acc.append(acc)
        optim.step()
        sched.step()
        run_loss.append(ls / args.episodes_per_step)

        if step % args.val_every == 0 or step == steps:
            v_acc, _, _, _ = validate(
                model, clean_train_imgs, val_imgs, classes, k_val, 25, device, args.seed + 1, args.temperature
            )
            mean_loss = float(np.mean(run_loss))
            mean_train_acc = float(np.mean(run_acc))
            row = {"step": step, "episodes": step * args.episodes_per_step, "train_loss": mean_loss,
                   "train_acc": mean_train_acc, "val_acc": v_acc}
            log.append(row)
            print(f"step {step:4d}/{steps} | loss {mean_loss:.3f} | train acc {mean_train_acc:.3f} | val acc {v_acc:.3f}")
            run_loss, run_acc = [], []

            if v_acc > best_acc:
                best_acc, best_state = v_acc, copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at step {step} (Best Val Acc: {best_acc:.3f})")
                    break

    model.load_state_dict(best_state)
    model.eval()

    n_val = sum(len(v) for v in val_imgs.values())
    t_fit, ece, final_val = args.temperature, float("nan"), float("nan")
    if n_val:
        final_val, cos, ys, _ = validate(model, clean_train_imgs, val_imgs, classes, k_val, 50, device,
                                         args.seed + 2, args.temperature)
        t_fit = fit_temperature(cos, ys)
        _, _, _, ece = validate(model, clean_train_imgs, val_imgs, classes, k_val, 50, device, args.seed + 3, t_fit)

    final = {
        "best_val_acc": best_acc,
        "val_acc": final_val,
        "fitted_temperature": t_fit,
        "val_ece": ece,
        "classes": classes,
        "episode_shape": f"{args.n_way}-way {args.k_shot}-shot",
    }
    print("\nFinal Calibrated Parameters:", json.dumps(final, indent=2))

    save_checkpoint(
        model,
        args.out,
        extra={"temperature": t_fit, "metric": "cosine", "mode": "regularized_compactness",
               "args": vars(args), **final},
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "train_log_nursery.json").write_text(json.dumps({"log": log, "final": final}, indent=2))
    print(f"\nSaved optimal model checkpoint -> {args.out}")


if __name__ == "__main__":
    main()