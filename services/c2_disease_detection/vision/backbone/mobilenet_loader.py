"""
Embedding network: pretrained backbone + regularized 128-D projection head + L2 norm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

EMBEDDING_DIM: int = 128

BACKBONE_FEATURE_DIMS: Dict[str, int] = {
    "mobilenet_v3_small": 576,
    "mobilenet_v2": 1280,
    "resnet18": 512,
}


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_extractor(name: str, pretrained: bool) -> nn.Module:
    if name == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    if name == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    if name == "resnet18":
        w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=w)
        return nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONE_FEATURE_DIMS)}")


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        embedding_dim: int = EMBEDDING_DIM,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.feature_dim = BACKBONE_FEATURE_DIMS[backbone_name]
        self.backbone = build_feature_extractor(backbone_name, pretrained)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        self.freeze_backbone = not trainable

    def unfreeze_top_blocks(self, n_blocks: int = 1) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.backbone_name == "mobilenet_v3_small":
            features = self.backbone[0]
            total_layers = len(features)
            for idx in range(max(0, total_layers - n_blocks), total_layers):
                for p in features[idx].parameters():
                    p.requires_grad = True
        self.freeze_backbone = False

    def train(self, mode: bool = True) -> "EmbeddingNet":
        super().train(mode)
        # Always freeze BatchNorm statistics to prevent batch-size distortion in few-shot regimes
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        return self.backbone(x)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(h), p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.backbone(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(x)
        self.train(was_training)
        return out


def build_model(
    backbone_name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: Optional[torch.device] = None,
) -> EmbeddingNet:
    model = EmbeddingNet(backbone_name, EMBEDDING_DIM, pretrained, freeze_backbone)
    return model.to(device or get_device())


def save_checkpoint(model: EmbeddingNet, path: str | Path, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_extra = json.loads(json.dumps(extra or {}, default=str))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": model.backbone_name,
            "embedding_dim": model.embedding_dim,
            "extra": safe_extra,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, device: Optional[torch.device] = None
) -> Tuple[EmbeddingNet, dict]:
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = EmbeddingNet(
        ckpt["backbone_name"], ckpt["embedding_dim"], pretrained=False, freeze_backbone=True
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt.get("extra", {})