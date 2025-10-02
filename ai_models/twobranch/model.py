"""Two-Branch CNN model definition.

Structure:
- Shared backbone (chosen via torchvision) applied separately to front/back.
- Each branch outputs an embedding (L2-normalized optional).
- Fusion by concatenation -> optional projection -> classifier head.
- Optional dropout & batchnorm.

The model is intentionally lightweight to fit latency/memory targets.
"""
from __future__ import annotations
from typing import Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
except Exception as e:  # pragma: no cover
    models = None  # type: ignore


class GeM(nn.Module):  # Generalized Mean Pooling (optional)
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = torch.clamp(x, min=self.eps)
        x = x.pow(self.p).mean(dim=(-1, -2)).pow(1.0 / self.p)
        return x


def _get_backbone(name: str):
    """Return backbone feature extractor and its raw feature dimension.

    Honors env var TB_PRETRAINED=0 to disable downloading pretrained weights
    (useful in offline / fast CI contexts). Falls back gracefully if
    torchvision weights cannot be loaded.
    """
    if models is None:
        raise RuntimeError("torchvision not available. Install torchvision to use this model.")
    use_pretrained = os.getenv("TB_PRETRAINED", "1") == "1"
    try:
        if name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
            m = models.mobilenet_v2(weights=weights)
            feat_dim = 1280
            backbone = m.features
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
            m = models.efficientnet_b0(weights=weights)
            feat_dim = 1280
            backbone = m.features
        else:
            raise ValueError(f"Unsupported backbone: {name}")
    except Exception as e:  # pragma: no cover
        # Fallback: retry without pretrained weights if initial attempt failed
        if use_pretrained:
            if name == "mobilenet_v2":
                m = models.mobilenet_v2(weights=None)
                feat_dim = 1280
                backbone = m.features
            elif name == "efficientnet_b0":
                m = models.efficientnet_b0(weights=None)
                feat_dim = 1280
                backbone = m.features
            else:
                raise
        else:
            raise
    return backbone, feat_dim


class TwoBranchCNN(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "mobilenet_v2", embedding_dim: int = 256, dropout: float = 0.2, pooling: str = "avg", use_batchnorm: bool = True):
        super().__init__()
        self.backbone_name = backbone
        self.shared_backbone, feat_dim = _get_backbone(backbone)

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        elif pooling == "gem":
            self.pool = GeM()
        else:
            raise ValueError("Unsupported pooling type")

        proj_layers = []
        proj_layers.append(nn.Linear(feat_dim, embedding_dim))
        if use_batchnorm:
            proj_layers.append(nn.BatchNorm1d(embedding_dim))
        proj_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            proj_layers.append(nn.Dropout(dropout))
        self.projection = nn.Sequential(*proj_layers)

        fusion_in = embedding_dim * 2
        classifier_layers = [nn.Linear(fusion_in, num_classes)]
        self.classifier = nn.Sequential(*classifier_layers)

    def forward_branch(self, x):
        x = self.shared_backbone(x)
        x = self.pool(x) if not isinstance(self.pool, GeM) else self.pool(x)
        if x.dim() == 4:
            x = x.flatten(1)
        x = self.projection(x)
        return x

    def forward(self, front: torch.Tensor, back: torch.Tensor):
        emb_front = self.forward_branch(front)
        emb_back = self.forward_branch(back)
        fused = torch.cat([emb_front, emb_back], dim=1)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "embeddings": (emb_front, emb_back),
            "fused": fused,
        }


__all__ = ["TwoBranchCNN"]
