"""Configuration objects for Two-Branch CNN model.

Separating config centralizes hyperparameters and facilitates future
sweeps / overrides via environment variables or CLI.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List
import os


def _get_env(name: str, default, cast):
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        return default


@dataclass
class OptimConfig:
    lr: float = _get_env("TB_LR", 1e-3, float)
    weight_decay: float = _get_env("TB_WEIGHT_DECAY", 1e-4, float)
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_epochs: int = _get_env("TB_WARMUP_EPOCHS", 2, int)
    scheduler: str = _get_env("TB_SCHEDULER", "cosine", str)  # cosine | step | none


@dataclass
class TrainConfig:
    epochs_head: int = _get_env("TB_EPOCHS_HEAD", 5, int)
    epochs_finetune: int = _get_env("TB_EPOCHS_FT", 15, int)
    batch_size: int = _get_env("TB_BATCH_SIZE", 16, int)
    num_workers: int = _get_env("TB_NUM_WORKERS", 2, int)
    mixed_precision: bool = _get_env("TB_MIXED_PRECISION", "1", str) == "1"
    grad_clip: float = _get_env("TB_GRAD_CLIP", 5.0, float)
    freeze_backbone_epochs: int = _get_env("TB_FREEZE_EPOCHS", 3, int)


@dataclass
class ModelConfig:
    backbone: str = _get_env("TB_BACKBONE", "mobilenet_v2", str)  # mobilenet_v2 | efficientnet_b0
    embedding_dim: int = _get_env("TB_EMB_DIM", 256, int)
    dropout: float = _get_env("TB_DROPOUT", 0.2, float)
    use_batchnorm: bool = _get_env("TB_USE_BN", "1", str) == "1"
    pooling: str = _get_env("TB_POOL", "avg", str)  # avg | gem | max


@dataclass
class DataConfig:
    img_size: Tuple[int, int] = (224, 224)
    augment: bool = _get_env("TB_AUG", "1", str) == "1"
    color_jitter: float = _get_env("TB_COLOR_JITTER", 0.1, float)
    random_erasing: float = _get_env("TB_RANDOM_ERASE", 0.0, float)
    # Advanced preprocessing toggles
    use_advanced_preprocess: bool = _get_env("TB_ADV_PRE", "0", str) == "1"
    preprocess_mode: str = _get_env("TB_PRE_MODE", "refine", str)  # quick | refine
    bg_mode: str = _get_env("TB_BG_MODE", "zero", str)  # zero | mean | blur
    pad_px: int = _get_env("TB_PAD_PX", 12, int)
    backbone_preprocess: str = _get_env("TB_BACKBONE_PRE", "torchvision", str)  # torchvision | mobilenet_v2_tf
    # Augmentation granular controls
    aug_sync: bool = _get_env("TB_AUG_SYNC", "0", str) == "1"  # sync geom for front/back
    aug_rotate_deg: float = _get_env("TB_AUG_ROT", 12.0, float)
    aug_translate: float = _get_env("TB_AUG_TRANS", 0.06, float)
    aug_hflip_prob: float = _get_env("TB_AUG_HFLIP", 0.5, float)
    aug_brightness: float = _get_env("TB_AUG_BRIGHT", 0.15, float)
    aug_contrast: float = _get_env("TB_AUG_CONTRAST", 0.15, float)
    aug_saturation: float = _get_env("TB_AUG_SAT", 0.15, float)


@dataclass
class LoggingConfig:
    log_interval: int = _get_env("TB_LOG_INTERVAL", 20, int)
    ckpt_dir: str = _get_env("TB_CKPT_DIR", "trained_twobranch", str)
    best_metric: str = _get_env("TB_BEST_METRIC", "f1_macro", str)
    save_top_k: int = _get_env("TB_SAVE_TOP_K", 1, int)


@dataclass
class TwoBranchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


__all__ = [
    "OptimConfig",
    "TrainConfig",
    "ModelConfig",
    "DataConfig",
    "LoggingConfig",
    "TwoBranchConfig",
]
