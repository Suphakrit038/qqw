"""Dataset + utilities for Two-Branch CNN.

Expects a metadata CSV with columns:
image_front,image_back,label

If image_back is missing/blank, we reuse front as back (graceful degrade).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from .preprocess import preprocess_pair, preprocess_pair_advanced, PreprocessConfig
try:
    # Optional import to allow passing full TwoBranchConfig
    from .config import TwoBranchConfig
except Exception:  # pragma: no cover
    TwoBranchConfig = None  # type: ignore


@dataclass
class TwoBranchSample:
    front_path: str
    back_path: str
    label: str


class TwoBranchDataset(Dataset):
    def __init__(self, metadata_csv: str, label_encoder: Optional[Dict[str, int]] = None, transform: Optional[Callable] = None, preprocess_cfg: Optional[PreprocessConfig] = None, two_branch_config: Any = None, use_advanced: Optional[bool] = None, advanced_options: Optional[Dict[str, Any]] = None):
        self.df = pd.read_csv(metadata_csv)
        required = {"image_front", "image_back", "label"}
        if not required.issubset(self.df.columns):
            missing = required - set(self.df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        self.label_encoder = label_encoder or self._build_label_encoder()
        self.transform = transform
        self.pre_cfg = preprocess_cfg or PreprocessConfig()
        self.tb_cfg = two_branch_config
        # precedence: explicit arg > config.data.use_advanced_preprocess
        if use_advanced is not None:
            self.use_advanced = use_advanced
        else:
            self.use_advanced = bool(getattr(getattr(two_branch_config, 'data', object()), 'use_advanced_preprocess', False)) if two_branch_config else False
        self.advanced_options = advanced_options or {
            'mode': getattr(getattr(two_branch_config, 'data', object()), 'preprocess_mode', 'refine') if two_branch_config else 'refine',
            'pad_px': getattr(getattr(two_branch_config, 'data', object()), 'pad_px', 12) if two_branch_config else 12,
            'bg_mode': getattr(getattr(two_branch_config, 'data', object()), 'bg_mode', 'zero') if two_branch_config else 'zero'
        }

    def _build_label_encoder(self):
        labels = sorted(self.df["label"].unique())
        return {l: i for i, l in enumerate(labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = row.image_front
        bp = row.image_back if isinstance(row.image_back, str) and row.image_back else row.image_front

        front = cv2.imread(fp)
        if front is None:
            raise FileNotFoundError(fp)
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        back = cv2.imread(bp)
        if back is None:
            back = front.copy()
        else:
            back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)

        if self.use_advanced:
            prep = preprocess_pair_advanced(front, back, self.pre_cfg, self.advanced_options)
        else:
            prep = preprocess_pair(front, back, self.pre_cfg, return_tensor=True)
        front_t = prep["tensors"]["front"]
        back_t = prep["tensors"]["back"]

        if self.transform:
            front_t = self.transform(front_t)
            back_t = self.transform(back_t)

        label = self.label_encoder[row.label]
        return {
            "front": front_t,
            "back": back_t,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def num_classes(self):
        return len(self.label_encoder)


__all__ = ["TwoBranchDataset", "TwoBranchSample"]
