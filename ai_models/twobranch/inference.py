"""Inference wrapper for Two-Branch CNN.

Provides an interface similar to existing EnhancedProductionClassifier.predict_production
for easier API integration & shadow deployment.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from pathlib import Path
from .model import TwoBranchCNN
from .preprocess import preprocess_pair, PreprocessConfig, advanced_object_preprocess, preprocess_pair_advanced
import json
import numpy as np


logger = logging.getLogger("inference")

# Default threshold template (embedded from thresholds_template.json)
DEFAULT_THRESHOLDS = {
    "schema_version": 1,
    "confidence_reject": 0.55,
    "entropy_reject": 1.2,
    "centroid_distance_reject": 2.5,
    "notes": "Template thresholds; calibrate using validation set + OOD set and overwrite."
}


def _normalize_threshold_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize possible short keys to internal *_reject form.

    Accepts either: confidence_reject / entropy_reject / centroid_distance_reject
    or: confidence / entropy / centroid_distance.
    """
    if not raw:
        return {}
    out: Dict[str, Any] = {}
    mapping = {
        'confidence': 'confidence_reject',
        'entropy': 'entropy_reject',
        'centroid_distance': 'centroid_distance_reject'
    }
    for k, v in raw.items():
        if k in mapping:
            out[mapping[k]] = v
        else:
            out[k] = v
    return out


class TwoBranchInference:
    def __init__(self, checkpoint_dir: str, device: str = None, use_advanced: bool = False, thresholds_path: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        # Prefer final fine-tuned model, else stage1
        ckpt_path = None
        for name in ["twobranch_final.pt", "stage1_head.pt"]:
            p = ckpt_dir / name
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path is None:
            raise FileNotFoundError("No valid checkpoint (twobranch_final.pt or stage1_head.pt) found")
        data = torch.load(ckpt_path, map_location=self.device)
        label_encoder = data.get("label_encoder", {})
        # build inverse mapping
        self.idx2label = {v: k for k, v in label_encoder.items()}
        num_classes = len(self.idx2label)
        # heuristic defaults
        self.model = TwoBranchCNN(num_classes=num_classes)
        self.model.load_state_dict(data["model_state"], strict=False)
        self.model.to(self.device).eval()
        self.pre_cfg = PreprocessConfig()
        self.use_advanced = use_advanced
        # Load thresholds / centroids if provided
        self.thresholds: Optional[Dict[str, Any]] = None
        self.centroids: Optional[torch.Tensor] = None
        if thresholds_path is None:
            # attempt auto-discover
            tfile = ckpt_dir / 'thresholds.json'
            if tfile.exists():
                thresholds_path = str(tfile)
        if thresholds_path and Path(thresholds_path).exists():
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                self.thresholds = _normalize_threshold_keys(json.load(f))
            cfile = ckpt_dir / 'centroids.npy'
            if cfile.exists():
                self.centroids = torch.from_numpy(np.load(cfile)).float().to(self.device)

    def _lazy_env_threshold_load(self):
        """Load thresholds/centroids from environment-specified paths if not already loaded.

        Useful for tests where env vars are set after class init.
        """
        if self.thresholds is not None:
            return
        tpath = os.getenv('THRESHOLDS_PATH')
        cpath = os.getenv('CENTROIDS_PATH')
        try:
            if tpath and Path(tpath).exists():
                with open(tpath, 'r', encoding='utf-8') as f:
                    self.thresholds = _normalize_threshold_keys(json.load(f))
            if cpath and Path(cpath).exists():
                self.centroids = torch.from_numpy(np.load(cpath)).float().to(self.device)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Env threshold load failed: {e}")

    def _predict_entropy(self, probs: np.ndarray) -> float:
        eps = 1e-9
        return float(-np.sum(probs * np.log(probs + eps)))

    def _centroid_distance(self, fused_vec: torch.Tensor) -> float:
        if self.centroids is None:
            return 0.0
        # fused_vec shape (F,)
        dists = torch.norm(self.centroids - fused_vec.unsqueeze(0), dim=1)
        return float(torch.min(dists).cpu().item())

    @torch.inference_mode()
    def predict(self, front: np.ndarray, back: np.ndarray, request_id: str = None, use_advanced: bool = None) -> Dict[str, Any]:
        use_adv = self.use_advanced if use_advanced is None else use_advanced
        # attempt env-based lazy load (especially for unit tests)
        self._lazy_env_threshold_load()
        if use_adv:
            # leverage existing advanced pair helper
            adv = preprocess_pair_advanced(front, back, self.pre_cfg, {
                'mode': 'refine', 'pad_px': 12, 'bg_mode': 'zero'
            })
            front_t = adv['tensors']['front'].unsqueeze(0).to(self.device)
            back_t = adv['tensors']['back'].unsqueeze(0).to(self.device)
        else:
            prep = preprocess_pair(front, back, self.pre_cfg, return_tensor=True)
            front_t = prep["tensors"]["front"].unsqueeze(0).to(self.device)
            back_t = prep["tensors"]["back"].unsqueeze(0).to(self.device)

        out = self.model(front_t, back_t)
        logits = out["logits"]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = self.idx2label[pred_idx]

        detailed = [
            {"class_name": self.idx2label[i], "confidence": float(p)}
            for i, p in enumerate(probs)
        ]
        detailed.sort(key=lambda x: x["confidence"], reverse=True)

        confidence = float(probs[pred_idx])
        entropy = self._predict_entropy(probs)
        centroid_d = self._centroid_distance(out['fused'][0])

        status = 'success'
        is_supported = True
        reject_reason = None
        if self.thresholds:
            cthr = self.thresholds.get('confidence_reject')
            ethr = self.thresholds.get('entropy_reject')
            dthr = self.thresholds.get('centroid_distance_reject')
            reason = None
            if cthr is not None and confidence < cthr:
                reason = 'low_confidence'
            elif ethr is not None and entropy > ethr:
                reason = 'high_entropy'
            elif dthr is not None and centroid_d > dthr:
                reason = 'large_centroid_distance'
            if reason:
                status = 'rejected'
                is_supported = False
                reject_reason = {
                    'reason': reason,
                    'confidence': confidence,
                    'entropy': entropy,
                    'centroid_distance': centroid_d,
                    'thresholds': self.thresholds,
                }
                logger.warning(
                    "Rejected prediction",
                    extra={
                        'request_id': request_id,
                        'reason': reason,
                        'confidence': confidence,
                        'entropy': entropy,
                        'centroid_distance': centroid_d,
                        'model_version': 'twobranch_v1'
                    }
                )

        return {
            'status': status,
            'is_supported': is_supported,
            'predicted_class': pred_label if is_supported else None,
            'confidence': confidence,
            'entropy': entropy,
            'centroid_distance': centroid_d,
            'rejection': reject_reason,
            'detailed_results': detailed,
            'model_version': 'twobranch_v1',
            'request_id': request_id,
        }


__all__ = ["TwoBranchInference"]
