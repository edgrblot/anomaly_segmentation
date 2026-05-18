"""
anomaly_datasets.py
-------------------
Drop-in dataloaders for the 5 anomaly segmentation benchmarks.
Label convention after loading: 0 = InD, 1 = OoD, 255 = ignore.

Usage in your notebook:
    from anomaly_datasets import get_dataset

    dataset = get_dataset("RoadAnomaly",    root=data_path)
    dataset = get_dataset("RoadAnomaly21",  root=data_path)
    dataset = get_dataset("RoadObstacle21", root=data_path)
    dataset = get_dataset("fs_static",      root=data_path)
    dataset = get_dataset("FS_LostFound",   root=data_path)

Each dataset item returns:
    img    : torch.Tensor  [3, H, W]  float32, ImageNet-normalised (matches EoMT input)
    target : torch.Tensor  [H, W]     int64,   values in {0, 1, 255}
"""

import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
IGNORE_INDEX = 255

# ImageNet normalisation — same as EoMT / DINOv2 preprocessing
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────
class _AnomalyDataset(Dataset):
    """
    Generic loader for datasets with the layout:
        <root>/<dataset_name>/images/        ← RGB images (.jpg / .webp / .png)
        <root>/<dataset_name>/labels_masks/  ← grayscale PNGs
    """

    # Subclasses set this to the folder name under `root`
    FOLDER: str = ""

    def __init__(self, root: str):
        self.root = Path(root)
        img_dir  = self.root / self.FOLDER / "images"
        self.img_paths = sorted(
            p for ext in ("*.jpg", "*.webp", "*.png")
            for p in img_dir.glob(ext)
        )
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def _mask_path(self, img_path: Path) -> Path:
        """Default: swap 'images' → 'labels_masks', force .png extension."""
        p = img_path.parent.parent / "labels_masks" / img_path.name
        return p.with_suffix(".png")

    def _load_mask_raw(self, img_path: Path) -> np.ndarray:
        """Load the raw PNG mask as a uint8 numpy array."""
        mask_path = self._mask_path(img_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        return np.array(Image.open(mask_path))

    def _remap(self, raw: np.ndarray) -> np.ndarray:
        """
        Convert dataset-specific pixel values → {0, 1, 255}.
        Override in subclasses that need non-trivial remapping.
        Default: pass-through (assumes mask is already {0, 1, 255}).
        """
        return raw.astype(np.int64)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # ── Image ──────────────────────────────────────────────────────────
        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img)          # [3, H, W] float32 in [0, 1]
        # NO normalisation — EoMT's window_imgs_semantic handles its own preprocessing

        # ── Mask ───────────────────────────────────────────────────────────
        raw     = self._load_mask_raw(img_path)
        label   = self._remap(raw)
        label_t = torch.from_numpy(label).long()

        return img_t, label_t

    @property
    def name(self):
        return self.FOLDER


# ──────────────────────────────────────────────────────────────────────────────
# Dataset subclasses
# ──────────────────────────────────────────────────────────────────────────────

class RoadAnomalyDataset(_AnomalyDataset):
    """
    RoadAnomaly (Hendrycks-style).
    Raw mask values: 0 = InD, 1 = OoD, 2 = OoD (second anomaly class).
    """
    FOLDER = "RoadAnomaly"

    def _remap(self, raw):
        out = raw.astype(np.int64)
        out[raw == 2] = 1          # merge second OoD class into 1
        # anything else stays as-is; no explicit ignore in this dataset
        return out


class RoadAnomaly21Dataset(_AnomalyDataset):
    """
    RoadAnomaly21 (SMIYC track 3).
    Raw mask values: 0 = InD, 1 = OoD, 255 = ignore.
    """
    FOLDER = "RoadAnomaly21"
    # default _remap is fine


class RoadObstacle21Dataset(_AnomalyDataset):
    """
    RoadObstacle21 (SMIYC track 1).
    Raw mask values: 0 = InD, 1 = OoD, 255 = ignore.
    Note: folder is spelled 'RoadObsticle21' on disk (typo preserved).
    """
    FOLDER = "RoadObsticle21"   # ← matches the actual folder name you showed
    # default _remap is fine


class FsStaticDataset(_AnomalyDataset):
    """
    FS Static (Fishyscapes Static).
    Raw mask values: 0 = InD, 1 = OoD, 255 = ignore.
    """
    FOLDER = "fs_static"
    # default _remap is fine


class FSLostFoundDataset(_AnomalyDataset):
    """
    FS LostAndFound (Fishyscapes Lost & Found full).
    Raw mask values:
        0        → ignore  (unlabelled)
        1        → InD
        2–200    → OoD (obstacle classes)
    Remapped to: InD=0, OoD=1, ignore=255.
    """
    FOLDER = "FS_LostFound_full"

    #def _remap(self, raw):
    #    out = np.full_like(raw, IGNORE_INDEX, dtype=np.int64)
    #    out[raw == 1]                        = 0   # InD
    #    out[(raw >= 2) & (raw <= 200)]       = 1   # OoD
    #    # raw == 0 stays 255 (ignore)
    #    return out


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
_REGISTRY = {
    "RoadAnomaly":    RoadAnomalyDataset,
    "RoadAnomaly21":  RoadAnomaly21Dataset,
    "RoadObstacle21": RoadObstacle21Dataset,
    "fs_static":      FsStaticDataset,
    "FS_LostFound":   FSLostFoundDataset,
}

def get_dataset(name: str, root: str) -> _AnomalyDataset:
    """
    Args:
        name : one of 'RoadAnomaly', 'RoadAnomaly21', 'RoadObstacle21',
                       'fs_static', 'FS_LostFound'
        root : path to the ValidationDatasets directory
    Returns:
        A torch Dataset yielding (img_tensor, label_tensor) pairs.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Choose from: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](root)
