# -*- coding: utf-8 -*-
r"""
data.py
Data loading, splitting, tensorization, and device-agnostic utilities.

Notes:
- Reproducibility: set_seed is expected to be called by the entry script.
- Tensors are created as float32 by default; optional half precision casting is handled outside this module.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducible data splitting."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(dataset: Dataset, flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a torchvision Dataset to tensors (X, y).
    If flatten=True, images are reshaped to [784]; otherwise keep [1, 28, 28].
    """
    xs, ys = [], []
    for img, label in dataset:
        if flatten:
            xs.append(img.view(-1))  # -> [784]
        else:
            xs.append(img)           # -> [1, 28, 28]
        ys.append(label)
    X = torch.stack(xs, dim=0).float()  # float32 by default
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def load_mnist(
    root: str,
    flatten: bool = True,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Load MNIST and split the original training set into train/val, with val_ratio for validation.
    Returns a dict with keys: X_train / y_train / X_val / y_val / X_test / y_test.
    """
    set_seed(seed)
    tfm = transforms.ToTensor()

    train_full = datasets.MNIST(root=root, train=True,  download=True, transform=tfm)
    test_set   = datasets.MNIST(root=root, train=False, download=True, transform=tfm)

    X_full, y_full = _to_tensor(train_full, flatten=flatten)

    n_full = X_full.shape[0]
    idx = np.arange(n_full)
    np.random.shuffle(idx)

    n_val = int(n_full * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val,   y_val   = X_full[val_idx],  y_full[val_idx]

    X_test, y_test = _to_tensor(test_set, flatten=flatten)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val':   X_val,
        'y_val':   y_val,
        'X_test':  X_test,
        'y_test':  y_test,
    }
