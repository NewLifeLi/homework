# -*- coding: utf-8 -*-
r"""
data.py
—— 数据集的下载/读取/划分/张量化/设备管理（Windows 友好：使用原始字符串 docstring）

步骤2说明：
- set_seed 由 main 在程序一开始调用，确保 train/val 划分可复现
- 数据默认以 float32 存放；是否 half 由 main 统一转换，不在 data.py 中处理
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def set_seed(seed: int = 0) -> None:
    """设置随机种子以确保数据划分与实验可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(dataset: Dataset, flatten: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 torchvision Dataset 转为 (X, y) 张量。flatten=True 时扁平化到 [784]。"""
    xs, ys = [], []
    for img, label in dataset:
        if flatten:
            xs.append(img.view(-1))  # -> [784]
        else:
            xs.append(img)           # -> [1,28,28]
        ys.append(label)
    X = torch.stack(xs, dim=0).float()  # 默认 float32
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def load_mnist(
    root: str,
    flatten: bool = True,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    下载并加载 MNIST；从官方训练集随机抽取 10% 作为验证集，其余为训练集。
    返回字典：X_train / y_train / X_val / y_val / X_test / y_test
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
