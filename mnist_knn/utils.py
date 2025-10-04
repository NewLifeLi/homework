# -*- coding: utf-8 -*-
r"""
utils.py
General utilities: timing helpers, accuracy, lightweight logging, and tqdm wrapper.
"""
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Callable, Optional, Dict, Any
import torch

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


def maybe_tqdm(iterable, desc: Optional[str] = None):
    """Return a tqdm-wrapped iterator if tqdm is available; otherwise return the iterable unchanged."""
    if HAS_TQDM:
        return _tqdm(iterable, desc=desc)
    return iterable


@contextmanager
def timer(msg: str = "elapsed"):
    """Simple timing context: print elapsed wall time in seconds with a tag."""
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[TIME] {msg}: {t1 - t0:.3f}s")


def timeit(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Execute a function and return a dict with the result and elapsed seconds."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    secs = time.time() - t0
    return {"result": result, "seconds": secs}


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute classification accuracy given two 1D tensors of equal length."""
    assert y_pred.shape == y_true.shape
    return float((y_pred == y_true).float().mean().item())


def pretty_kv(d: Dict[str, Any]) -> str:
    """Render a dict as a single-line 'key=value' string."""
    return " ".join([f"{k}={v}" for k, v in d.items()])
