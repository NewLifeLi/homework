# -*- coding: utf-8 -*-
r"""
utils.py
—— 通用工具：计时装饰器/上下文、准确率、日志与进度条封装、参数打印等
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
    """若安装 tqdm 则显示进度条，否则原样返回迭代器。"""
    if HAS_TQDM:
        return _tqdm(iterable, desc=desc)
    return iterable


@contextmanager
def timer(msg: str = "elapsed"):
    """简单计时上下文：打印耗时（秒）。"""
    t0 = time.time()
    yield
    t1 = time.time()
    print(f"[TIME] {msg}: {t1 - t0:.3f}s")


def timeit(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """计时执行函数，返回 {'result': ..., 'seconds': ...}。"""
    t0 = time.time()
    result = fn(*args, **kwargs)
    secs = time.time() - t0
    return {"result": result, "seconds": secs}


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """计算分类准确率，输入为 1D 长度相同的张量。"""
    assert y_pred.shape == y_true.shape
    return float((y_pred == y_true).float().mean().item())


def pretty_kv(d: Dict[str, Any]) -> str:
    """将字典打印成 key=value 的一行字符串。"""
    return " ".join([f"{k}={v}" for k, v in d.items()])
