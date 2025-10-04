# -*- coding: utf-8 -*-
r"""
knn.py
—— kNN 的核心：距离函数、投票策略，以及四种实现形态
(a) 迭代版（单样本）
(b) 广播版（单样本）
(c) 朴素批量广播（一次性对所有新样本）——演示用途，可能爆显存
(d) 分批 + cdist/topk 的高效实现，可在 GPU 上稳妥跑完全量

步骤2增强：
- (d) 支持自适应批大小（CUDA OOM 时自动将 batch_size 减半重试，直到 >= min_batch_size）
- 当 device='cuda' 且失败、且允许回退时，自动切换到 CPU 继续完成（由 main 控制策略）
- 余弦度量数值稳健：半精度输入时强制用 float32 做 normalize 和 matmul，再输出 float32“距离”
- tqdm 进度条（若安装）显示批次进度
"""
from __future__ import annotations
from typing import Tuple, Literal, Optional
import torch
import torch.nn.functional as F

from .utils import maybe_tqdm  # 仅用于显示进度条

Metric = Literal['l2', 'l1', 'cosine']
Vote   = Literal['majority', 'distance']


# =========================
# 距离/相似度的底层计算
# =========================
def pairwise_distance_single_iterative(x: torch.Tensor, X: torch.Tensor, metric: Metric) -> torch.Tensor:
    """(a) 迭代方式：对单个样本 x 与每个训练样本计算距离，返回 [N_train]。"""
    assert x.dim() == 1, "x 应为扁平化的一维向量 [D]"
    dists = []
    if metric == 'cosine':
        x_norm = x / (x.norm(p=2) + 1e-12)
        for i in range(X.shape[0]):
            xi = X[i]
            xi_norm = xi / (xi.norm(p=2) + 1e-12)
            sim = torch.dot(x_norm, xi_norm)
            dists.append(1.0 - sim)
        return torch.stack(dists)
    elif metric == 'l2':
        for i in range(X.shape[0]):
            dists.append(torch.norm(x - X[i], p=2))
        return torch.stack(dists)
    elif metric == 'l1':
        for i in range(X.shape[0]):
            dists.append(torch.norm(x - X[i], p=1))
        return torch.stack(dists)
    else:
        raise ValueError(f"未知 metric: {metric}")


def pairwise_distance_single_broadcast(x: torch.Tensor, X: torch.Tensor, metric: Metric) -> torch.Tensor:
    """(b) 广播方式：对单个样本 x 与训练集 X 的距离（或 1-相似度）。"""
    if metric == 'cosine':
        x_norm = F.normalize(x, p=2, dim=0)
        X_norm = F.normalize(X, p=2, dim=1)
        sims = X_norm @ x_norm  # [N]
        return 1.0 - sims
    elif metric == 'l2':
        diff = X - x.unsqueeze(0)  # [N, D]
        return torch.sqrt(torch.sum(diff * diff, dim=1))
    elif metric == 'l1':
        diff = torch.abs(X - x.unsqueeze(0))  # [N, D]
        return torch.sum(diff, dim=1)
    else:
        raise ValueError(f"未知 metric: {metric}")


def pairwise_distance_block(
    A: torch.Tensor, B: torch.Tensor, metric: Metric
) -> torch.Tensor:
    """
    用于 (d) 的块级两两距离：
    A: [n_a, D], B: [n_b, D]
    返回 dist [n_a, n_b]（若 metric=cosine 则返回“距离”= 1-相似度）
    - l2 / l1: 使用 torch.cdist（p=2/p=1）
    - cosine: 数值稳健实现：强制 float32 计算 normalize & matmul，再返回 float32 距离
    """
    if metric in ('l2', 'l1'):
        p = 2.0 if metric == 'l2' else 1.0
        return torch.cdist(A, B, p=p)
    elif metric == 'cosine':
        # 为了半精度稳健，提升到 float32 做归一化与乘法，然后输出 float32
        A32 = A.float()
        B32 = B.float()
        A_n = F.normalize(A32, p=2, dim=1)
        B_n = F.normalize(B32, p=2, dim=1)
        sim = A_n @ B_n.t()     # [n_a, n_b], float32
        return 1.0 - sim
    else:
        raise ValueError(f"未知 metric: {metric}")


# =========================
# 投票策略（含一致的平票处理）
# =========================
def vote_labels(
    neighbor_labels: torch.Tensor,           # [*, k]
    neighbor_scores: Optional[torch.Tensor], # [*, k]，对 majority 可为 None；对 distance 表示“距离”
    vote: Vote,
    metric: Metric
) -> torch.Tensor:
    """
    - majority：先多数投票；若平票，用“更近（或更相似）的总量”打破；再平选最小标签
    - distance：L1/L2 用 w=1/(dist+eps)，cosine 用相似度 w=(1-dist)
    """
    eps = 1e-12
    if neighbor_labels.dim() == 1:
        neighbor_labels = neighbor_labels.unsqueeze(0)  # [1, k]
        if neighbor_scores is not None:
            neighbor_scores = neighbor_scores.unsqueeze(0)

    B, K = neighbor_labels.shape
    preds = torch.zeros(B, dtype=torch.long, device=neighbor_labels.device)

    for i in range(B):
        labs = neighbor_labels[i]  # [k]
        if vote == 'majority':
            unique, counts = torch.unique(labs, return_counts=True)
            max_count = counts.max()
            cand = unique[counts == max_count]
            if cand.numel() == 1:
                preds[i] = cand.item()
            else:
                if neighbor_scores is not None:
                    if metric == 'cosine':
                        sims = 1.0 - neighbor_scores[i]  # 相似度
                        best_lab, best_score = None, None
                        for c in cand:
                            score = sims[labs == c].sum()
                            if (best_score is None) or (score > best_score):
                                best_score = score
                                best_lab = c
                        preds[i] = best_lab.item()
                    else:
                        d = neighbor_scores[i]
                        best_lab, best_score = None, None
                        for c in cand:
                            score = -(d[labs == c].sum())
                            if (best_score is None) or (score > best_score):
                                best_score = score
                                best_lab = c
                        preds[i] = best_lab.item()
                else:
                    preds[i] = cand.min().item()

        elif vote == 'distance':
            assert neighbor_scores is not None, "distance 加权需要 neighbor_scores"
            if metric == 'cosine':
                weights = 1.0 - neighbor_scores[i]  # 相似度
            else:
                d = neighbor_scores[i]
                weights = 1.0 / (d + eps)

            unique = torch.unique(labs)
            best_lab, best_w = None, None
            for c in unique:
                w_sum = weights[labs == c].sum()
                if (best_w is None) or (w_sum > best_w):
                    best_w = w_sum
                    best_lab = c
            preds[i] = best_lab.item()
        else:
            raise ValueError(f"未知 vote: {vote}")

    return preds if preds.shape[0] > 1 else preds.squeeze(0)


# =========================
# (a) / (b) 单样本 kNN
# =========================
def knn_single(
    x: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor,
    k: int = 5, metric: Metric = 'l2', vote: Vote = 'majority',
    mode: Literal['iter', 'broadcast'] = 'iter'
) -> int:
    """对单样本做 kNN 分类。mode='iter' 对应 (a)，'broadcast' 对应 (b)。"""
    if mode == 'iter':
        dists = pairwise_distance_single_iterative(x, X_train, metric)
    elif mode == 'broadcast':
        dists = pairwise_distance_single_broadcast(x, X_train, metric)
    else:
        raise ValueError("mode 需为 'iter' 或 'broadcast'")

    topk_vals, topk_idx = torch.topk(dists, k=k, largest=False)
    neighbor_labels = y_train[topk_idx]  # [k]
    pred = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)
    return int(pred.item())


# =========================
# (c) 朴素批量广播（演示）
# =========================
def knn_all_naive_broadcast(
    X_test: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor,
    k: int = 5, metric: Metric = 'l2', vote: Vote = 'majority'
) -> torch.Tensor:
    """警告：会构造 [N_test, N_train, D] 巨型张量，极易 OOM，仅作演示。"""
    if metric == 'cosine':
        Xtr_n = F.normalize(X_train, p=2, dim=1)
        Xte_n = F.normalize(X_test,  p=2, dim=1)
        sims = Xte_n @ Xtr_n.t()   # [Nte, Ntr]
        dists = 1.0 - sims
    else:
        diff = X_test.unsqueeze(1) - X_train.unsqueeze(0)  # [Nte, Ntr, D]
        if metric == 'l2':
            dists = torch.sqrt(torch.sum(diff * diff, dim=2))
        else:
            dists = torch.sum(torch.abs(diff), dim=2)

    topk_vals, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)  # [Nte, k]
    neighbor_labels = y_train[topk_idx]
    preds = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)
    return preds


# =========================
# (d) 分批 + cdist：可扩展解法（自适应批大小 & 可选 CPU 回退）
# =========================
@torch.inference_mode()
def knn_all_cdist_batched(
    X_test: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor,
    k: int = 5, metric: Metric = 'l2', vote: Vote = 'majority',
    batch_size: int = 512, device: Optional[torch.device] = None,
    adaptive_batch: bool = True, min_batch_size: int = 32,
    show_progress: bool = True
) -> torch.Tensor:
    """
    使用 torch.cdist（l2/l1）或 1-cosine 相似度的分批方法，避免构造巨大三维张量。
    - 自适应批大小：CUDA OOM 时将 batch_size 减半重试，直到 >= min_batch_size。
    - 若依然失败，抛出异常，让上层（main）决定是否回退到 CPU。
    - 进度条显示（如安装 tqdm）。
    返回：preds [N_test]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 训练集常驻 device
    X_train = X_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    Nte = X_test.shape[0]
    preds = torch.empty(Nte, dtype=torch.long, device=device)

    # 进度条
    iterator = range(0, Nte, batch_size)
    if show_progress:
        iterator = maybe_tqdm(iterator, desc="kNN(d)")

    start_idx = 0
    while start_idx < Nte:
        end_idx = min(start_idx + batch_size, Nte)
        try:
            Xb = X_test[start_idx:end_idx].to(device, non_blocking=True)  # [B, D]
            dists = pairwise_distance_block(Xb, X_train, metric=metric)   # [B, Ntr]
            topk_vals, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)
            neighbor_labels = y_train[topk_idx]
            preds[start_idx:end_idx] = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)

            # 下一批
            start_idx = end_idx
            if show_progress and hasattr(iterator, 'update'):
                iterator.update(1)

            # 释放中间显存
            del Xb, dists, topk_vals, topk_idx, neighbor_labels
            torch.cuda.empty_cache()

        except RuntimeError as e:
            msg = str(e).lower()
            is_cuda_oom = 'out of memory' in msg and device.type == 'cuda'
            if not (adaptive_batch and is_cuda_oom):
                # 非 OOM 或不允许自适应，直接抛出交给上层
                raise
            # OOM：缩小 batch 再试
            new_bsz = max(min_batch_size, batch_size // 2)
            if new_bsz == batch_size:
                # 已经到达最小 batch_size，仍 OOM -> 交给上层处理（可能回退到 CPU）
                raise
            print(f"[WARN] CUDA OOM at batch_size={batch_size}, shrink to {new_bsz} and retry from index {start_idx}...")
            batch_size = new_bsz
            # 重新构造带进度的 iterator（仅用于显示；核心用 start_idx 控制）
            if show_progress:
                iterator = maybe_tqdm(range(0, (Nte + batch_size - 1)//batch_size), desc="kNN(d)-adaptive")

    return preds
