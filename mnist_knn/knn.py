# -*- coding: utf-8 -*-
r"""
knn.py
Core components for k-Nearest Neighbors:
- Distance functions and voting rules
- Four implementations:
  (a) iterative single-sample
  (b) broadcast single-sample
  (c) naive full broadcast (memory-inefficient; for demonstration)
  (d) batched + torch.cdist/topk (scalable)

Enhancements:
- (d) supports adaptive batch sizing (halve on CUDA OOM until >= min_batch_size)
- cosine metric uses float32 normalization/matmul for numerical stability with half inputs
- optional progress bar via tqdm (if available)
"""
from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn.functional as F

from .utils import maybe_tqdm  # progress bar helper (no-op if tqdm is unavailable)

Metric = Literal['l2', 'l1', 'cosine']
Vote   = Literal['majority', 'distance']


# =========================
# Pairwise distances (single sample)
# =========================
def pairwise_distance_single_iterative(x: torch.Tensor, X: torch.Tensor, metric: Metric) -> torch.Tensor:
    """
    (a) Iterative distances between a single query x [D] and all rows in X [N, D].
    Returns a tensor of shape [N].
    """
    assert x.dim() == 1, "x must be a 1D flattened vector [D]"
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
        raise ValueError(f"Unknown metric: {metric}")


def pairwise_distance_single_broadcast(x: torch.Tensor, X: torch.Tensor, metric: Metric) -> torch.Tensor:
    """
    (b) Broadcast distances between a single query x [D] and all rows in X [N, D].
    Returns a tensor of shape [N].
    """
    if metric == 'cosine':
        x_norm = F.normalize(x, p=2, dim=0)
        X_norm = F.normalize(X, p=2, dim=1)
        sims = X_norm @ x_norm  # [N]
        return 1.0 - sims
    elif metric == 'l2':
        diff = X - x.unsqueeze(0)          # [N, D]
        return torch.sqrt(torch.sum(diff * diff, dim=1))
    elif metric == 'l1':
        diff = torch.abs(X - x.unsqueeze(0))  # [N, D]
        return torch.sum(diff, dim=1)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =========================
# Pairwise distances (block)
# =========================
def pairwise_distance_block(A: torch.Tensor, B: torch.Tensor, metric: Metric) -> torch.Tensor:
    """
    Blockwise pairwise distances between A [n_a, D] and B [n_b, D].
    Returns a distance matrix [n_a, n_b].
    - l2 / l1: torch.cdist with p=2/p=1
    - cosine: compute similarity with float32 normalization/matmul, then return 1 - sim
    """
    if metric in ('l2', 'l1'):
        p = 2.0 if metric == 'l2' else 1.0
        return torch.cdist(A, B, p=p)
    elif metric == 'cosine':
        # Promote to float32 for stable normalization/matmul (esp. when inputs are float16)
        A32 = A.float()
        B32 = B.float()
        A_n = F.normalize(A32, p=2, dim=1)
        B_n = F.normalize(B32, p=2, dim=1)
        sim = A_n @ B_n.t()   # [n_a, n_b], float32
        return 1.0 - sim
    else:
        raise ValueError(f"Unknown metric: {metric}")


# =========================
# Voting rules (with tie-breaking)
# =========================
def vote_labels(
    neighbor_labels: torch.Tensor,            # [*, k]
    neighbor_scores: Optional[torch.Tensor],  # [*, k] distances; can be None for majority
    vote: Vote,
    metric: Metric
) -> torch.Tensor:
    """
    Aggregation over k neighbors.
    - majority: majority vote; on tie, prefer the class with smaller total distance (or larger total similarity for cosine),
      then fall back to the smallest label id.
    - distance: weighted vote; l1/l2 use 1/(dist+eps), cosine uses similarity = (1 - distance).
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
                        sims = 1.0 - neighbor_scores[i]  # similarity
                        best_lab, best_score = None, None
                        for c in cand:
                            score = sims[labs == c].sum()
                            if (best_score is None) or (score > best_score):
                                best_score = score
                                best_lab = c
                        preds[i] = best_lab.item()
                    else:
                        d = neighbor_scores[i]  # distances
                        best_lab, best_score = None, None
                        for c in cand:
                            score = -(d[labs == c].sum())  # smaller total distance is better
                            if (best_score is None) or (score > best_score):
                                best_score = score
                                best_lab = c
                        preds[i] = best_lab.item()
                else:
                    preds[i] = cand.min().item()

        elif vote == 'distance':
            assert neighbor_scores is not None, "distance voting requires neighbor_scores"
            if metric == 'cosine':
                weights = 1.0 - neighbor_scores[i]  # similarity
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
            raise ValueError(f"Unknown vote: {vote}")

    return preds if preds.shape[0] > 1 else preds.squeeze(0)


# =========================
# (a)/(b) single-sample kNN
# =========================
def knn_single(
    x: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor,
    k: int = 5, metric: Metric = 'l2', vote: Vote = 'majority',
    mode: Literal['iter', 'broadcast'] = 'iter'
) -> int:
    """
    Classify a single query with kNN.
    mode='iter' -> (a), mode='broadcast' -> (b).
    """
    if mode == 'iter':
        dists = pairwise_distance_single_iterative(x, X_train, metric)
    elif mode == 'broadcast':
        dists = pairwise_distance_single_broadcast(x, X_train, metric)
    else:
        raise ValueError("mode must be 'iter' or 'broadcast'")

    topk_vals, topk_idx = torch.topk(dists, k=k, largest=False)
    neighbor_labels = y_train[topk_idx]  # [k]
    pred = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)
    return int(pred.item())


# =========================
# (c) naive full broadcast (memory-inefficient)
# =========================
def knn_all_naive_broadcast(
    X_test: torch.Tensor, X_train: torch.Tensor, y_train: torch.Tensor,
    k: int = 5, metric: Metric = 'l2', vote: Vote = 'majority'
) -> torch.Tensor:
    """
    Naive full-broadcast distances for all queries at once.
    This forms a [N_test, N_train, D] tensor for l1/l2 and is memory-heavy.
    """
    if metric == 'cosine':
        Xtr_n = F.normalize(X_train, p=2, dim=1)
        Xte_n = F.normalize(X_test,  p=2, dim=1)
        sims = Xte_n @ Xtr_n.t()   # [N_test, N_train]
        dists = 1.0 - sims
    else:
        diff = X_test.unsqueeze(1) - X_train.unsqueeze(0)  # [N_test, N_train, D]
        if metric == 'l2':
            dists = torch.sqrt(torch.sum(diff * diff, dim=2))
        else:
            dists = torch.sum(torch.abs(diff), dim=2)

    topk_vals, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)  # [N_test, k]
    neighbor_labels = y_train[topk_idx]
    preds = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)
    return preds


# =========================
# (d) batched + cdist (scalable; adaptive batch & optional CPU fallback controlled by caller)
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
    Batched kNN inference using torch.cdist (l2/l1) or (1 - cosine similarity).
    - Adaptive batch sizing: on CUDA OOM, halve batch_size until >= min_batch_size, then re-raise.
    - Progress bar can be shown if tqdm is available.
    Returns predictions: [N_test].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Keep train set resident on device
    X_train = X_train.to(device, non_blocking=True)
    y_train = y_train.to(device, non_blocking=True)

    Nte = X_test.shape[0]
    preds = torch.empty(Nte, dtype=torch.long, device=device)

    # Progress iterator (purely cosmetic)
    iterator = range(0, Nte, batch_size)
    if show_progress:
        iterator = maybe_tqdm(iterator, desc="kNN(d)")

    start_idx = 0
    while start_idx < Nte:
        end_idx = min(start_idx + batch_size, Nte)
        try:
            Xb = X_test[start_idx:end_idx].to(device, non_blocking=True)  # [B, D]
            dists = pairwise_distance_block(Xb, X_train, metric=metric)   # [B, N_train]
            topk_vals, topk_idx = torch.topk(dists, k=k, largest=False, dim=1)
            neighbor_labels = y_train[topk_idx]
            preds[start_idx:end_idx] = vote_labels(neighbor_labels, topk_vals, vote=vote, metric=metric)

            # Advance to next block
            start_idx = end_idx
            if show_progress and hasattr(iterator, 'update'):
                iterator.update(1)

            # Free intermediates
            del Xb, dists, topk_vals, topk_idx, neighbor_labels
            torch.cuda.empty_cache()

        except RuntimeError as e:
            msg = str(e).lower()
            is_cuda_oom = ('out of memory' in msg) and (device.type == 'cuda')
            if not (adaptive_batch and is_cuda_oom):
                # Not an OOM or adaptive disabled -> re-raise
                raise
            # Shrink batch size and retry
            new_bsz = max(min_batch_size, batch_size // 2)
            if new_bsz == batch_size:
                # Already at minimum batch size; let caller decide fallback
                raise
            print(f"[WARN] CUDA OOM at batch_size={batch_size}, shrink to {new_bsz} and retry from index {start_idx}...")
            batch_size = new_bsz
            if show_progress:
                iterator = maybe_tqdm(range(0, (Nte + batch_size - 1) // batch_size), desc="kNN(d)-adaptive")

    return preds
