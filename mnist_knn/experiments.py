# -*- coding: utf-8 -*-
r"""
experiments.py
Grid search over k / metric / vote / batch_size on the validation set with timing and CSV export.
Evaluate the best configuration on the test set and optionally return predictions for downstream export.

Artifacts:
- CSV: runs/val_grid.csv with fields {k, metric, vote, batch_size, val_acc, val_time}

Selection rule:
- Maximize val_acc
- Tie-break by smaller val_time
- Then smaller k
- Then lexicographic order of (metric, vote, batch_size)

Console output:
- Top-3 configurations summary
- Best configuration summary
"""
from __future__ import annotations
from typing import Sequence, Dict, Any, Tuple, List
import csv
import os
import torch
from .knn import knn_all_cdist_batched, Metric, Vote
from .utils import accuracy, timeit, pretty_kv


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _sort_key(rec: Dict[str, Any]):
    # Sorting key: (-val_acc, val_time, k, metric, vote, batch_size)
    return (-rec["val_acc"], rec["val_time"], rec["k"], rec["metric"], rec["vote"], rec["batch_size"])


def _print_top(records: List[Dict[str, Any]], topk: int = 3) -> None:
    print("\n[VAL] Top-{} configs:".format(min(topk, len(records))))
    for i, r in enumerate(sorted(records, key=_sort_key)[:topk], 1):
        print(f"  #{i} {pretty_kv(r)}")


def run_grid_on_val(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor,   y_val: torch.Tensor,
    k_list: Sequence[int],
    metric_list: Sequence[Metric],
    vote_list: Sequence[Vote],
    batch_list: Sequence[int],
    device: torch.device,
    csv_path: str = "runs/val_grid.csv"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run grid search on the validation set and return the best record with all records.
    Selection rule: maximize val_acc; then smaller val_time; then smaller k; then (metric, vote, batch_size) lexicographic.
    """
    ensure_dir(os.path.dirname(csv_path))
    print("[VAL] search space:",
          f"k_list={list(k_list)} metric_list={list(metric_list)} "
          f"vote_list={list(vote_list)} batch_list={list(batch_list)}")

    records: List[Dict[str, Any]] = []
    best = None

    for k in k_list:
        for metric in metric_list:
            for vote in vote_list:
                for bsz in batch_list:
                    cfg = {"k": k, "metric": metric, "vote": vote, "batch_size": bsz}
                    print(f"[VAL] running: {pretty_kv(cfg)}")

                    out = timeit(
                        knn_all_cdist_batched,
                        X_val, X_train, y_train,
                        k=k, metric=metric, vote=vote, batch_size=bsz, device=device,
                        adaptive_batch=True, min_batch_size=32, show_progress=False
                    )
                    y_pred = out["result"]
                    val_time = out["seconds"]
                    val_acc = accuracy(y_pred, y_val.to(device))

                    rec = dict(cfg)
                    rec.update({"val_acc": val_acc, "val_time": val_time})
                    records.append(rec)
                    print(f"[VAL] acc={val_acc:.4f} time={val_time:.3f}s")

                    if best is None:
                        best = rec
                    else:
                        # Compare by the sorting key
                        if _sort_key(rec) < _sort_key(best):
                            best = rec

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[VAL] grid results saved to: {csv_path}")

    # Print Top-3 and best
    _print_top(records, topk=3)
    print("[VAL] best:", pretty_kv(best))

    return best, records


def evaluate_on_test(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor,  y_test: torch.Tensor,
    best_cfg: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate the best configuration on the test set.
    Returns:
      {
        "test_acc": float,
        "test_time": float,
        "y_pred": torch.LongTensor [N_test]
      }
    """
    print(f"[TEST] evaluating with best cfg: {pretty_kv(best_cfg)}")
    out = timeit(
        knn_all_cdist_batched,
        X_test, X_train, y_train,
        k=best_cfg["k"],
        metric=best_cfg["metric"],
        vote=best_cfg["vote"],
        batch_size=best_cfg["batch_size"],
        device=device,
        adaptive_batch=True, min_batch_size=32, show_progress=False
    )
    y_pred = out["result"]
    test_time = out["seconds"]
    test_acc = accuracy(y_pred, y_test.to(device))
    print(f"[TEST] acc={test_acc:.4f} time={test_time:.3f}s")
    return {"test_acc": test_acc, "test_time": test_time, "y_pred": y_pred}
