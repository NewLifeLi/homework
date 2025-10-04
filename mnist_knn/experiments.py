# -*- coding: utf-8 -*-
r"""
experiments.py
—— (e)(f) 网格搜索：超参数（k/metric/vote/batch_size）多配置试验 + 计时 + CSV 导出
—— (g) 使用验证集最优超参在测试集上评估一次，并支持返回预测以导出

步骤3增强：
- 运行开始回显搜索空间
- CSV: runs/val_grid.csv（包含 k, metric, vote, batch_size, val_acc, val_time）
- 选择规则固定：先 val_acc，再 val_time，再 k 更小，最后按 (metric, vote, batch_size) 字典序
- 控制台打印 Top-3 概览与最佳条目
- evaluate_on_test 返回 y_pred，便于主程序导出 runs/test_pred.pt
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
    # 排序键：(-val_acc, val_time, k, metric, vote, batch_size)
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
    在验证集上做网格搜索，返回最优条目（dict）及全部记录（list of dict）。
    选择规则：最大化 val_acc；若相同更短 val_time；再相同 k 更小；再相同按 (metric, vote, batch_size) 字典序。
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
                        # 用排序键比较
                        if _sort_key(rec) < _sort_key(best):
                            best = rec

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[VAL] grid results saved to: {csv_path}")

    # 打印 Top-3 与最佳
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
    使用最优超参在测试集上评估一次。
    返回：
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
