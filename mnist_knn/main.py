# -*- coding: utf-8 -*-
r"""
main.py
— Entry point to run steps (a–g), grid search logging, and optional figure generation.

Changes for Step 4:
- Add CLI flag --make_figs to generate 10 publication-style figures into runs/figs/
  after (g) finishes and artifacts (runs/val_grid.csv, runs/test_pred.pt) exist.
"""
from __future__ import annotations
import argparse
import os
import time
import torch

from .data import load_mnist, set_seed
from .knn import (
    knn_single,
    knn_all_naive_broadcast,
    knn_all_cdist_batched,
)
from .utils import accuracy
from .experiments import run_grid_on_val, evaluate_on_test


def parse_args():
    p = argparse.ArgumentParser(description="MNIST kNN (a–g) pipeline • Step3+4: logging & figures")
    # data & basics
    p.add_argument("--root", type=str, default="./data", help=r"MNIST data directory (e.g., .\data)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (affects train/val split)")
    p.add_argument("--flatten", action="store_true", help="Flatten images to [784] (default on)")
    p.add_argument("--no-flatten", dest="flatten", action="store_false", help="Keep [1,28,28]")
    p.set_defaults(flatten=True)

    # hyper-params (also used by a/b/c/d runs)
    p.add_argument("--k", type=int, default=5, help="k for kNN (default 5)")
    p.add_argument("--metric", type=str, default="l2", choices=["l2", "l1", "cosine"], help="Distance metric")
    p.add_argument("--vote", type=str, default="majority", choices=["majority", "distance"], help="Voting strategy")
    p.add_argument("--batch_size", type=int, default=1024, help="(d) initial batch size (can adapt down)")

    # device & precision (from Step 2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device policy: auto prefers CUDA else CPU")
    p.add_argument("--adaptive_batch", action="store_true", help="Enable adaptive batch size on CUDA OOM")
    p.add_argument("--no_adaptive_batch", dest="adaptive_batch", action="store_false")
    p.set_defaults(adaptive_batch=True)
    p.add_argument("--min_batch_size", type=int, default=32, help="Lower bound for adaptive batch size")
    p.add_argument("--half", action="store_true", help="Store tensors as float16 on CUDA (experimental)")

    # (c) skip naive broadcast (OOM demo)
    p.add_argument("--skip_naive", action="store_true", help="Skip (c) naive broadcast (OOM demonstration)")

    # (e)(f) grid search — default ON
    p.add_argument("--do_grid", dest="do_grid", action="store_true", help="Run grid search (default on)")
    p.add_argument("--no_grid", dest="do_grid", action="store_false", help="Disable grid search")
    p.set_defaults(do_grid=True)
    p.add_argument("--k_list", nargs="+", type=int, default=[3, 5, 7], help="k search list (default 3 5 7)")
    p.add_argument("--metric_list", nargs="+", type=str, default=["l2", "l1", "cosine"], help="metric search list")
    p.add_argument("--vote_list", nargs="+", type=str, default=["majority", "distance"], help="vote search list")
    p.add_argument("--batch_list", nargs="+", type=int, default=[256, 512, 1024], help="batch_size search list")

    # outputs
    p.add_argument("--out_dir", type=str, default="./runs", help="Directory for logs/CSV/preds")
    p.add_argument("--save_val_pred", action="store_true", help="Save val predictions to runs/val_pred.pt")

    # Step 4: figures
    p.add_argument("--make_figs", action="store_true",
                   help="Generate Nature-style figures into runs/figs after evaluation")

    return p.parse_args()


def pick_device(policy: str) -> torch.device:
    if policy == "cuda":
        return torch.device("cuda")
    if policy == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_cast_half(t: torch.Tensor, use_half: bool, device: torch.device) -> torch.Tensor:
    """If use_half and on CUDA, cast to float16; otherwise keep float32."""
    if use_half and device.type == "cuda":
        return t.half()
    return t.float()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    print(f"[ENV] torch={torch.__version__} policy={args.device} -> device={device} "
          f"cuda_available={torch.cuda.is_available()} half={args.half} grid={args.do_grid}")

    # ===== data =====
    data = load_mnist(root=args.root, flatten=args.flatten, seed=args.seed)
    X_train = maybe_cast_half(data['X_train'], args.half, device)
    X_val   = maybe_cast_half(data['X_val'],   args.half, device)
    X_test  = maybe_cast_half(data['X_test'],  args.half, device)
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print(f"[DATA] train={len(y_train)} val={len(y_val)} test={len(y_test)} dim={X_train.shape[1:]}")

    # ===== (a) single-sample iterative =====
    print("\n==== (a) single-sample kNN: iterative ====")
    x0 = X_val[0].to(device)
    gt0 = int(y_val[0].item())
    t0 = time.time()
    pred_a = knn_single(x0, X_train.to(device), y_train.to(device),
                        args.k, args.metric, args.vote, 'iter')
    t_a = time.time() - t0
    print(f"(a) pred={pred_a} gt={gt0} time={t_a:.3f}s")

    # ===== (b) single-sample broadcast =====
    print("\n==== (b) single-sample kNN: broadcast ====")
    t0 = time.time()
    pred_b = knn_single(x0, X_train.to(device), y_train.to(device),
                        args.k, args.metric, args.vote, 'broadcast')
    t_b = time.time() - t0
    same = (pred_a == pred_b)
    print(f"(b) pred={pred_b} gt={gt0} time={t_b:.3f}s (same as (a)? {same})")

    # ===== (c) naive broadcast (demo; possible OOM) =====
    print("\n==== (c) naive broadcast: full-val demo ====")
    if args.skip_naive:
        print("(c) skipped by --skip_naive")
    else:
        try:
            t0 = time.time()
            y_pred_c = knn_all_naive_broadcast(X_val.to(device), X_train.to(device), y_train.to(device),
                                               args.k, args.metric, args.vote)
            t_c = time.time() - t0
            acc_c = accuracy(y_pred_c, y_val.to(device))
            print(f"(c) val_acc={acc_c:.4f} time={t_c:.3f}s  [note: huge memory usage]")
        except RuntimeError as e:
            print(f"(c) failed (likely OOM): {str(e)[:200]}")

    # ===== (d) batched cdist (adaptive & fallback) =====
    print("\n==== (d) batched cdist on validation (adaptive batch) ====")
    cur_device = device
    try:
        t0 = time.time()
        y_pred_d = knn_all_cdist_batched(
            X_val, X_train, y_train,
            k=args.k, metric=args.metric, vote=args.vote,
            batch_size=args.batch_size, device=cur_device,
            adaptive_batch=args.adaptive_batch, min_batch_size=args.min_batch_size,
            show_progress=True
        )
        t_d = time.time() - t0
    except RuntimeError as e:
        msg = str(e).lower()
        if args.device == "auto" and "out of memory" in msg and cur_device.type == "cuda":
            print("[WARN] CUDA OOM repeatedly; fallback to CPU (auto policy).")
            cur_device = torch.device("cpu")
            t0 = time.time()
            y_pred_d = knn_all_cdist_batched(
                X_val.float(), X_train.float(), y_train,
                k=args.k, metric=args.metric, vote=args.vote,
                batch_size=max(args.batch_size, 1024), device=cur_device,
                adaptive_batch=False, min_batch_size=args.min_batch_size,
                show_progress=True
            )
            t_d = time.time() - t0
        else:
            raise

    acc_d = accuracy(y_pred_d, y_val.to(cur_device))
    per_img = (t_d / len(y_val)) if len(y_val) > 0 else float('nan')
    print(f"(d) val_acc={acc_d:.4f} time={t_d:.3f}s ({per_img*1e3:.3f} ms/img on {cur_device})")

    # optional: save val predictions
    if args.save_val_pred:
        os.makedirs(args.out_dir, exist_ok=True)
        save_path = os.path.join(args.out_dir, "val_pred.pt")
        torch.save({"y_pred": y_pred_d.cpu(), "y_val": y_val}, save_path)
        print(f"[SAVE] val predictions -> {save_path}")

    # ===== (e)(f) grid search =====
    if args.do_grid:
        print("\n==== (e)(f) grid search: k/metric/vote/batch_size (default on) ====")
        os.makedirs(args.out_dir, exist_ok=True)
        best, recs = run_grid_on_val(
            X_train, y_train, X_val, y_val,
            k_list=args.k_list,
            metric_list=[m for m in args.metric_list if m in ("l2", "l1", "cosine")],
            vote_list=[v for v in args.vote_list if v in ("majority", "distance")],
            batch_list=args.batch_list,
            device=cur_device,
            csv_path=os.path.join(args.out_dir, "val_grid.csv")
        )
    else:
        best = {"k": args.k, "metric": args.metric, "vote": args.vote,
                "batch_size": args.batch_size, "val_acc": acc_d, "val_time": t_d}
        print("\n[INFO] grid search disabled (--no_grid). Using (d) config as best candidate.")

    # ===== (g) test evaluation & export =====
    print("\n==== (g) evaluate best config on test ====")
    test_res = evaluate_on_test(X_train, y_train, X_test, y_test, best, cur_device)

    os.makedirs(args.out_dir, exist_ok=True)
    test_save = os.path.join(args.out_dir, "test_pred.pt")
    torch.save({"y_pred": test_res["y_pred"].cpu(), "y_test": y_test}, test_save)
    print(f"[SAVE] test predictions -> {test_save}")

    # report-friendly summary
    summary = (f"BEST(k={best['k']}, metric={best['metric']}, vote={best['vote']}, bsz={best['batch_size']}): "
               f"val_acc={best.get('val_acc', float('nan')):.4f}, val_time={best.get('val_time', float('nan')):.3f}s, "
               f"test_acc={test_res['test_acc']:.4f}, test_time={test_res['test_time']:.3f}s")
    print("\n" + summary + "\n")
    print(f"[DONE] All artifacts in: {args.out_dir}")

    # ===== Step 4: optional figure generation =====
    if args.make_figs:
        try:
            from .figures import generate_all, set_nature_rc
            set_nature_rc()
            figs_dir = args.out_dir.rstrip("/\\") + "/figs"
            generate_all(fig_dir=figs_dir)
        except Exception as e:
            print(f"[WARN] figure generation failed: {e}")


if __name__ == "__main__":
    main()
