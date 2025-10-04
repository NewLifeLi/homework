# -*- coding: utf-8 -*-
r"""
figures.py
— Generate publication-style figures from saved artifacts.

Inputs:
- runs/val_grid.csv       # k, metric, vote, batch_size, val_acc, val_time
- runs/test_pred.pt       # { y_pred: LongTensor[N_test], y_test: LongTensor[N_test] }
- Optional: MNIST dataset # For image grid of misclassifications

Outputs (saved to runs/figs/):
  1  fig01_acc_vs_k_by_metric.png
  2  fig02_metric_vote_bar.png
  3  fig03_time_vs_batch_for_best.png
  4  fig04_throughput_vs_batch.png
  5  fig05_pareto_valacc_valtime.png
  6  fig06_confusion_matrix.png
  7  fig07_per_class_accuracy.png
  8  fig08_misclassified_grid.png
  9  fig09_top1_distance_hist.png
 10  fig10_naive_broadcast_memory.png
"""
from __future__ import annotations
import os
import csv
from typing import List, Dict, Tuple, Any
import math
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

# Optional torchvision for image grid (misclassified samples)
try:
    from torchvision import datasets, transforms
    TORCHVISION_OK = True
except Exception:
    TORCHVISION_OK = False


# ---------------------------
# Matplotlib "Nature-like" rc
# ---------------------------
def set_nature_rc():
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "figure.autolayout": True,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
    })


# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def read_val_grid(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "k": int(r["k"]),
                "metric": r["metric"],
                "vote": r["vote"],
                "batch_size": int(r["batch_size"]),
                "val_acc": float(r["val_acc"]),
                "val_time": float(r["val_time"]),
            })
    return rows


def best_selector_key(rec: Dict[str, Any]):
    # Order: maximize acc, then minimize time, then smaller k, then lexicographic metric/vote/batch
    return (-rec["val_acc"], rec["val_time"], rec["k"], rec["metric"], rec["vote"], rec["batch_size"])


def pick_best(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(records, key=best_selector_key)[0]


def load_test_pred(pt_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    blob = torch.load(pt_path, map_location="cpu")
    y_pred = blob["y_pred"].long()
    y_test = blob["y_test"].long()
    return y_pred, y_test


# ---------------------------
# Figure 1: val_acc vs k by metric
# ---------------------------
def fig_acc_vs_k_by_metric(records: List[Dict[str, Any]], out_path: str):
    # Group by metric; for each k, take the best over vote/batch by acc>time>k
    metrics = sorted({r["metric"] for r in records})
    ks = sorted({r["k"] for r in records})

    data = {m: [] for m in metrics}
    for m in metrics:
        for k in ks:
            sub = [r for r in records if r["metric"] == m and r["k"] == k]
            if not sub:
                data[m].append(np.nan)
                continue
            best = pick_best(sub)
            data[m].append(best["val_acc"])

    set_nature_rc()
    plt.figure(figsize=(3.35, 2.4))  # ~85mm width
    for m in metrics:
        plt.plot(ks, data[m], marker="o", label=m)
    plt.xlabel("k")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy vs k")
    plt.legend(frameon=False)
    plt.xticks(ks)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 2: metric×vote accuracy (grouped bars)
# ---------------------------
def fig_metric_vote_bar(records: List[Dict[str, Any]], out_path: str):
    # For each (metric, vote), choose best across batch_size (acc>time>k)
    pairs = sorted({(r["metric"], r["vote"]) for r in records})
    accs, labels = [], []
    for m, v in pairs:
        sub = [r for r in records if r["metric"] == m and r["vote"] == v]
        best = pick_best(sub)
        accs.append(best["val_acc"])
        labels.append(f"{m}\n{v}")  # two-line label (metric on top, vote below)

    x = np.arange(len(pairs))
    set_nature_rc()

    # --- key change: figure width grows with number of bars ---
    width_inch = max(3.35, 0.7 * len(pairs) + 0.6)  # keep Nature-like width when few bars
    plt.figure(figsize=(width_inch, 2.4))

    plt.bar(x, accs)
    # center-aligned, a bit more padding to avoid overlap with axis
    plt.xticks(x, labels, ha="center")
    plt.tick_params(axis="x", pad=3)

    plt.ylabel("Validation accuracy")
    plt.title("Accuracy by metric × vote")
    plt.tight_layout()  # ensure labels are not clipped
    plt.savefig(out_path)
    plt.close()



# ---------------------------
# Figure 3 & 4: time and throughput vs batch_size (for best k/metric/vote)
# ---------------------------
def _best_triplet(records: List[Dict[str, Any]]) -> Tuple[int, str, str]:
    best_all = pick_best(records)
    return best_all["k"], best_all["metric"], best_all["vote"]


def fig_time_vs_batch_for_best(records: List[Dict[str, Any]], out_path: str):
    k, m, v = _best_triplet(records)
    rows = sorted([r for r in records if r["k"] == k and r["metric"] == m and r["vote"] == v],
                  key=lambda r: r["batch_size"])
    bsz = [r["batch_size"] for r in rows]
    times = [r["val_time"] for r in rows]

    set_nature_rc()
    fig, ax = plt.subplots(figsize=(3.35, 2.4))  # 保持统一尺寸与风格
    ax.plot(bsz, times, marker="o")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Validation time (s)")

    # ——关键改动：两行标题 + 适度上边距——
    title_main = "Time vs batch size"
    title_sub  = f"(k={k}, {m}, {v})"
    ax.set_title(f"{title_main}\n{title_sub}", pad=4)

    ax.set_xticks(bsz)

    # 为标题预留空间，避免被裁剪
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_path)
    plt.close(fig)



def fig_throughput_vs_batch(records: List[Dict[str, Any]], out_path: str, n_val: int = 6000):
    k, m, v = _best_triplet(records)
    rows = sorted([r for r in records if r["k"] == k and r["metric"] == m and r["vote"] == v],
                  key=lambda r: r["batch_size"])
    bsz = [r["batch_size"] for r in rows]
    thr = [n_val / r["val_time"] if r["val_time"] > 0 else np.nan for r in rows]

    set_nature_rc()
    plt.figure(figsize=(3.35, 2.4))
    plt.plot(bsz, thr, marker="o")
    plt.xlabel("Batch size")
    plt.ylabel("Throughput (images/s)")
    plt.title(f"Throughput vs batch size (k={k}, {m}, {v})")
    plt.xticks(bsz)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 5: Pareto scatter (val_acc vs val_time)
# ---------------------------
def fig_pareto(records: List[Dict[str, Any]], out_path: str):
    # Color by metric (default cycle), marker by vote, size by batch_size
    votes = sorted({r["vote"] for r in records})
    vote_marker = {v: ("o" if i == 0 else "^") for i, v in enumerate(votes)}

    set_nature_rc()
    plt.figure(figsize=(3.6, 2.6))
    for v in votes:
        sub = [r for r in records if r["vote"] == v]
        sizes = [(r["batch_size"] / max(1, max(rr["batch_size"] for rr in records))) * 300 for r in sub]
        plt.scatter([r["val_time"] for r in sub],
                    [r["val_acc"] for r in sub],
                    s=sizes, marker=vote_marker[v], alpha=0.75, label=f"vote={v}")
    plt.xlabel("Validation time (s)")
    plt.ylabel("Validation accuracy")
    plt.title("Pareto frontier of accuracy–time")
    plt.legend(frameon=False, title=None)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 6: Confusion matrix (best config)
# ---------------------------
def fig_confusion_matrix(y_pred: torch.Tensor, y_true: torch.Tensor, out_path: str, normalize: bool = True):
    cm = torch.zeros(10, 10, dtype=torch.float32)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t, p] += 1.0
    if normalize:
        cm = cm / (cm.sum(dim=1, keepdim=True) + 1e-12)

    set_nature_rc()
    plt.figure(figsize=(3.1, 3.1))
    plt.imshow(cm.numpy(), aspect="equal", interpolation="nearest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 7: Per-class accuracy bars
# ---------------------------
def fig_per_class_acc(y_pred: torch.Tensor, y_true: torch.Tensor, out_path: str):
    cls = 10
    acc = []
    for c in range(cls):
        mask = (y_true == c)
        num = int(mask.sum().item())
        if num == 0:
            acc.append(np.nan)
        else:
            acc.append(float((y_pred[mask] == y_true[mask]).float().mean().item()))
    set_nature_rc()
    plt.figure(figsize=(3.35, 2.4))
    plt.bar(list(range(cls)), acc)
    plt.xlabel("Class")
    plt.ylabel("Test accuracy")
    plt.title("Per-class accuracy")
    plt.xticks(range(10))
    plt.ylim(0, 1.0)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 8: Misclassified image grid (pred→true)
# ---------------------------
def fig_misclassified_grid(pt_path: str, out_path: str, max_examples: int = 36):
    y_pred, y_true = load_test_pred(pt_path)

    # Load MNIST test images (order should match)
    if not TORCHVISION_OK:
        raise RuntimeError("torchvision not available to render image grid.")
    tfm = transforms.ToTensor()
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    wrong_idx = (y_pred != y_true).nonzero(as_tuple=False).view(-1).tolist()
    sel = wrong_idx[:max_examples]
    n = len(sel)
    if n == 0:
        # Still output an empty canvas with a note
        set_nature_rc()
        plt.figure(figsize=(3.2, 2.0))
        plt.text(0.5, 0.5, "No misclassification found.", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path)
        plt.close()
        return

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    set_nature_rc()
    plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for i, idx in enumerate(sel):
        img, _ = test_set[idx]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img.squeeze(0).numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"{int(y_pred[idx])} → {int(y_true[idx])}", pad=2.0)
    plt.tight_layout(w_pad=0.5, h_pad=0.8)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 9: Top-1 distance/similarity distributions
# ---------------------------
def fig_top1_distance_hist(csv_path: str, pt_path: str, out_path: str, sample: int = 2000):
    # Recompute top-1 distance for best config on a subset of test images.
    # Uses full MNIST train split for simplicity.
    if not TORCHVISION_OK:
        raise RuntimeError("torchvision required for distance histogram.")

    # Best config from CSV
    records = read_val_grid(csv_path)
    best = pick_best(records)
    metric = best["metric"]
    k = best["k"]

    # Load data
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    Xtr = torch.stack([train[i][0].view(-1) for i in range(len(train))], dim=0).float()
    Xte = torch.stack([test[i][0].view(-1) for i in range(len(test))],  dim=0).float()

    # Load predictions to split correct/incorrect
    y_pred, y_true = load_test_pred(pt_path)
    N = min(sample, Xte.shape[0])
    idx = torch.arange(N)

    # Compute pairwise distances blockwise with torch.cdist / cosine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = Xtr.to(device)
    Xte = Xte[idx].to(device)

    if metric in ("l2", "l1"):
        p = 2.0 if metric == "l2" else 1.0
        dists = torch.cdist(Xte, Xtr, p=p)  # [N, Ntr]
        vals, _ = torch.topk(dists, k=1, largest=False, dim=1)
        top1 = vals.view(-1).cpu()
        # smaller distance is better
        correct = top1[(y_pred[idx] == y_true[idx])]
        wrong   = top1[(y_pred[idx] != y_true[idx])]
        xlabel = "Top-1 distance"
    else:
        # cosine: similarity = (x/||x||)·(y/||y||); use distance = 1 - sim
        Xtr_n = torch.nn.functional.normalize(Xtr.float(), p=2, dim=1)
        Xte_n = torch.nn.functional.normalize(Xte.float(), p=2, dim=1)
        sim = Xte_n @ Xtr_n.t()  # [N, Ntr]
        dist = 1.0 - sim
        vals, _ = torch.topk(dist, k=1, largest=False, dim=1)
        top1 = vals.view(-1).cpu()
        correct = top1[(y_pred[idx] == y_true[idx])]
        wrong   = top1[(y_pred[idx] != y_true[idx])]
        xlabel = "Top-1 (1 − cosine similarity)"

    set_nature_rc()
    plt.figure(figsize=(3.4, 2.4))
    bins = 30
    plt.hist(correct.numpy(), bins=bins, alpha=0.6, label="Correct")
    plt.hist(wrong.numpy(),   bins=bins, alpha=0.6, label="Wrong")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title("Top-1 neighbor distance distribution")
    plt.legend(frameon=False)
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Figure 10: Naive broadcast memory curve
# ---------------------------
def fig_naive_memory_curve(out_path: str, ntr: int = 54000, d: int = 784, dtype_bytes: int = 4):
    # mem ≈ Nte * Ntr * D * dtype_bytes (bytes)
    ntes = np.linspace(0, 10000, 50)
    mem_gib = ntes * ntr * d * dtype_bytes / (1024**3)

    set_nature_rc()
    plt.figure(figsize=(3.35, 2.4))
    plt.plot(ntes, mem_gib, marker="o")
    plt.xlabel("N_test")
    plt.ylabel("Theoretical memory (GiB)")
    plt.title("Naive broadcast memory growth")
    plt.savefig(out_path)
    plt.close()


# ---------------------------
# Orchestrator
# ---------------------------
def generate_all(fig_dir: str = "./runs/figs"):
    ensure_dir(fig_dir)

    val_csv = "./runs/val_grid.csv"
    test_pt = "./runs/test_pred.pt"

    records = read_val_grid(val_csv)
    y_pred, y_test = load_test_pred(test_pt)

    # 1
    fig_acc_vs_k_by_metric(records, os.path.join(fig_dir, "fig01_acc_vs_k_by_metric.png"))
    # 2
    fig_metric_vote_bar(records,  os.path.join(fig_dir, "fig02_metric_vote_bar.png"))
    # 3
    fig_time_vs_batch_for_best(records, os.path.join(fig_dir, "fig03_time_vs_batch_for_best.png"))
    # 4
    fig_throughput_vs_batch(records, os.path.join(fig_dir, "fig04_throughput_vs_batch.png"), n_val=6000)
    # 5
    fig_pareto(records, os.path.join(fig_dir, "fig05_pareto_valacc_valtime.png"))
    # 6
    fig_confusion_matrix(y_pred, y_test, os.path.join(fig_dir, "fig06_confusion_matrix.png"))
    # 7
    fig_per_class_acc(y_pred, y_test, os.path.join(fig_dir, "fig07_per_class_accuracy.png"))
    # 8
    fig_misclassified_grid(test_pt, os.path.join(fig_dir, "fig08_misclassified_grid.png"), max_examples=36)
    # 9
    fig_top1_distance_hist(val_csv, test_pt, os.path.join(fig_dir, "fig09_top1_distance_hist.png"), sample=2000)
    # 10
    fig_naive_memory_curve(os.path.join(fig_dir, "fig10_naive_broadcast_memory.png"))

    print(f"[FIG] Saved figures to: {fig_dir}")


if __name__ == "__main__":
    set_nature_rc()
    generate_all()
