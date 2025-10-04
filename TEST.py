import argparse, os, time
import torch

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["val","test","all"], default="val")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--metric", choices=["l2","l1"], default="l2")
    p.add_argument("--weighted", type=int, default=0)  # 0/1
    p.add_argument("--batch", type=int, default=1024)  # 分块大小
    p.add_argument("--outdir", type=str, default="outputs")
    return p.parse_args()

args = get_args()
device = torch.device("cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
os.makedirs(args.outdir, exist_ok=True)
print(f"=> device: {device}, k={args.k}, metric={args.metric}, weighted={bool(args.weighted)}, batch={args.batch}")

# 你的数据加载、kNN 实现...
# 关键：在批量预测时做“分块”
def predict_in_blocks(Xq, Xtr, Ytr, k=5, metric="l2", weighted=False, block=1024, device="cpu"):
    preds = []
    for s in range(0, Xq.size(0), block):
        e = min(s+block, Xq.size(0))
        d = torch.cdist(Xq[s:e].to(device), Xtr.to(device), p=2 if metric=="l2" else 1)  # [b,N]
        topd, topi = torch.topk(d, k, largest=False, dim=1)
        labs = Ytr[topi]  # [b,k]
        # 简易多数投票（可替换为你的 vote 函数）
        for i in range(labs.size(0)):
            vals, counts = torch.unique(labs[i].cpu(), return_counts=True)
            pred = vals[torch.argmax(counts)].item()
            preds.append(pred)
        torch.cuda.synchronize() if device=="cuda" else None
    return torch.tensor(preds)
