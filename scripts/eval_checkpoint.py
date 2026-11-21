"""Evaluate a saved BinSight checkpoint and write summary + qualitative figure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import BIN_CLASSES, BinSightDataset  # noqa: E402
from models import build_model, count_parameters  # noqa: E402


def evaluate(model, loader):
    ce_bin = nn.CrossEntropyLoss()
    ce_mat = nn.CrossEntropyLoss()
    preds, ys = [], []
    mats_ok = n = loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, yb, ym in loader:
            pb, pm = model(x)
            loss += (ce_bin(pb, yb) + 0.5 * ce_mat(pm, ym)).item() * x.size(0)
            pred = pb.argmax(1)
            preds.append(pred)
            ys.append(yb)
            mats_ok += (pm.argmax(1) == ym).sum().item()
            n += x.size(0)
    pred = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    f1s = []
    for c in range(3):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return {
        "n": int(n),
        "loss": loss / n,
        "bin_acc": float((pred == y).mean()),
        "macro_f1": float(np.mean(f1s)),
        "mat_acc": mats_ok / n,
        "pred": pred,
        "y": y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/hierarchical_best.pt")
    ap.add_argument("--model", default="hierarchical")
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(ROOT / args.ckpt, map_location=device, weights_only=True)
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = count_parameters(model)
    print("epoch", ckpt.get("epoch"), "params", n_params)

    val_loader = DataLoader(BinSightDataset(ROOT / "data/processed", "val"), batch_size=64)
    test_loader = DataLoader(BinSightDataset(ROOT / "data/processed", "test"), batch_size=64)
    val = evaluate(model, val_loader)
    test = evaluate(model, test_loader)
    print("VAL", {k: val[k] for k in val if k not in ("pred", "y")})
    print("TEST", {k: test[k] for k in test if k not in ("pred", "y")})

    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv")
    tm = manifest[manifest["split"] == "test"].reset_index(drop=True)
    tm["pred"] = test["pred"]
    tm["y"] = test["y"]
    per = tm.assign(ok=lambda d: d.pred == d.y).groupby("source").ok.mean().to_dict()
    print("per_source", per)

    out = ROOT / "results"
    summary = {
        "model": args.model,
        "params": n_params,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "val_bin_acc": val["bin_acc"],
        "val_bin_macro_f1": val["macro_f1"],
        "val_mat_acc": val["mat_acc"],
        "test_bin_acc": test["bin_acc"],
        "test_bin_macro_f1": test["macro_f1"],
        "test_mat_acc": test["mat_acc"],
        "test_per_source_bin_acc": {k: float(v) for k, v in per.items()},
    }
    (out / f"{args.model}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    cidx = np.where(test["pred"] == test["y"])[0][:8]
    widx = np.where(test["pred"] != test["y"])[0][:8]
    for row, idxs in ((0, cidx), (1, widx)):
        for j, i in enumerate(idxs):
            axes[row, j].imshow(plt.imread(ROOT / "data/processed" / tm.at[int(i), "path"]))
            axes[row, j].axis("off")
            p = BIN_CLASSES[int(test["pred"][i])][0]
            t = BIN_CLASSES[int(test["y"][i])][0]
            axes[row, j].set_title(f"P:{p}\nT:{t}", fontsize=7)
    fig.suptitle(f"{args.model}: qualitative bin predictions")
    fig.tight_layout()
    fig.savefig(out / f"{args.model}_qualitative.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
