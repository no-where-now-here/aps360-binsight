"""Train BinSight primary CNN and ResNet-18 baseline; write metrics and curves.

Usage (from project root):
    python scripts/train_models.py --model hierarchical --epochs 12
    python scripts/train_models.py --model resnet18 --epochs 8
    python scripts/train_models.py --all
"""
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import BIN_CLASSES, MATERIAL_CLASSES, BinSightDataset  # noqa: E402
from models import build_model, count_parameters  # noqa: E402

OUT = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, ce_bin, ce_mat, lam):
    model.eval()
    tot_loss = tot_bin = tot_mat = n = 0
    all_pred, all_y, all_src = [], [], []
    for x, yb, ym in loader:
        x, yb, ym = x.to(DEVICE), yb.to(DEVICE), ym.to(DEVICE)
        pb, pm = model(x)
        loss = ce_bin(pb, yb) + lam * ce_mat(pm, ym)
        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_bin += accuracy(pb, yb) * bs
        tot_mat += accuracy(pm, ym) * bs
        n += bs
        all_pred.append(pb.argmax(1).cpu())
        all_y.append(yb.cpu())
    pred = torch.cat(all_pred).numpy()
    y = torch.cat(all_y).numpy()
    # macro-F1 for bins
    f1s = []
    for c in range(len(BIN_CLASSES)):
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return {
        "loss": tot_loss / n,
        "bin_acc": tot_bin / n,
        "mat_acc": tot_mat / n,
        "bin_macro_f1": float(np.mean(f1s)),
        "pred": pred,
        "y": y,
    }


def train_one(name, epochs, batch_size, lam, lr_head, lr_backbone):
    OUT.mkdir(parents=True, exist_ok=True)
    train_ds = BinSightDataset(ROOT / "data/processed", "train")
    val_ds = BinSightDataset(ROOT / "data/processed", "val")
    test_ds = BinSightDataset(ROOT / "data/processed", "test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(name).to(DEVICE)
    n_params = count_parameters(model)
    print(f"[{name}] params={n_params:,} device={DEVICE} train={len(train_ds)} val={len(val_ds)}")

    ce_bin = nn.CrossEntropyLoss(weight=train_ds.class_weights("bin").to(DEVICE))
    ce_mat = nn.CrossEntropyLoss(weight=train_ds.class_weights("material").to(DEVICE))

    if name == "resnet18":
        # stage-ish: lower lr on backbone
        opt = torch.optim.Adam(
            [
                {"params": model.backbone.parameters(), "lr": lr_backbone},
                {"params": list(model.shared.parameters()) + list(model.bin_head.parameters()) + list(model.material_head.parameters()), "lr": lr_head},
            ]
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr_head)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    best = {"bin_macro_f1": -1}
    best_path = OUT / f"{name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = run_bin = run_n = 0
        for x, yb, ym in tqdm(train_loader, desc=f"{name} ep{epoch}", leave=False):
            x, yb, ym = x.to(DEVICE), yb.to(DEVICE), ym.to(DEVICE)
            opt.zero_grad()
            pb, pm = model(x)
            loss = ce_bin(pb, yb) + lam * ce_mat(pm, ym)
            loss.backward()
            opt.step()
            bs = x.size(0)
            run_loss += loss.item() * bs
            run_bin += accuracy(pb.detach(), yb) * bs
            run_n += bs
        sched.step()
        val = evaluate(model, val_loader, ce_bin, ce_mat, lam)
        row = {
            "epoch": epoch,
            "train_loss": run_loss / run_n,
            "train_bin_acc": run_bin / run_n,
            "val_loss": val["loss"],
            "val_bin_acc": val["bin_acc"],
            "val_mat_acc": val["mat_acc"],
            "val_bin_macro_f1": val["bin_macro_f1"],
        }
        history.append(row)
        print(
            f"[{name}] ep{epoch:02d} train_acc={row['train_bin_acc']:.3f} "
            f"val_acc={row['val_bin_acc']:.3f} val_f1={row['val_bin_macro_f1']:.3f}"
        )
        if val["bin_macro_f1"] > best["bin_macro_f1"]:
            best = {**val, "epoch": epoch, "bin_macro_f1": val["bin_macro_f1"]}
            torch.save({"model": model.state_dict(), "epoch": epoch, "name": name}, best_path)

    # reload best and test
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, ce_bin, ce_mat, lam)

    # per-source accuracy on test
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv")
    test_m = manifest[manifest["split"] == "test"].reset_index(drop=True)
    test_m["pred"] = test["pred"]
    test_m["y"] = test["y"]
    per_source = (
        test_m.assign(correct=lambda d: d["pred"] == d["y"])
        .groupby("source")["correct"]
        .mean()
        .to_dict()
    )

    # qualitative grid: 8 correct + 8 wrong
    correct_idx = np.where(test["pred"] == test["y"])[0][:8]
    wrong_idx = np.where(test["pred"] != test["y"])[0][:8]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    for row, idxs, title in ((0, correct_idx, "correct"), (1, wrong_idx, "wrong")):
        for j, i in enumerate(idxs):
            path = ROOT / "data/processed" / test_m.at[int(i), "path"]
            img = plt.imread(path)
            axes[row, j].imshow(img)
            axes[row, j].set_axis_off()
            pred = BIN_CLASSES[int(test["pred"][i])]
            true = BIN_CLASSES[int(test["y"][i])]
            axes[row, j].set_title(f"P:{pred[0]}\nT:{true[0]}", fontsize=7)
        if len(idxs) == 0:
            axes[row, 0].set_title(f"no {title}")
    fig.suptitle(f"{name}: qualitative bin predictions (P=pred, T=true)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_qualitative.png", dpi=160)
    plt.close(fig)

    # learning curves
    hist = pd.DataFrame(history)
    hist.to_csv(OUT / f"{name}_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title(f"{name} loss")
    ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_bin_acc"], label="train bin acc")
    ax[1].plot(hist["epoch"], hist["val_bin_acc"], label="val bin acc")
    ax[1].set_title(f"{name} bin accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "model": name,
        "params": n_params,
        "best_epoch": int(ckpt["epoch"]),
        "val_bin_acc": float(best["bin_acc"]),
        "val_bin_macro_f1": float(best["bin_macro_f1"]),
        "val_mat_acc": float(best["mat_acc"]),
        "test_bin_acc": float(test["bin_acc"]),
        "test_bin_macro_f1": float(test["bin_macro_f1"]),
        "test_mat_acc": float(test["mat_acc"]),
        "test_per_source_bin_acc": {k: float(v) for k, v in per_source.items()},
    }
    (OUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hierarchical", "resnet18"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    args = ap.parse_args()
    models = ["hierarchical", "resnet18"] if args.all else [args.model]
    if models == [None]:
        ap.error("pass --model or --all")
    defaults = {"hierarchical": 12, "resnet18": 8}
    summaries = []
    for name in models:
        epochs = args.epochs or defaults[name]
        summaries.append(train_one(name, epochs, args.batch_size, args.lam, args.lr_head, args.lr_backbone))
    pd.DataFrame(summaries).to_csv(OUT / "nn_comparison.csv", index=False)


if __name__ == "__main__":
    main()
