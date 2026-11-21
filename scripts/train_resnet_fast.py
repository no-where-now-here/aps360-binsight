"""Fast ResNet-18 baseline: freeze backbone for a few epochs, then lightly fine-tune."""
from __future__ import annotations

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
from dataset import BIN_CLASSES, BinSightDataset  # noqa: E402
from models import ResNet18Baseline, count_parameters  # noqa: E402

OUT = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, ce_bin, ce_mat, lam=0.5):
    model.eval()
    tot_loss = tot_bin = tot_mat = n = 0
    preds, ys = [], []
    for x, yb, ym in loader:
        x, yb, ym = x.to(DEVICE), yb.to(DEVICE), ym.to(DEVICE)
        pb, pm = model(x)
        loss = ce_bin(pb, yb) + lam * ce_mat(pm, ym)
        bs = x.size(0)
        tot_loss += loss.item() * bs
        tot_bin += accuracy(pb, yb) * bs
        tot_mat += accuracy(pm, ym) * bs
        n += bs
        preds.append(pb.argmax(1).cpu())
        ys.append(yb.cpu())
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
        "loss": tot_loss / n,
        "bin_acc": tot_bin / n,
        "mat_acc": tot_mat / n,
        "bin_macro_f1": float(np.mean(f1s)),
        "pred": pred,
        "y": y,
    }


def set_backbone_trainable(model, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train_ds = BinSightDataset(ROOT / "data/processed", "train")
    val_ds = BinSightDataset(ROOT / "data/processed", "val")
    test_ds = BinSightDataset(ROOT / "data/processed", "test")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = ResNet18Baseline().to(DEVICE)
    ce_bin = nn.CrossEntropyLoss(weight=train_ds.class_weights("bin").to(DEVICE))
    ce_mat = nn.CrossEntropyLoss(weight=train_ds.class_weights("material").to(DEVICE))
    history = []
    best = {"bin_macro_f1": -1.0}
    best_path = OUT / "resnet18_best.pt"

    stages = [
        ("freeze", 4, 1e-3, False),
        ("finetune", 3, 1e-4, True),
    ]
    global_epoch = 0
    for stage_name, epochs, lr, train_backbone in stages:
        set_backbone_trainable(model, train_backbone)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        print(f"[{stage_name}] trainable params={sum(p.numel() for p in params):,} lr={lr}")
        for _ in range(epochs):
            global_epoch += 1
            model.train()
            run_loss = run_bin = run_n = 0
            for x, yb, ym in tqdm(train_loader, desc=f"resnet {stage_name} ep{global_epoch}", leave=False):
                x, yb, ym = x.to(DEVICE), yb.to(DEVICE), ym.to(DEVICE)
                opt.zero_grad()
                pb, pm = model(x)
                loss = ce_bin(pb, yb) + 0.5 * ce_mat(pm, ym)
                loss.backward()
                opt.step()
                bs = x.size(0)
                run_loss += loss.item() * bs
                run_bin += accuracy(pb.detach(), yb) * bs
                run_n += bs
            val = evaluate(model, val_loader, ce_bin, ce_mat)
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "train_loss": run_loss / run_n,
                "train_bin_acc": run_bin / run_n,
                "val_loss": val["loss"],
                "val_bin_acc": val["bin_acc"],
                "val_mat_acc": val["mat_acc"],
                "val_bin_macro_f1": val["bin_macro_f1"],
            }
            history.append(row)
            print(
                f"[resnet18] ep{global_epoch:02d} ({stage_name}) "
                f"train={row['train_bin_acc']:.3f} val={row['val_bin_acc']:.3f} f1={row['val_bin_macro_f1']:.3f}"
            )
            if val["bin_macro_f1"] > best["bin_macro_f1"]:
                best = {**val, "epoch": global_epoch}
                torch.save({"model": model.state_dict(), "epoch": global_epoch, "name": "resnet18"}, best_path)

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, ce_bin, ce_mat)
    manifest = pd.read_csv(ROOT / "data/processed/manifest.csv")
    tm = manifest[manifest["split"] == "test"].reset_index(drop=True)
    tm["pred"] = test["pred"]
    tm["y"] = test["y"]
    per = tm.assign(ok=lambda d: d.pred == d.y).groupby("source").ok.mean().to_dict()

    hist = pd.DataFrame(history)
    hist.to_csv(OUT / "resnet18_history.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title("resnet18 loss")
    ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_bin_acc"], label="train")
    ax[1].plot(hist["epoch"], hist["val_bin_acc"], label="val")
    ax[1].set_title("resnet18 bin acc")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "resnet18_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3.6))
    cidx = np.where(test["pred"] == test["y"])[0][:8]
    widx = np.where(test["pred"] != test["y"])[0][:8]
    for row, idxs in ((0, cidx), (1, widx)):
        for j, i in enumerate(idxs):
            axes[row, j].imshow(plt.imread(ROOT / "data/processed" / tm.at[int(i), "path"]))
            axes[row, j].axis("off")
            axes[row, j].set_title(
                f"P:{BIN_CLASSES[int(test['pred'][i])][0]}\nT:{BIN_CLASSES[int(test['y'][i])][0]}",
                fontsize=7,
            )
    fig.suptitle("resnet18 qualitative")
    fig.tight_layout()
    fig.savefig(OUT / "resnet18_qualitative.png", dpi=160)
    plt.close(fig)

    # also rebuild hierarchical curves if history missing: skip
    summary = {
        "model": "resnet18",
        "params": count_parameters(model),
        "best_epoch": int(ckpt["epoch"]),
        "val_bin_acc": float(best["bin_acc"]),
        "val_bin_macro_f1": float(best["bin_macro_f1"]),
        "val_mat_acc": float(best["mat_acc"]),
        "test_bin_acc": float(test["bin_acc"]),
        "test_bin_macro_f1": float(test["bin_macro_f1"]),
        "test_mat_acc": float(test["mat_acc"]),
        "test_per_source_bin_acc": {k: float(v) for k, v in per.items()},
    }
    (OUT / "resnet18_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
