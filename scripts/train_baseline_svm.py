"""Classical HOG + HSV + PCA + RBF-SVM baseline for BinSight.

Usage:
    python scripts/train_baseline_svm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import BIN_CLASSES  # noqa: E402

OUT = ROOT / "results"
SIZE = 128


def colour_hist(img: Image.Image) -> np.ndarray:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    hist = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=8, range=(0, 256), density=True)
        hist.append(h)
    return np.concatenate(hist)


def featurize(paths):
    feats = []
    for p in tqdm(paths, desc="HOG+HSV"):
        img = Image.open(p).convert("RGB").resize((SIZE, SIZE))
        gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        h = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        feats.append(np.concatenate([h, colour_hist(img)]))
    return np.asarray(feats, dtype=np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/manifest.csv")
    df["abs"] = df["path"].map(lambda p: str(ROOT / "data/processed" / p))
    df["y"] = df["bin"].map(BIN_CLASSES.index)

    splits = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        X = featurize(sub["abs"].tolist())
        y = sub["y"].to_numpy()
        splits[split] = (X, y, sub.reset_index(drop=True))

    # small grid on val
    best, best_score = None, -1
    for C in (0.1, 1, 10):
        for gamma in ("scale", 1e-3):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=256, random_state=42)),
                    ("svm", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced")),
                ]
            )
            pipe.fit(splits["train"][0], splits["train"][1])
            pred = pipe.predict(splits["val"][0])
            score = f1_score(splits["val"][1], pred, average="macro")
            print(f"C={C} gamma={gamma} val_macro_f1={score:.3f}")
            if score > best_score:
                best_score, best = score, (C, gamma, pipe)

    C, gamma, pipe = best
    # refit on train+val for a slightly stronger test estimate of the baseline
    X_tr = np.concatenate([splits["train"][0], splits["val"][0]])
    y_tr = np.concatenate([splits["train"][1], splits["val"][1]])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(splits["test"][0])
    y_te = splits["test"][1]
    test_df = splits["test"][2].copy()
    test_df["pred"] = pred

    acc = accuracy_score(y_te, pred)
    macro = f1_score(y_te, pred, average="macro")
    cm = confusion_matrix(y_te, pred)
    per_source = (
        test_df.assign(correct=lambda d: d["pred"] == d["y"])
        .groupby("source")["correct"]
        .mean()
        .to_dict()
    )

    summary = {
        "model": "hog_hsv_svm",
        "params": "HOG 1764-d + HSV 512-d -> PCA 256 + RBF SVM",
        "C": C,
        "gamma": gamma,
        "val_bin_macro_f1": float(best_score),
        "test_bin_acc": float(acc),
        "test_bin_macro_f1": float(macro),
        "test_per_source_bin_acc": {k: float(v) for k, v in per_source.items()},
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_te, pred, target_names=BIN_CLASSES),
    }
    (OUT / "svm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary["report"])
    print(json.dumps({k: summary[k] for k in summary if k != "report"}, indent=2))

    # qualitative figure
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    correct = np.where(pred == y_te)[0][:6]
    wrong = np.where(pred != y_te)[0][:6]
    for row, idxs in enumerate((correct, wrong)):
        for j, i in enumerate(idxs):
            axes[row, j].imshow(plt.imread(test_df.at[int(i), "abs"]))
            axes[row, j].set_axis_off()
            axes[row, j].set_title(
                f"P:{BIN_CLASSES[pred[i]][0]} T:{BIN_CLASSES[y_te[i]][0]}", fontsize=7
            )
    fig.suptitle("SVM baseline qualitative (P=pred, T=true)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "svm_qualitative.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), BIN_CLASSES)
    ax.set_yticks(range(3), BIN_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "svm_confusion.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
