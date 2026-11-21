"""Build comparison figures for the progress report from saved JSON summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def load(name):
    p = OUT / f"{name}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    svm = load("svm")
    hier = load("hierarchical")
    res = load("resnet18")
    rows = []
    if svm:
        rows.append(("HOG+SVM", svm["test_bin_acc"], svm["test_bin_macro_f1"], None))
    if hier:
        rows.append(("Hierarchical CNN\n(primary)", hier["test_bin_acc"], hier["test_bin_macro_f1"], hier["params"]))
    if res:
        rows.append(("ResNet-18\n(strong baseline)", res["test_bin_acc"], res["test_bin_macro_f1"], res["params"]))

    names = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    f1 = [r[2] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    w = 0.35
    ax.bar(x - w / 2, acc, w, label="test bin acc", color="#2E86C1")
    ax.bar(x + w / 2, f1, w, label="test macro-F1", color="#E67E22")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("BinSight progress: primary vs baselines")
    ax.legend(fontsize=8)
    for i, r in enumerate(rows):
        if r[3] is not None:
            ax.text(i, max(r[1], r[2]) + 0.03, f"{r[3]/1e6:.2f}M params", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close(fig)

    # per-source comparison if available
    sources = ["trashnet", "kaggle", "taco"]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    width = 0.25
    plotted = False
    for i, (label, obj, color) in enumerate(
        (("SVM", svm, "#5D6D7E"), ("Hier. CNN", hier, "#27AE60"), ("ResNet-18", res, "#8E44AD"))
    ):
        if not obj or "test_per_source_bin_acc" not in obj:
            continue
        vals = [obj["test_per_source_bin_acc"].get(s, 0) for s in sources]
        ax.bar(np.arange(len(sources)) + (i - 1) * width, vals, width, label=label, color=color)
        plotted = True
    if plotted:
        ax.set_xticks(np.arange(len(sources)), sources)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("bin accuracy")
        ax.set_title("Per-source test accuracy (style confound check)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "per_source_accuracy.png", dpi=160)
    plt.close(fig)
    print("wrote comparison figures")


if __name__ == "__main__":
    main()
