"""Write dataset summary figures into results/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data/processed/manifest.csv")

print("TOTAL", len(df))
print(pd.crosstab(df["split"], df["bin"], margins=True))
print(pd.crosstab(df["split"], df["material"], margins=True))
print(pd.crosstab(df["source"], df["bin"], margins=True))

fig, ax = plt.subplots(figsize=(5.2, 3))
ct = pd.crosstab(df["split"], df["bin"])[["blue", "green", "garbage"]]
ct.loc[["train", "val", "test"]].plot(kind="bar", ax=ax, color=["#2E86C1", "#27AE60", "#7F8C8D"])
ax.set_ylabel("count")
ax.set_title("Cleaned dataset: bin counts by split")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(OUT / "data_split_bins.png", dpi=160)
plt.close()

fig, ax = plt.subplots(figsize=(5.2, 3))
ct = pd.crosstab(df["source"], df["bin"])[["blue", "green", "garbage"]]
ct.plot(kind="bar", ax=ax, color=["#2E86C1", "#27AE60", "#7F8C8D"])
ax.set_ylabel("count")
ax.set_title("Source confound: Green mostly from Kaggle")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(OUT / "data_source_bins.png", dpi=160)
plt.close()

samples = []
for b in ("blue", "green", "garbage"):
    sub = df[df["bin"] == b]
    samples.append(sub.sample(min(3, len(sub)), random_state=0))
grid = pd.concat(samples).head(8)

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for ax, (_, r) in zip(axes.flat, grid.iterrows()):
    ax.imshow(Image.open(ROOT / "data/processed" / r.path))
    ax.set_axis_off()
    ax.set_title(f"{r.bin}/{r.material}\n{r.source}", fontsize=7)
fig.suptitle("Cleaned training examples")
fig.tight_layout()
fig.savefig(OUT / "data_examples.png", dpi=160)
plt.close()
print("wrote figures to", OUT)
