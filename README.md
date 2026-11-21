# BinSight (APS360)

Deep learning project that predicts a Toronto household waste item’s **bin** (Blue / Green / Garbage) and **material** from one photo. Low-confidence predictions abstain with “check Waste Wizard”.

## What is new in this repo (responding to proposal feedback)

- **Primary model:** a from-scratch hierarchical CNN. A material head produces a soft prior through a learnable material→bin rule matrix; a correction head adds residual logits for municipal exceptions (e.g. soiled tissues).
- **Strong DL baseline:** ImageNet-pretrained ResNet-18 with two heads (no longer treated as the main contribution).
- **Classical baseline:** HOG + HSV colour histogram + PCA + RBF-SVM.
- **Data work:** TrashNet + TACO crops + cleaned Kaggle organics, remapped with `data/label_map.csv`, pHash de-duplication, grouped 70/15/15 split.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/clean_data.py
python scripts/train_baseline_svm.py
python scripts/train_models.py --model hierarchical --epochs 10
python scripts/train_models.py --model resnet18 --epochs 8
```

Processed images and raw downloads are **not** committed. See `data/processed/report.md` after cleaning for counts.

## Layout

```
data/label_map.csv          # municipal remapping
scripts/download_data.py
scripts/clean_data.py
scripts/train_baseline_svm.py
scripts/train_models.py
src/dataset.py
src/models.py
proposal/                   # submitted proposal sources
progress/                   # progress report sources
results/                    # metrics and figures
```

## Author

Haoyue Feng · 1013983076 · haoyue.feng@mail.utoronto.ca
