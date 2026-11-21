"""PyTorch Dataset over data/processed/manifest.csv (built by scripts/clean_data.py)."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

BIN_CLASSES = ["blue", "green", "garbage"]
MATERIAL_CLASSES = ["cardboard", "paper", "plastic", "metal", "glass", "organic", "other"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class BinSightDataset(Dataset):
    """Returns (image, bin_index, material_index)."""

    def __init__(self, root="data/processed", split="train", transform=None):
        self.root = Path(root)
        df = pd.read_csv(self.root / "manifest.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)
        self.bin_idx = self.df["bin"].map(BIN_CLASSES.index).to_numpy()
        self.mat_idx = self.df["material"].map(MATERIAL_CLASSES.index).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.root / self.df.at[i, "path"]).convert("RGB")
        return self.transform(img), int(self.bin_idx[i]), int(self.mat_idx[i])

    def class_weights(self, task="bin"):
        """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
        idx, n = (self.bin_idx, len(BIN_CLASSES)) if task == "bin" else (self.mat_idx, len(MATERIAL_CLASSES))
        counts = torch.bincount(torch.as_tensor(idx), minlength=n).float()
        w = counts.sum() / counts.clamp(min=1)
        return w / w.mean()
