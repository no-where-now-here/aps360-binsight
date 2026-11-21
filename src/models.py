"""BinSight models: hierarchical from-scratch CNN (primary) and ResNet-18 baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dataset import BIN_CLASSES, MATERIAL_CLASSES

# Soft Toronto prior: rows = material, cols = blue/green/garbage.
# Strong diagonals for clear cases; tissues/paper can also go green; other leans garbage.
DEFAULT_RULE = torch.tensor(
    [
        [4.0, 0.0, 1.0],  # cardboard -> blue
        [3.0, 2.0, 1.0],  # paper -> blue, but green for soiled tissues
        [4.0, 0.0, 1.0],  # plastic -> blue
        [4.0, 0.0, 1.0],  # metal -> blue
        [4.0, 0.0, 1.0],  # glass -> blue
        [0.0, 4.0, 1.0],  # organic -> green
        [0.5, 0.5, 4.0],  # other -> garbage
    ],
    dtype=torch.float32,
)


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class HierarchicalBinCNN(nn.Module):
    """From-scratch CNN: material head + rule prior + residual bin correction.

    This is the *primary* model. The material softmax is multiplied by a (learnable)
    material→bin rule matrix to form a soft prior; a small correction head can override
    that prior for municipal exceptions such as soiled tissues.
    """

    def __init__(self, num_bins=3, num_materials=7, dropout=0.3, channels=(32, 64, 128, 256)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.features = nn.Sequential(
            _conv_block(3, c1),
            _conv_block(c1, c2),
            _conv_block(c2, c3),
            _conv_block(c3, c4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.material_head = nn.Linear(c4, num_materials)
        self.correction_head = nn.Sequential(
            nn.Linear(c4, c4 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, num_bins),
        )
        self.rule_logits = nn.Parameter(DEFAULT_RULE.clone())
        self.num_bins = num_bins
        self.num_materials = num_materials

    def forward(self, x):
        h = self.gap(self.features(x)).flatten(1)
        h = self.dropout(h)
        mat_logits = self.material_head(h)
        mat_probs = F.softmax(mat_logits, dim=1)
        rule = F.softmax(self.rule_logits, dim=1)
        prior = mat_probs @ rule
        correction = self.correction_head(h)
        # log-space mix keeps gradients well behaved; correction learns exceptions.
        bin_logits = torch.log(prior.clamp_min(1e-6)) + correction
        return bin_logits, mat_logits


class ResNet18Baseline(nn.Module):
    """ImageNet-pretrained ResNet-18 with shared FC and two heads (strong baseline)."""

    def __init__(self, num_bins=3, num_materials=7, dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.bin_head = nn.Linear(256, num_bins)
        self.material_head = nn.Linear(256, num_materials)

    def forward(self, x):
        h = self.shared(self.backbone(x))
        return self.bin_head(h), self.material_head(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str) -> nn.Module:
    if name == "hierarchical":
        return HierarchicalBinCNN()
    if name == "resnet18":
        return ResNet18Baseline()
    raise ValueError(name)


if __name__ == "__main__":
    for n in ("hierarchical", "resnet18"):
        m = build_model(n)
        x = torch.randn(2, 3, 224, 224)
        b, mat = m(x)
        print(n, "params", count_parameters(m), "bin", tuple(b.shape), "mat", tuple(mat.shape))
        print("  bins", BIN_CLASSES, "materials", MATERIAL_CLASSES)
