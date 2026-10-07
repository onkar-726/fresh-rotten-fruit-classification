"""Plots used in the notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from .config import CLASS_NAMES  # noqa: E402


def sample_grid(index: pd.DataFrame, dataset_root, per_class: int = 4, seed: int = 0):
    """A grid of original photos, one row per class."""
    rng = np.random.default_rng(seed)
    orig = index[index.is_original]
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(2.2 * per_class, 2.2 * len(CLASS_NAMES)))
    for r, cls in enumerate(CLASS_NAMES):
        rows = orig[orig.class_name == cls]
        pick = rows.iloc[rng.choice(len(rows), min(per_class, len(rows)), replace=False)]
        for c in range(per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(Path(dataset_root) / pick.iloc[c].image_path).convert("RGB"))
            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")
    fig.tight_layout()
    return fig


def split_timeline(index_split: pd.DataFrame):
    """Capture time of every source photo, coloured by split, one row per class."""
    colours = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868", "dropped": "#BBBBBB"}
    src = index_split.drop_duplicates("source_id")
    src = src[src.capture_time.notna()]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, cls in enumerate(CLASS_NAMES):
        for split, col in colours.items():
            s = src[(src.class_name == cls) & (src.split == split)]
            ax.scatter(s.capture_time, [i] * len(s), s=10, color=col, label=split if i == 0 else None)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("capture time (from file name)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    return fig


def plot_history(history_csv, title: str = ""):
    """Loss and accuracy per epoch (train vs validation); dashed line = stage change."""
    h = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(h["epoch"], h[key], label="train")
        ax.plot(h["epoch"], h[f"val_{key}"], label="validation")
        if "stage" in h and h["stage"].nunique() > 1:
            switch = h.index[h["stage"] != h["stage"].iloc[0]][0]
            ax.axvline(h["epoch"].iloc[switch] - 0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
