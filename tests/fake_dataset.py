"""Create a tiny synthetic dataset with the SAME structure/naming as the Kaggle one.

train/ and test/ folders, one folder per class, screenshot-style file names with
capture times, pre-augmented copies with prefixes, and originals that appear in
both folders. Images are coloured blobs so a model can learn something.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
FRUIT_RGB = {"apples": (200, 30, 40), "banana": (235, 210, 60), "oranges": (240, 130, 20)}


def _name(ts: datetime) -> str:
    h12 = ts.hour % 12 or 12
    return f"Screen Shot {ts:%Y-%m-%d} at {h12}.{ts.minute:02d}.{ts.second:02d} {'PM' if ts.hour >= 12 else 'AM'}.png"


def _draw(cls: str, rng: random.Random, size: int = 64) -> Image.Image:
    fruit = next(v for k, v in FRUIT_RGB.items() if cls.endswith(k))
    rotten = cls.startswith("rotten")
    bg = tuple(rng.randint(180, 255) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    base = tuple(int(c * (0.45 if rotten else 1.0)) for c in fruit)
    r = rng.randint(size // 4, size // 3)
    cx, cy = size // 2 + rng.randint(-6, 6), size // 2 + rng.randint(-6, 6)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=base)
    if rotten:
        for _ in range(rng.randint(3, 6)):
            x, y, s = cx + rng.randint(-r, r) // 2, cy + rng.randint(-r, r) // 2, rng.randint(2, 6)
            d.ellipse([x - s, y - s, x + s, y + s], fill=(60, 35, 15))
    return im


def _augment(im: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind.startswith("rotated_by_"):
        return im.rotate(int(kind.split("_")[-1]), fillcolor=(0, 0, 0))
    if kind == "vertical_flip":
        return im.transpose(Image.FLIP_TOP_BOTTOM)
    if kind == "translation":
        return im.transform(im.size, Image.AFFINE, (1, 0, 6, 0, 1, 4))
    arr = np.array(im)
    mask = np.random.default_rng(rng.randint(0, 9999)).random(arr.shape[:2])
    arr[mask < 0.02], arr[mask > 0.98] = 0, 255
    return Image.fromarray(arr)


def make_fake_dataset(root, sources_per_class: int = 60, seed: int = 0) -> Path:
    rng = random.Random(seed)
    root = Path(root) / "dataset"
    aug_kinds = ["rotated_by_15", "rotated_by_30", "rotated_by_75", "rotated_by_105",
                 "vertical_flip", "translation", "saltandpepper", "rotated_by_45"]
    for ci, cls in enumerate(CLASSES):
        for folder in ("train", "test"):
            (root / folder / cls).mkdir(parents=True, exist_ok=True)
        t = datetime(2018, 6, 7 + ci % 2, 10, 0, 0)
        for _ in range(sources_per_class):
            t += timedelta(seconds=rng.randint(8, 40))
            if rng.random() < 0.08:
                t += timedelta(minutes=rng.randint(5, 20))  # session break
            base = _name(t)
            im = _draw(cls, rng)
            files = {base: im}
            for k in aug_kinds:
                files[f"{k}_{base}"] = _augment(im, k, rng)
            for fname, img in files.items():
                folders = ["train"] if rng.random() < 0.7 else ["train", "test"]  # duplicates across folders
                if rng.random() < 0.1:
                    folders = ["test"]
                for folder in folders:
                    img.save(root / folder / cls / fname)
    return root.parent
