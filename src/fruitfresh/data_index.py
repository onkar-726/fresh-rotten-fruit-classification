"""Build a clean table ("index") of every image in the Kaggle dataset.

What the dataset looks like (this is why the index matters):

* The Kaggle folder has ``train/`` and ``test/`` sub-folders, each with one
  folder per class.
* There are only ~1,500 ORIGINAL photos. Most files are pre-augmented copies of
  those originals (rotated, flipped, translated, salt-and-pepper noise); the
  augmentation is written into the file-name prefix, e.g.
  ``rotated_by_15_Screen Shot 2018-06-08 at 5.18.26 PM.png``.
* The same original photo can appear in BOTH the Kaggle train and test folders.
* The originals are screenshots whose file name contains the capture time.
  Screenshots taken seconds apart usually show the same physical fruit.

So every file is described by: its class, its *source photo* (``source_id``),
whether it is an original or an augmented copy, and when it was captured.
"""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import CLASS_NAMES, CLASS_TO_INDEX, KAGGLE_DATASET

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KAGGLE_FOLDERS = ("train", "test")

# Augmentation prefixes used by the dataset authors. Several can be chained.
AUG_RE = re.compile(
    r"^(?P<aug>rotated_by_\d+|vertical_flip|horizontal_flip|translation|"
    r"saltandpepper|salt_and_pepper)_(?P<base>.+)$",
    re.IGNORECASE,
)
# "Screen Shot 2018-06-08 at 5.18.26 PM.png"  (also tolerates U+202F before PM)
TIME_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})\s+at\s+(\d{1,2})\.(\d{2})\.(\d{2})\s*([AP]M)",
    re.IGNORECASE,
)


def parse_capture_time(name: str):
    """Return the capture time found in a screenshot file name, or NaT."""
    m = TIME_RE.search(name)
    if not m:
        return pd.NaT
    y, mo, d, h, mi, s, ampm = m.groups()
    try:
        return pd.Timestamp(
            datetime.strptime(f"{y}-{mo}-{d} {h}:{mi}:{s} {ampm.upper()}", "%Y-%m-%d %I:%M:%S %p")
        )
    except ValueError:
        return pd.NaT


def parse_filename(file_name: str) -> dict:
    """Split a file name into (augmentation prefix, original name, capture time)."""
    name = file_name
    augs = []
    while True:
        m = AUG_RE.match(name)
        if not m:
            break
        augs.append(m.group("aug").lower())
        name = m.group("base")
    return {
        "base_name": name,
        "aug_type": "+".join(augs),
        "is_original": not augs,
        "capture_time": parse_capture_time(name),
    }


def resolve_dataset_root(path) -> Path:
    """Find the folder that directly contains ``train/`` and ``test/``."""
    path = Path(path)
    candidates = [path, path / "dataset"]
    candidates += [p for p in path.glob("*") if p.is_dir()]
    for cand in candidates:
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find train/ and test/ folders under {path}")


def get_dataset_root() -> Path:
    """Dataset root from FRUIT_DATASET_ROOT, or download from Kaggle (public)."""
    env = os.getenv("FRUIT_DATASET_ROOT")
    if env:
        return resolve_dataset_root(env)
    import kagglehub  # imported lazily so unit tests do not need it

    return resolve_dataset_root(kagglehub.dataset_download(KAGGLE_DATASET))


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_index(dataset_root, drop_exact_duplicates: bool = True) -> pd.DataFrame:
    """One row per image file with source/augmentation/time information."""
    root = Path(dataset_root)
    rows = []
    for folder in KAGGLE_FOLDERS:
        folder_path = root / folder
        for class_dir in sorted(p for p in folder_path.iterdir() if p.is_dir()):
            if class_dir.name not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected class folder: {class_dir.name}")
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                info = parse_filename(f.name)
                rows.append(
                    {
                        "image_path": f"{folder}/{class_dir.name}/{f.name}",
                        "kaggle_folder": folder,
                        "class_name": class_dir.name,
                        "file_name": f.name,
                        **info,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {root}")

    n_dupes = 0
    if drop_exact_duplicates:
        dup = df[df.duplicated(["class_name", "file_name"], keep=False)]
        if len(dup):
            md5 = [_md5(root / p) for p in dup.image_path]
            dup = dup.assign(md5=md5)
            drop_idx = dup[dup.duplicated(["class_name", "file_name", "md5"], keep="first")].index
            n_dupes = len(drop_idx)
            df = df.drop(index=drop_idx)

    df = df.reset_index(drop=True)
    df["source_id"] = df["class_name"] + "/" + df["base_name"]
    df["has_timestamp"] = df["capture_time"].notna()
    df["label"] = df["class_name"].map(CLASS_TO_INDEX).astype(int)
    df.attrs["exact_duplicates_dropped"] = n_dupes
    return df


def summarize_index(index: pd.DataFrame) -> dict:
    """Numbers worth printing and quoting in the report."""
    by_folder = index.groupby("source_id")["kaggle_folder"].nunique()
    times = index.drop_duplicates("source_id")["capture_time"].dropna()
    return {
        "n_images": int(len(index)),
        "n_sources": int(index.source_id.nunique()),
        "share_original_images": float(index.is_original.mean()),
        "n_original_images": int(index.is_original.sum()),
        "sources_in_both_kaggle_folders": int((by_folder > 1).sum()),
        "sources_with_timestamp": int(index.drop_duplicates("source_id").has_timestamp.sum()),
        "capture_dates": sorted({str(t.date()) for t in times}),
        "augmentation_types": index.loc[~index.is_original, "aug_type"]
        .str.replace(r"_\d+", "_N", regex=True)
        .value_counts()
        .to_dict(),
        "exact_duplicates_dropped": int(index.attrs.get("exact_duplicates_dropped", 0)),
    }


def find_unreadable(index: pd.DataFrame, dataset_root, limit: int | None = None) -> list[str]:
    """Return image paths that PIL cannot open (corrupt files)."""
    from PIL import Image

    bad = []
    paths = index.image_path.tolist()[:limit]
    for p in paths:
        try:
            with Image.open(Path(dataset_root) / p) as im:
                im.verify()
        except Exception:  # noqa: BLE001 - any failure means "unreadable"
            bad.append(p)
    return bad
