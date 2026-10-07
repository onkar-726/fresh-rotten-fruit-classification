"""Leakage-safe train / validation / test splits.

Two problems must be handled for this dataset (see docs/METHODOLOGY.md):

1. AUGMENTED COPIES. All copies of one original photo share a ``source_id`` and
   must stay together, otherwise the test set contains rotated/flipped versions
   of training photos.
2. NEIGHBOURING SCREENSHOTS. Originals taken seconds apart often show the same
   fruit. The *time-block split* therefore
     a. sorts the source photos of each class by capture time,
     b. cuts the timeline into contiguous blocks,
     c. gives WHOLE blocks to train / validation / test,
     d. drops every source closer than ``buffer_seconds`` to a source of a
        different split (same class), leaving a real time gap between splits.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_NAMES

SPLITS = ("train", "validation", "test")
_DROP_PRIORITY = {"train": 0, "validation": 1, "test": 2}  # lower = dropped first


# ----------------------------------------------------------------- helpers
def _source_table(index: pd.DataFrame) -> pd.DataFrame:
    src = (
        index.groupby("source_id", sort=True)
        .agg(class_name=("class_name", "first"), capture_time=("capture_time", "min"))
        .reset_index()
    )
    src["has_timestamp"] = src["capture_time"].notna()
    src["split"] = ""
    src["block_id"] = ""
    return src


def _timed_blocks(src: pd.DataFrame, class_name: str, block_size: int):
    """Timed sources of one class sorted by time + their block numbers."""
    cls = src[(src.class_name == class_name) & src.has_timestamp]
    cls = cls.sort_values(["capture_time", "source_id"], kind="mergesort")
    return cls, np.arange(len(cls)) // block_size


def _assign_blocks(block_numbers, val_fraction, test_fraction, rng) -> dict:
    """Greedy block assignment so val/test sizes land close to their targets."""
    ids = np.unique(block_numbers)
    if len(ids) < 3:
        raise ValueError(
            f"Only {len(ids)} time blocks for a class; need >= 3. Lower block_size."
        )
    sizes = {b: int((block_numbers == b).sum()) for b in ids}
    n = len(block_numbers)
    targets = {"test": test_fraction * n, "validation": val_fraction * n}
    counts = {"test": 0, "validation": 0}
    order = ids.copy()
    rng.shuffle(order)
    assign = {}
    for b in order:
        best, best_gain = "train", 0.0
        for name in ("test", "validation"):
            gain = abs(counts[name] - targets[name]) - abs(counts[name] + sizes[b] - targets[name])
            if gain > best_gain:
                best, best_gain = name, gain
        assign[b] = best
        if best != "train":
            counts[best] += sizes[b]
    for name in ("test", "validation"):  # every split gets at least one block
        if counts[name] == 0:
            train_blocks = [b for b in order if assign[b] == "train"]
            smallest = min(train_blocks, key=lambda b: sizes[b])
            assign[smallest] = name
            counts[name] += sizes[smallest]
    if not any(v == "train" for v in assign.values()):
        raise ValueError("No block left for training; lower block_size.")
    return assign


def _random_assign(n, val_fraction, test_fraction, rng):
    perm = rng.permutation(n)
    n_test, n_val = int(round(test_fraction * n)), int(round(val_fraction * n))
    out = np.array(["train"] * n, dtype=object)
    out[perm[:n_test]] = "test"
    out[perm[n_test : n_test + n_val]] = "validation"
    return out


def _apply_buffer(src: pd.DataFrame, buffer_seconds: float) -> int:
    """Mark sources too close in time to another split as 'dropped'."""
    dropped = 0
    for class_name, cls in src[src.has_timestamp & src.split.isin(SPLITS)].groupby("class_name"):
        t = cls["capture_time"].to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        sp = cls["split"].to_numpy()
        pr = np.array([_DROP_PRIORITY[s] for s in sp])
        close = (np.abs(t[:, None] - t[None, :]) < buffer_seconds) & (sp[:, None] != sp[None, :])
        lose = np.zeros(len(cls), dtype=bool)
        for i, j in np.argwhere(np.triu(close, 1)):
            lose[i if pr[i] < pr[j] else j] = True
        src.loc[cls.index[lose], "split"] = "dropped"
        dropped += int(lose.sum())
    return dropped


def _attach(index: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    out = index.drop(columns=[c for c in ("split", "block_id") if c in index.columns])
    return out.merge(src[["source_id", "split", "block_id"]], on="source_id", how="left")


# --------------------------------------------------------------- the splits
def make_time_block_split(
    index: pd.DataFrame,
    seed: int = 42,
    block_size: int = 25,
    buffer_seconds: float = 60,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
):
    """Official split. Returns (index with 'split' column, report dict)."""
    rng = np.random.default_rng(seed)
    src = _source_table(index)
    for class_name in sorted(src.class_name.unique()):
        timed, blocks = _timed_blocks(src, class_name, block_size)
        if len(timed):
            assign = _assign_blocks(blocks, val_fraction, test_fraction, rng)
            src.loc[timed.index, "block_id"] = [f"{class_name}::block_{b:03d}" for b in blocks]
            src.loc[timed.index, "split"] = [assign[b] for b in blocks]
        untimed = src[(src.class_name == class_name) & ~src.has_timestamp]
        if len(untimed):  # fallback: no timestamp -> random by source
            src.loc[untimed.index, "split"] = _random_assign(
                len(untimed), val_fraction, test_fraction, rng
            )
    n_before = int(len(src))
    n_dropped = _apply_buffer(src, buffer_seconds)
    out = _attach(index, src)
    report = {
        "method": "time_block",
        "n_sources_before_buffer": n_before,
        "n_sources_dropped_by_buffer": n_dropped,
        "n_sources_untimed": int((~src.has_timestamp).sum()),
    }
    return out, report


# ------------------------------------------------------------- verification
def _min_cross_split_gap(kept: pd.DataFrame):
    src = kept.drop_duplicates("source_id")
    src = src[src.capture_time.notna()]
    worst = np.inf
    for _, cls in src.groupby("class_name"):
        t = cls["capture_time"].to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9
        sp = cls["split"].to_numpy()
        diff = sp[:, None] != sp[None, :]
        if diff.any():
            worst = min(worst, float(np.abs(t[:, None] - t[None, :])[diff].min()))
    return None if worst == np.inf else worst


def verify_split(index_split: pd.DataFrame, buffer_seconds: float | None = None) -> dict:
    """Hard checks. Raises AssertionError if the split is not leakage-safe."""
    kept = index_split[index_split.split.isin(SPLITS)]
    sets = {s: set(kept.loc[kept.split == s, "source_id"]) for s in SPLITS}
    assert sets["train"].isdisjoint(sets["validation"]), "source overlap train/validation"
    assert sets["train"].isdisjoint(sets["test"]), "source overlap train/test"
    assert sets["validation"].isdisjoint(sets["test"]), "source overlap validation/test"
    paths = {s: set(kept.loc[kept.split == s, "image_path"]) for s in SPLITS}
    assert paths["train"].isdisjoint(paths["test"]) and paths["train"].isdisjoint(paths["validation"])
    for s in SPLITS:
        got = set(kept.loc[kept.split == s, "class_name"])
        assert got == set(CLASS_NAMES), f"{s} is missing classes: {set(CLASS_NAMES) - got}"
    min_gap = _min_cross_split_gap(kept)
    if buffer_seconds is not None and min_gap is not None:
        assert min_gap >= buffer_seconds, f"closest cross-split gap is {min_gap:.0f}s < {buffer_seconds}s"
    blocks = (
        kept[kept.block_id != ""].drop_duplicates("source_id")
        .groupby(["class_name", "split"]).block_id.nunique().unstack(fill_value=0)
        if "block_id" in kept else None
    )
    return {
        "images": {s: int((kept.split == s).sum()) for s in SPLITS},
        "sources": {s: len(sets[s]) for s in SPLITS},
        "images_per_class": {
            s: kept[kept.split == s].class_name.value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict()
            for s in SPLITS
        },
        "blocks_per_class": None if blocks is None else blocks.reindex(CLASS_NAMES).to_dict(orient="index"),
        "n_images_dropped": int((index_split.split == "dropped").sum()),
        "n_sources_dropped": int(index_split[index_split.split == "dropped"].source_id.nunique()),
        "min_cross_split_gap_seconds": min_gap,
    }


# -------------------------------------------------------------- persistence
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_split(index_split: pd.DataFrame, out_dir, params: dict, stats: dict) -> dict:
    """Write train/validation/test CSVs + split_manifest.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "stats": stats,
        "files": {},
    }
    for s in SPLITS:
        path = out / f"{s}.csv"
        index_split[index_split.split == s].reset_index(drop=True).to_csv(path, index=False)
        manifest["files"][f"{s}.csv"] = _sha256(path)
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def load_split(split_dir):
    """Return (train_df, validation_df, test_df) written by ``save_split``."""
    d = Path(split_dir)
    frames = []
    for s in SPLITS:
        df = pd.read_csv(d / f"{s}.csv", parse_dates=["capture_time"])
        df["is_original"] = df["is_original"].astype(bool)
        df["aug_type"] = df["aug_type"].fillna("")
        frames.append(df)
    manifest = d / "split_manifest.json"
    if manifest.exists():  # detect an edited / mismatching CSV
        recorded = json.loads(manifest.read_text())["files"]
        for s in SPLITS:
            assert _sha256(d / f"{s}.csv") == recorded[f"{s}.csv"], f"{s}.csv changed since the split was created"
    return tuple(frames)
