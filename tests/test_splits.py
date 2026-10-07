import numpy as np
import pandas as pd
import pytest

from fruitfresh.config import CLASS_NAMES
from fruitfresh.splits import SPLITS, make_time_block_split, save_split, load_split, verify_split


def synthetic_index(n_per_class=140, copies=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cls in CLASS_NAMES:
        t = pd.Timestamp("2018-06-07 10:00:00")
        for i in range(n_per_class):
            t += pd.Timedelta(seconds=int(rng.integers(6, 40)))
            if rng.random() < 0.05:
                t += pd.Timedelta(minutes=int(rng.integers(5, 30)))
            sid = f"{cls}/shot_{i:04d}.png"
            for c in range(copies):
                rows.append({"image_path": f"train/{cls}/{c}_{i}.png", "class_name": cls, "source_id": sid,
                             "capture_time": t, "is_original": c == 0, "aug_type": "" if c == 0 else "x",
                             "label": CLASS_NAMES.index(cls)})
    return pd.DataFrame(rows)


def test_time_block_split_is_leak_free_and_reproducible():
    idx = synthetic_index()
    a, rep = make_time_block_split(idx, seed=1, block_size=20, buffer_seconds=60)
    b, _ = make_time_block_split(idx, seed=1, block_size=20, buffer_seconds=60)
    assert a.split.tolist() == b.split.tolist()          # reproducible
    stats = verify_split(a, buffer_seconds=60)            # raises if leaky
    assert stats["min_cross_split_gap_seconds"] >= 60
    # all copies of a source stay together
    assert (a.groupby("source_id").split.nunique() == 1).all()
    assert stats["n_sources_dropped"] == rep["n_sources_dropped_by_buffer"]
    sizes = stats["sources"]
    assert sizes["train"] > sizes["validation"] and sizes["train"] > sizes["test"]


def test_different_seed_changes_assignment():
    idx = synthetic_index()
    a, _ = make_time_block_split(idx, seed=1, block_size=20)
    b, _ = make_time_block_split(idx, seed=2, block_size=20)
    assert a.split.tolist() != b.split.tolist()


def test_too_few_blocks_raises():
    with pytest.raises(ValueError):
        make_time_block_split(synthetic_index(n_per_class=30), block_size=25)


def test_save_and_load_roundtrip_detects_edits(tmp_path):
    idx = synthetic_index()
    s, _ = make_time_block_split(idx, block_size=20)
    save_split(s, tmp_path, {"seed": 42}, verify_split(s))
    tr, va, te = load_split(tmp_path)
    assert len(tr) + len(va) + len(te) == int(s.split.isin(SPLITS).sum())
    (tmp_path / "test.csv").write_text((tmp_path / "test.csv").read_text() + " ")
    with pytest.raises(AssertionError):
        load_split(tmp_path)
