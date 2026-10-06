"""
Time-block split with a time buffer for the fresh/rotten fruit dataset.

Why: the source images are screenshots taken seconds apart, so neighbouring
screenshots of the same class are usually the same physical fruit.

Grouping by exact minute still leaves train/test photos only a few seconds
apart. This script instead:

1. sorts the source photos of each class by capture time,
2. cuts each class timeline into contiguous blocks,
3. assigns WHOLE blocks to train / validation / test (per class, seeded),
4. drops every source photo that lies within BUFFER_SECONDS of a source
   photo of a different split (same class), so the splits are separated
   by a real time gap.

All augmented copies of a source photo always stay with that source photo.

Usage (from the repo root):

    python src/make_time_block_split.py

or from a notebook:

    from src.make_time_block_split import build_time_block_split
    train_df, val_df, test_df = build_time_block_split("data/session_aware")
"""

from pathlib import Path

import hashlib
import json

import numpy as np
import pandas as pd


SEED = 42
BLOCK_SIZE = 25
BUFFER_SECONDS = 60
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]


def load_all_images(input_dir):
    """Read the three existing CSVs and return one row per image."""

    input_dir = Path(input_dir)

    frames = [
        pd.read_csv(
            input_dir / f"{name}_session_aware.csv"
        )
        for name in (
            "train",
            "validation",
            "test",
        )
    ]

    images = pd.concat(
        frames,
        ignore_index=True
    )

    images["capture_time"] = pd.to_datetime(
        images["capture_time"]
    )

    assert images["image_path"].is_unique

    return images


def assign_blocks(sources, rng):
    """Give every source a split by assigning whole time blocks."""

    sources = (
        sources
        .sort_values(
            [
                "class_name",
                "capture_time"
            ]
        )
        .copy()
    )

    sources["split_tb"] = ""
    sources["block_id"] = ""

    for class_name, group in sources.groupby("class_name"):

        idx = group.index.to_numpy()
        n = len(idx)

        block_numbers = (
            np.arange(n) // BLOCK_SIZE
        )

        sources.loc[idx, "block_id"] = [
            f"{class_name}::block_{b}"
            for b in block_numbers
        ]

        block_ids = np.unique(
            block_numbers
        )

        rng.shuffle(block_ids)

        sizes = {
            b: int(
                (block_numbers == b).sum()
            )
            for b in block_ids
        }

        target_val = (
            VAL_FRACTION * n
        )

        target_test = (
            TEST_FRACTION * n
        )

        counts = {
            "validation": 0,
            "test": 0
        }

        targets = {
            "validation": target_val,
            "test": target_test
        }

        assignment = {}

        for b in block_ids:

            # Start with train.
            best = "train"
            best_gain = 0.0

            # Check whether assigning this block to an
            # evaluation split improves its target closeness.
            for name in (
                "test",
                "validation"
            ):

                before = abs(
                    counts[name]
                    - targets[name]
                )

                after = abs(
                    counts[name]
                    + sizes[b]
                    - targets[name]
                )

                gain = (
                    before
                    - after
                )

                if gain > best_gain:
                    best = name
                    best_gain = gain

            assignment[b] = best

            if best != "train":
                counts[best] += sizes[b]

        sources.loc[idx, "split_tb"] = [
            assignment[b]
            for b in block_numbers
        ]

    return sources


def drop_buffer_zone(
    sources,
    buffer_seconds
):
    """
    Drop sources that are too close in time to a source of another split
    (same class).

    If a train source is involved, the train source is dropped.
    For a validation/test pair, the validation source is dropped.
    """

    keep = pd.Series(
        True,
        index=sources.index
    )

    # Lower value = dropped first.
    priority = {
        "train": 0,
        "validation": 1,
        "test": 2
    }

    for _, group in sources.groupby(
        "class_name"
    ):

        times = (
            group["capture_time"]
            .to_numpy()
        )

        splits = (
            group["split_tb"]
            .to_numpy()
        )

        gaps = (
            np.abs(
                times[:, None]
                - times[None, :]
            )
            / np.timedelta64(1, "s")
        )

        different = (
            splits[:, None]
            != splits[None, :]
        )

        too_close = (
            (gaps < buffer_seconds)
            & different
        )

        for i in range(len(group)):

            for j in np.where(
                too_close[i]
            )[0]:

                if j <= i:
                    continue

                loser = (
                    i
                    if priority[splits[i]]
                    < priority[splits[j]]
                    else j
                )

                keep.loc[
                    group.index[loser]
                ] = False

    return (
        sources[keep].copy(),
        int((~keep).sum())
    )


def verify(
    sources,
    buffer_seconds
):
    """Hard checks: no source overlap and a real time gap between splits."""

    ids = {
        split_name: set(
            group["source_id"]
        )
        for split_name, group
        in sources.groupby("split_tb")
    }

    assert ids["train"].isdisjoint(
        ids["validation"]
    )

    assert ids["train"].isdisjoint(
        ids["test"]
    )

    assert ids["validation"].isdisjoint(
        ids["test"]
    )

    worst = np.inf

    for _, group in sources.groupby(
        "class_name"
    ):

        times = (
            group["capture_time"]
            .to_numpy()
        )

        splits = (
            group["split_tb"]
            .to_numpy()
        )

        gaps = (
            np.abs(
                times[:, None]
                - times[None, :]
            )
            / np.timedelta64(1, "s")
        )

        different = (
            splits[:, None]
            != splits[None, :]
        )

        if different.any():
            worst = min(
                worst,
                gaps[different].min()
            )

    assert worst >= buffer_seconds, (
        f"closest cross-split gap is {worst}s"
    )

    for (
        split_name,
        group
    ) in sources.groupby("split_tb"):

        assert set(
            group["class_name"]
        ) == set(CLASS_NAMES), split_name

    return worst


def build_time_block_split(
    input_dir="data/session_aware",
    output_dir="data/time_block",
    save=True
):
    images = load_all_images(
        input_dir
    )

    sources = (
        images
        .drop_duplicates("source_id")
        [
            [
                "source_id",
                "class_name",
                "capture_time"
            ]
        ]
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(
        SEED
    )

    sources = assign_blocks(
        sources,
        rng
    )

    before = len(sources)

    sources, dropped = drop_buffer_zone(
        sources,
        BUFFER_SECONDS
    )

    closest_gap = verify(
        sources,
        BUFFER_SECONDS
    )

    print(
        f"Source photos before buffer : {before}"
    )

    print(
        f"Dropped by {BUFFER_SECONDS}s buffer      : {dropped}"
    )

    print(
        f"Source photos kept          : {len(sources)}"
    )

    print(
        f"Closest cross-split gap     : {closest_gap:.0f} s"
    )

    # ---------------------------------------------------------
    # Apply the new time-block assignment to all image records
    # ---------------------------------------------------------

    lookup = (
        sources
        .set_index("source_id")
        [
            [
                "split_tb",
                "block_id"
            ]
        ]
    )

    images = images.join(
        lookup,
        on="source_id",
        how="inner"
    )

    # Preserve the previous session-aware split
    # for provenance.
    images["session_aware_split"] = (
        images["split"]
    )

    # The standard split column now represents
    # the NEW time-block split.
    images["split"] = (
        images["split_tb"]
    )

    # Keep an explicit copy for clarity.
    images["time_block_split"] = (
        images["split_tb"]
    )

    images = images.drop(
        columns=["split_tb"]
    )

    # ---------------------------------------------------------
    # Create train / validation / test outputs
    # ---------------------------------------------------------

    outputs = {}

    for name in (
        "train",
        "validation",
        "test"
    ):

        part = (
            images[
                images["time_block_split"]
                == name
            ]
            .reset_index(drop=True)
        )

        outputs[name] = part

        n_src = (
            part["source_id"]
            .nunique()
        )

        print(
            f"{name:<10} "
            f"images={len(part):>5} "
            f"sources={n_src:>4}"
        )

        print(
            "            ",
            part["class_name"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

    # ---------------------------------------------------------
    # Save output CSV files
    # ---------------------------------------------------------

    if save:

        out = Path(
            output_dir
        )

        out.mkdir(
            parents=True,
            exist_ok=True
        )

        for (
            name,
            part
        ) in outputs.items():

            part.to_csv(
                out
                / f"{name}_time_block.csv",
                index=False
            )

        print(
            f"Saved CSVs to {out}/"
        )

        write_split_manifest(
            output_dir=out,
            outputs=outputs,
            sources_before=before,
            dropped=dropped,
            closest_gap=closest_gap,
        )

    return (
        outputs["train"],
        outputs["validation"],
        outputs["test"]
    )


def write_split_manifest(
    output_dir,
    outputs,
    sources_before,
    dropped,
    closest_gap,
):
    output_dir = Path(output_dir)

    manifest = {
        "split_version": "v2",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "buffer_seconds": BUFFER_SECONDS,
        "validation_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "source_photos_before_buffer": int(sources_before),
        "source_photos_dropped": int(dropped),
        "source_photos_kept": int(
            sum(
                df["source_id"].nunique()
                for df in outputs.values()
            )
        ),
        "closest_cross_split_gap_seconds": float(
            closest_gap
        ),
        "image_counts": {},
        "source_counts": {},
        "class_counts": {},
        "blocks_per_class": {},
        "csv_sha256": {},
    }

    for split_name, df in outputs.items():

        manifest["image_counts"][split_name] = int(
            len(df)
        )

        manifest["source_counts"][split_name] = int(
            df["source_id"].nunique()
        )

        manifest["class_counts"][split_name] = {
            class_name: int(count)
            for class_name, count in (
                df["class_name"]
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        }

        manifest["blocks_per_class"][split_name] = {
            class_name: int(count)
            for class_name, count in (
                df.groupby("class_name")["block_id"]
                .nunique()
                .to_dict()
                .items()
            )
        }

        csv_path = (
            output_dir
            / f"{split_name}_time_block.csv"
        )

        sha256 = hashlib.sha256()

        with open(csv_path, "rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                sha256.update(chunk)

        manifest["csv_sha256"][split_name] = (
            sha256.hexdigest()
        )

    manifest_path = (
        output_dir
        / "split_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print(
        f"Saved manifest to {manifest_path}"
    )


if __name__ == "__main__":
    build_time_block_split()