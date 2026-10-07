"""MobileNetV2 experiments: run variants, compare them on VALIDATION, pick the best.

Protocol (fixed before looking at any result):
  * every variant uses the same split, seed and (unless it is the thing being tested)
    the same settings as the default `mobilenet_v2` run;
  * variants are judged on the VALIDATION split only - the test split is not used;
  * best run = highest validation macro-F1; among runs within `tolerance` of it,
    the lowest validation loss. Differences below ~1 point are noise (small validation set).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .config import FINETUNE_ONLY_KEYS, MOBILENET_EXPERIMENTS
from .evaluate import evaluate_validation
from .train import run_dir_for, train_model


def run_mobilenet_experiments(names, train_df, val_df, dataset_root, work_dir, seed, pretrained=True,
                              force=False, verbose=2) -> dict:
    """Train the default MobileNetV2 (if needed) and every experiment in ``names``."""
    infos = {"mobilenet_v2": train_model("mobilenet_v2", train_df, val_df, dataset_root, work_dir, seed=seed,
                                         pretrained=pretrained, force=force, verbose=verbose)}
    default_stage1 = run_dir_for(work_dir, "mobilenet_v2", seed) / "stage1.keras"
    for name in names:
        overrides = MOBILENET_EXPERIMENTS[name]
        reuse = default_stage1 if set(overrides) <= FINETUNE_ONLY_KEYS and default_stage1.exists() else None
        infos[name] = train_model("mobilenet_v2", train_df, val_df, dataset_root, work_dir, seed=seed,
                                  run_name=name, overrides=overrides, stage1_from=reuse,
                                  pretrained=pretrained, force=force, verbose=verbose)
    return infos


def summarize_runs(work_dir, run_names, seed, val_df, dataset_root, force: bool = False) -> pd.DataFrame:
    """One row per run: settings, size, time and VALIDATION metrics."""
    rows = []
    for run in run_names:
        run_dir = run_dir_for(work_dir, run, seed)
        info = json.loads((run_dir / "train_info.json").read_text())
        m = evaluate_validation(run_dir, val_df, dataset_root, force=force)
        rows.append({
            "run": run,
            "overrides": json.dumps(info["overrides"]),
            "params (M)": round(info["params"] / 1e6, 2),
            "epochs": info["epochs_run"],
            "best_epoch": info["best_epoch"],
            "train_min": round(info["training_seconds"] / 60, 1),
            "val_loss": round(info["best_val_loss"], 4),
            "val_accuracy": round(m["accuracy"], 4),
            "val_macro_f1": round(m["macro_f1"], 4),
        })
    return pd.DataFrame(rows)


def pick_best(table: pd.DataFrame, tolerance: float = 0.005) -> pd.Series:
    """Highest val macro-F1; among runs within ``tolerance`` of it, lowest val loss."""
    top = table["val_macro_f1"].max()
    close = table[table["val_macro_f1"] >= top - tolerance]
    return close.sort_values(["val_loss", "params (M)"]).iloc[0]


def plot_experiments(table: pd.DataFrame, reference: str = "mobilenet_v2"):
    t = table.sort_values("val_macro_f1")
    colours = ["#C44E52" if r == reference else "#4C72B0" for r in t["run"]]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    ax.barh(t["run"], t["val_macro_f1"], color=colours)
    ax.set_xlim(max(0, t["val_macro_f1"].min() - 0.05), 1.0)
    for y, v in enumerate(t["val_macro_f1"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("validation macro-F1 (red = default run)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
