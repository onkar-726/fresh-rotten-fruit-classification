"""Validation diagnostics for the models YOU train.

Scope note: the final TEST-set evaluation, confidence intervals and the model
comparison belong to the evaluation role (Rucha). Here we only need enough to
check that training worked and to choose between your own runs - and that must
be done on the VALIDATION split. The test split is never touched in this project.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from .config import CLASS_NAMES, NUM_CLASSES
from .datasets import make_dataset

LABELS = list(range(NUM_CLASSES))


def original_only(df: pd.DataFrame) -> pd.DataFrame:
    """One non-augmented image per source photo (helper for the evaluation role)."""
    return df[df["is_original"]].drop_duplicates("source_id").reset_index(drop=True)


def predict_df(model, df: pd.DataFrame, dataset_root, batch_size: int = 32) -> pd.DataFrame:
    """Run ``model`` over ``df`` (input size is read from the model) -> per-image table."""
    size = tuple(model.input_shape[1:3])
    ds = make_dataset(df, dataset_root, image_size=size, batch_size=batch_size, training=False, cache=False)
    probs = model.predict(ds, verbose=0)
    out = df[["image_path", "source_id", "class_name", "is_original", "aug_type", "label"]].copy()
    out = out.rename(columns={"label": "y_true"}).reset_index(drop=True)
    out["y_pred"] = probs.argmax(1)
    out["p_max"] = probs.max(1)
    for i, name in enumerate(CLASS_NAMES):
        out[f"p_{name}"] = probs[:, i]
    return out


def metrics_from_predictions(pred: pd.DataFrame) -> dict:
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    prec, rec, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "n_images": int(len(pred)),
        "n_sources": int(pred["source_id"].nunique()),
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1.mean()),
        "macro_precision": float(prec.mean()),
        "macro_recall": float(rec.mean()),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
            for i, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y, p, labels=LABELS).tolist(),
    }


def plot_confusion_matrix(cm, path=None, title="Confusion matrix", normalize=False):
    cm = np.asarray(cm, dtype=float)
    shown = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = shown[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}", ha="center", va="center",
                    color="white" if v > shown.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def evaluate_validation(run_dir, val_df: pd.DataFrame, dataset_root, force: bool = False) -> dict:
    """Validation metrics of one trained run (cached in <run_dir>/val_eval/)."""
    from tensorflow import keras

    run_dir = Path(run_dir)
    out = run_dir / "val_eval"
    cache = out / "metrics.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    out.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(run_dir / "model.keras")
    pred = predict_df(model, val_df, dataset_root)
    pred.to_csv(out / "predictions_validation.csv", index=False)
    m = metrics_from_predictions(pred)
    (out / "report_validation.txt").write_text(
        f"{run_dir.parent.name} | validation | {m['n_images']} images from {m['n_sources']} source photos\n"
        f"accuracy {m['accuracy']:.4f}   macro F1 {m['macro_f1']:.4f}\n\n"
        + classification_report(pred["y_true"], pred["y_pred"], labels=LABELS, target_names=CLASS_NAMES,
                                digits=4, zero_division=0)
    )
    fig = plot_confusion_matrix(m["confusion_matrix"], out / "confusion_validation.png",
                                f"{run_dir.parent.name} - validation")
    plt.close(fig)
    cache.write_text(json.dumps(m, indent=2))
    return m
