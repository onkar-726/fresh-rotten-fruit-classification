from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
):
    """
    Evaluate a model and return true labels, predictions,
    and class probabilities.
    """

    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    true_labels = np.concatenate(
        [
            labels.numpy()
            for _, labels in dataset
        ],
        axis=0,
    )

    if len(true_labels) != len(predictions):
        raise ValueError(
            "Number of labels and predictions does not match."
        )

    return (
        true_labels,
        predictions,
        probabilities,
    )


def export_results(
    y_true,
    y_pred,
    probabilities,
    class_names,
    output_dir,
    prefix,
    metadata=None,
):
    """
    Save classification report, confusion matrix,
    summary JSON, and prediction CSV.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probabilities = np.asarray(probabilities)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    report_path = (
        output_dir
        / f"{prefix}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.to_csv(
        output_dir
        / f"{prefix}_confusion_matrix.csv"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        matrix,
        interpolation="nearest",
    )

    plt.title(
        f"{prefix} Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(
        np.arange(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(len(class_names)),
        class_names,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / f"{prefix}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
    }

    if metadata is not None:
        summary["metadata"] = metadata

    summary_path = (
        output_dir
        / f"{prefix}_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    prediction_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "true_class": [
                class_names[int(i)]
                for i in y_true
            ],
            "predicted_class": [
                class_names[int(i)]
                for i in y_pred
            ],
            "confidence": probabilities.max(
                axis=1
            ),
        }
    )

    prediction_df.to_csv(
        output_dir
        / f"{prefix}_predictions.csv",
        index=False,
    )

    return summary


def plot_history(
    history,
    output_dir,
    prefix,
):
    """
    Save training history as CSV and
    create accuracy and loss plots.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_data = history.history

    history_df = pd.DataFrame(
        history_data
    )

    history_df.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(history_df) + 1,
        ),
    )

    history_csv = (
        output_dir
        / f"{prefix}_history.csv"
    )

    history_df.to_csv(
        history_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Accuracy
    # -----------------------------------------------------

    if (
        "accuracy" in history_df.columns
        and "val_accuracy" in history_df.columns
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history_df["epoch"],
            history_df["accuracy"],
            label="Train",
        )

        plt.plot(
            history_df["epoch"],
            history_df["val_accuracy"],
            label="Validation",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(
            f"{prefix} Accuracy"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{prefix}_accuracy.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    if (
        "loss" in history_df.columns
        and "val_loss" in history_df.columns
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history_df["epoch"],
            history_df["loss"],
            label="Train",
        )

        plt.plot(
            history_df["epoch"],
            history_df["val_loss"],
            label="Validation",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(
            f"{prefix} Loss"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            output_dir
            / f"{prefix}_loss.png",
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

    return history_df