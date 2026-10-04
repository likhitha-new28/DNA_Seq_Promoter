"""Metrics and plots for multiclass classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict:
    """Calculate common classification metrics using macro averaging."""
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    try:
        binary_labels = label_binarize(y_true, classes=range(len(class_names)))
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(binary_labels, probabilities, average="macro", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None
    return metrics


def save_evaluation(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_dir: str | Path,
) -> dict:
    """Write metrics JSON and a labeled confusion-matrix image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(y_true, probabilities, class_names)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_true, probabilities.argmax(axis=1), labels=range(len(class_names)))
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return metrics


def save_learning_curves(history: dict, output_path: str | Path) -> None:
    """Plot training and validation loss and accuracy."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
