"""Metrics and plots for multiclass classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict:
    """Calculate common classification metrics using macro averaging."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if (
        y_true.ndim != 1
        or not len(y_true)
        or probabilities.shape != (len(y_true), len(class_names))
        or len(class_names) < 2
        or len(set(class_names)) != len(class_names)
        or not np.issubdtype(y_true.dtype, np.integer)
        or (y_true < 0).any()
        or (y_true >= len(class_names)).any()
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Expected nonempty integer labels and valid class probabilities")
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=range(len(class_names)), average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    try:
        if set(y_true) != set(range(len(class_names))):
            raise ValueError("ROC-AUC requires every class in the evaluation set")
        if len(class_names) == 2:
            auc = roc_auc_score(y_true, probabilities[:, 1])
        else:
            binary_labels = label_binarize(y_true, classes=range(len(class_names)))
            auc = roc_auc_score(binary_labels, probabilities, average="macro", multi_class="ovr")
        metrics["roc_auc_ovr_macro"] = float(auc)
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
    report = classification_report(
        y_true,
        probabilities.argmax(axis=1),
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "confusion_matrix.json").write_text(
        json.dumps({"class_names": class_names, "matrix": matrix.tolist()}, indent=2),
        encoding="utf-8",
    )
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
