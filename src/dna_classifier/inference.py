"""Shared artifact and prediction validation for the CLI and local interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import LABELS, clean_sequence, standardize_sequence
from .preprocessing import encode_sequences

ARTIFACT_VERSION = 2


def read_settings(artifact_dir: Path) -> dict:
    path = artifact_dir / "settings.json"
    if not path.exists():
        raise ValueError("No trained model settings found. Run prepare-demo or train first.")
    settings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Model settings must be a JSON object")  # noqa: TRY004
    length = settings.get("sequence_length")
    if not isinstance(length, int) or isinstance(length, bool) or length < 18:
        raise ValueError("Model input length must be an integer of at least 18 bases")
    names = settings.get("class_names", [])
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or len(names) != len(LABELS)
        or set(names) != set(LABELS)
    ):
        raise ValueError("Model class names must contain each supported label exactly once")
    model_path = artifact_dir / "best_model.keras"
    if not model_path.exists():
        raise ValueError("Model file is missing. Run prepare-demo or train first.")
    if settings.get("model_sha256"):
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != settings["model_sha256"]:
            raise ValueError("Model and settings do not match. Retrain before predicting.")
    return settings


def load_artifacts(artifact_dir: Path):
    import tensorflow as tf

    settings = read_settings(artifact_dir)
    model = tf.keras.models.load_model(artifact_dir / "best_model.keras")
    if tuple(model.input_shape[1:]) != (settings["sequence_length"], 4):
        raise ValueError("Model shape does not match the input settings")
    if model.output_shape[-1] != len(settings["class_names"]):
        raise ValueError("Model output does not match the class names")
    return model, settings


def predict_sequence(model, settings: dict, sequence: str) -> dict:
    cleaned = clean_sequence(sequence)
    length = settings["sequence_length"]
    standardized = standardize_sequence(cleaned, length)
    if not (set(standardized) - {"N"}):
        raise ValueError("The model input has no known A/C/G/T bases to classify")
    probabilities = np.asarray(
        model.predict(encode_sequences([standardized], length), verbose=0)[0], dtype=float
    )
    if (
        probabilities.shape != (len(settings["class_names"]),)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Model returned invalid class probabilities")
    return {
        "prediction": settings["class_names"][int(probabilities.argmax())],
        "probabilities": dict(zip(settings["class_names"], probabilities.tolist())),
        "input_length": len(cleaned),
        "model_sequence": standardized,
        "known_bases": sum(base != "N" for base in standardized),
        "crop_start": max(0, (len(cleaned) - length) // 2),
        "padding_left": max(0, (length - len(cleaned)) // 2),
    }
