"""Convert DNA strings to arrays and create leak-free dataset splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import LABELS, standardize_sequence

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class DatasetSplit:
    """The three non-overlapping pieces used during model development."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def one_hot_encode(sequence: str, length: int) -> np.ndarray:
    """Encode A/C/G/T as four channels; unknown or padded N bases remain zero."""
    sequence = standardize_sequence(sequence, length)
    encoded = np.zeros((length, 4), dtype=np.float32)
    for position, base in enumerate(sequence):
        if base in BASE_TO_INDEX:
            encoded[position, BASE_TO_INDEX[base]] = 1.0
    return encoded


def encode_sequences(sequences: list[str] | pd.Series, length: int) -> np.ndarray:
    """Encode multiple sequences into a model-ready 3D array."""
    return np.stack([one_hot_encode(sequence, length) for sequence in sequences])


def encode_labels(labels: list[str] | pd.Series) -> np.ndarray:
    """Turn label names into stable integer class IDs."""
    lookup = {label: index for index, label in enumerate(LABELS)}
    try:
        return np.asarray([lookup[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Unknown label: {error.args[0]}") from error


def stratified_split(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    """Create reproducible splits while preserving class proportions."""
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than 1")
    train_validation, test = train_test_split(
        frame, test_size=test_size, stratify=frame["label"], random_state=seed
    )
    relative_validation_size = validation_size / (1 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation_size,
        stratify=train_validation["label"],
        random_state=seed,
    )
    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
