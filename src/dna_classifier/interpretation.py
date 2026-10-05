"""Simple, model-agnostic interpretation for individual DNA predictions."""

from __future__ import annotations

import numpy as np

from .data import standardize_sequence
from .preprocessing import one_hot_encode


def influential_windows(
    model, sequence: str, length: int, window_size: int = 10, top_k: int = 5
) -> list[dict]:
    """Rank windows by the prediction drop caused by masking them with N bases."""
    if window_size < 1 or top_k < 1:
        raise ValueError("window_size and top_k must be positive")
    standardized = standardize_sequence(sequence, length)
    original = one_hot_encode(standardized, length)
    original_probabilities = model.predict(original[None, ...], verbose=0)[0]
    predicted_class = int(original_probabilities.argmax())
    positions = list(range(0, length, window_size))
    batch = []
    for position in positions:
        masked = original.copy()
        masked[position : position + window_size] = 0
        batch.append(masked)
    masked_probabilities = model.predict(np.stack(batch), verbose=0)[:, predicted_class]
    candidates = []
    for position, masked_probability in zip(positions, masked_probabilities):
        candidates.append(
            {
                "start": position,
                "end": min(position + window_size, length),
                "sequence": standardized[position : position + window_size],
                "importance": float(original_probabilities[predicted_class] - masked_probability),
            }
        )
    return sorted(candidates, key=lambda item: item["importance"], reverse=True)[:top_k]
