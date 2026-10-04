"""Simple, model-agnostic interpretation for individual DNA predictions."""

from __future__ import annotations

from .data import standardize_sequence
from .preprocessing import one_hot_encode


def influential_windows(
    model, sequence: str, length: int, window_size: int = 10, top_k: int = 5
) -> list[dict]:
    """Rank windows by the prediction drop caused by masking them with N bases."""
    standardized = standardize_sequence(sequence, length)
    original = one_hot_encode(standardized, length)
    original_probabilities = model.predict(original[None, ...], verbose=0)[0]
    predicted_class = int(original_probabilities.argmax())
    candidates = []
    for position in range(0, length - window_size + 1, window_size):
        masked = original.copy()
        masked[position : position + window_size] = 0
        masked_probability = model.predict(masked[None, ...], verbose=0)[0, predicted_class]
        candidates.append(
            {
                "start": position,
                "end": position + window_size,
                "sequence": standardized[position : position + window_size],
                "importance": float(original_probabilities[predicted_class] - masked_probability),
            }
        )
    return sorted(candidates, key=lambda item: item["importance"], reverse=True)[:top_k]
