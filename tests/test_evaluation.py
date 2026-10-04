import numpy as np

from dna_classifier.evaluation import calculate_metrics


def test_perfect_predictions_have_perfect_metrics():
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    metrics = calculate_metrics(labels, probabilities, ["promoter", "enhancer", "background"])
    assert all(value == 1.0 for value in metrics.values())
