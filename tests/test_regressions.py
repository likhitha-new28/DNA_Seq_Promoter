"""Regression coverage for invalid input, leakage and incomplete explanations."""

import json

import numpy as np
import pandas as pd
import pytest

from dna_classifier.data import clean_sequence, generate_demo_dataset, load_csv, load_fasta
from dna_classifier.evaluation import calculate_metrics
from dna_classifier.inference import predict_sequence, read_settings
from dna_classifier.interpretation import influential_windows
from dna_classifier.preprocessing import encode_sequences, stratified_split


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.seen = []

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.seen.append(batch.copy())
        score = batch.sum(axis=(1, 2)) / batch.shape[1] * 0.8
        return np.column_stack([score, (1 - score) / 2, (1 - score) / 2])


@pytest.mark.parametrize("value", [None, 123, np.nan, "", " \n "])
def test_missing_or_nontext_sequence_is_rejected(value):
    with pytest.raises(ValueError):
        clean_sequence(value)


@pytest.mark.parametrize("body", ["sequence,label\n", "sequence,label\n,promoter\n"])
def test_csv_does_not_silently_drop_invalid_rows(tmp_path, body):
    path = tmp_path / "bad.csv"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_csv(path)


def test_valid_dna_that_looks_like_pandas_missing_value_is_preserved(tmp_path):
    path = tmp_path / "valid.csv"
    path.write_text("sequence,label\nNA,promoter\nNAN,enhancer\n")
    assert load_csv(path).sequence.tolist() == ["NA", "NAN"]


@pytest.mark.parametrize(
    "settings",
    [
        [],
        None,
        {"sequence_length": 200, "class_names": None},
        {"sequence_length": 200, "class_names": [["promoter"]]},
    ],
)
def test_malformed_settings_have_actionable_errors(tmp_path, settings):
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        read_settings(tmp_path)


@pytest.mark.parametrize("body", ["ACGT\n", ">one\n>two\nACGT\n", ">one\nACGT\n>two\n"])
def test_fasta_requires_nonempty_records_and_headers(tmp_path, body):
    path = tmp_path / "bad.fa"
    path.write_text(body)
    with pytest.raises(ValueError):
        load_fasta(path, "promoter")


def test_multiline_fasta(tmp_path):
    path = tmp_path / "valid.fa"
    path.write_text(">one\nacgt\nAC\n>two\nGGTT\n")
    assert load_fasta(path, "promoter").sequence.tolist() == ["ACGTAC", "GGTT"]


def test_empty_encoding_is_actionable():
    with pytest.raises(ValueError, match="At least one"):
        encode_sequences([], 20)


def test_duplicates_after_cropping_cannot_leak_across_splits():
    frame = generate_demo_dataset(20, 30)
    original = frame.iloc[0]
    duplicate = pd.DataFrame([{"sequence": "A" + original.sequence + "T", "label": original.label}])
    split = stratified_split(pd.concat([frame, duplicate]), 0.2, 0.2, sequence_length=30)
    parts = [set(getattr(split, name).sequence) for name in ("train", "validation", "test")]
    assert sum(map(len, parts)) == 60
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2])
    assert parts[1].isdisjoint(parts[2])


def test_conflicting_duplicate_labels_are_rejected():
    frame = generate_demo_dataset(20, 30)
    duplicate = frame.iloc[[0]].copy()
    duplicate["label"] = "enhancer" if duplicate.iloc[0].label == "promoter" else "promoter"
    with pytest.raises(ValueError, match="conflicting labels"):
        stratified_split(pd.concat([frame, duplicate]))


def test_small_dataset_has_helpful_split_error():
    with pytest.raises(ValueError, match="Not enough unique"):
        stratified_split(generate_demo_dataset(2, 30))


def test_occlusion_includes_tail_and_batches_model_calls():
    model = CountingModel()
    windows = influential_windows(model, "A" * 23, 23, window_size=10, top_k=3)
    assert {(row["start"], row["end"]) for row in windows} == {(0, 10), (10, 20), (20, 23)}
    assert model.calls == 2
    assert model.seen[1].shape == (3, 23, 4)
    assert all(row["importance"] > 0 for row in windows)


@pytest.mark.parametrize("window,top", [(0, 5), (10, 0)])
def test_occlusion_rejects_invalid_parameters(window, top):
    with pytest.raises(ValueError):
        influential_windows(CountingModel(), "A" * 20, 20, window, top)


def test_prediction_reports_padding_and_rejects_unknown_only_input():
    settings = {"sequence_length": 20, "class_names": ["promoter", "enhancer", "background"]}
    result = predict_sequence(CountingModel(), settings, "acgt")
    assert result["known_bases"] == 4
    assert result["padding_left"] == 8
    assert result["model_sequence"] == "N" * 8 + "ACGT" + "N" * 8
    with pytest.raises(ValueError, match="no known"):
        predict_sequence(CountingModel(), settings, "N" * 20)


def test_crop_removes_all_known_bases_is_rejected():
    settings = {"sequence_length": 20, "class_names": ["promoter", "enhancer", "background"]}
    with pytest.raises(ValueError, match="no known"):
        predict_sequence(CountingModel(), settings, "A" + "N" * 40 + "C")


def test_artifact_hash_mismatch_is_rejected(tmp_path):
    (tmp_path / "best_model.keras").write_bytes(b"wrong model")
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "sequence_length": 200,
                "class_names": ["promoter", "enhancer", "background"],
                "model_sha256": "incorrect",
            }
        )
    )
    with pytest.raises(ValueError, match="do not match"):
        read_settings(tmp_path)


def test_missing_class_auc_is_unavailable_and_macro_includes_all_classes():
    metrics = calculate_metrics(
        np.array([0, 0]),
        np.array([[1.0, 0, 0], [1.0, 0, 0]]),
        ["promoter", "enhancer", "background"],
    )
    assert metrics["accuracy"] == 1
    assert metrics["recall_macro"] == pytest.approx(1 / 3)
    assert metrics["roc_auc_ovr_macro"] is None


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.1, 0.1, 0.1]],
        [[np.nan, 0, 1]],
        [[-1, 1, 1]],
        [[1, 0]],
    ],
)
def test_invalid_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError, match="valid class probabilities"):
        calculate_metrics(np.array([0]), np.array(probabilities), ["p", "e", "b"])


def test_binary_auc_is_supported():
    metrics = calculate_metrics(np.array([0, 1]), np.array([[1.0, 0], [0, 1.0]]), ["a", "b"])
    assert metrics["roc_auc_ovr_macro"] == 1
