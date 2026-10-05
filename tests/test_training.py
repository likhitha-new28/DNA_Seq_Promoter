"""Small real TensorFlow round trip; accuracy is not a smoke-test assertion."""

import argparse
import hashlib
import json

import pytest

from dna_classifier import cli
from dna_classifier.data import generate_demo_dataset
from dna_classifier.inference import load_artifacts, predict_sequence


def training_args(tmp_path):
    data = tmp_path / "data.csv"
    generate_demo_dataset(12, 30).to_csv(data, index=False)
    return argparse.Namespace(
        data=str(data),
        length=30,
        epochs=1,
        batch_size=8,
        seed=42,
        test_size=0.2,
        validation_size=0.2,
        artifact_dir=str(tmp_path / "artifacts"),
        processed_dir=str(tmp_path / "processed"),
        data_source="synthetic test fixture",
    )


def test_train_load_predict_evaluate_and_reuse(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    cli.command_train(args)
    artifacts = tmp_path / "artifacts"
    original_metrics = (artifacts / "metrics.json").read_bytes()
    model, settings = load_artifacts(artifacts)
    result = predict_sequence(model, settings, "ACGT" * 8)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert len(result["model_sequence"]) == 30
    assert settings["split_counts"] == {"train": 21, "validation": 7, "test": 8}
    assert (
        settings["model_sha256"]
        == hashlib.sha256((artifacts / "best_model.keras").read_bytes()).hexdigest()
    )
    cli.command_evaluate(
        argparse.Namespace(
            data=str(tmp_path / "processed/test.csv"),
            artifact_dir=str(artifacts),
            output_dir=None,
        )
    )
    assert (artifacts / "metrics.json").read_bytes() == original_metrics
    assert (artifacts / "evaluation/classification_report.json").exists()
    assert (
        json.loads((artifacts / "evaluation/confusion_matrix.json").read_text())["class_names"]
        == settings["class_names"]
    )
    monkeypatch.setattr(cli, "command_train", lambda args: pytest.fail("Valid model was retrained"))
    cli.command_prepare_demo(
        argparse.Namespace(
            artifact_dir=str(artifacts),
            processed_dir=str(tmp_path / "processed"),
            force=False,
        )
    )


def test_failed_training_preserves_previous_artifacts(tmp_path, monkeypatch):
    args = training_args(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    model = artifacts / "best_model.keras"
    model.write_bytes(b"previous working model")
    settings = artifacts / "settings.json"
    settings.write_text('{"previous": true}')

    def fail_fit(*args):
        raise RuntimeError("simulated interrupted fit")

    monkeypatch.setattr(cli, "_train", fail_fit)
    with pytest.raises(RuntimeError, match="interrupted fit"):
        cli.command_train(args)
    assert model.read_bytes() == b"previous working model"
    assert settings.read_text() == '{"previous": true}'
    assert not list(artifacts.glob("training-*"))
