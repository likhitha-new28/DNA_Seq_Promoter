"""Exercise the actual Streamlit page without training or downloading a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import dna_classifier.inference

APP = Path(__file__).resolve().parents[1] / "src/dna_classifier/webapp.py"


class DemoModel:
    def predict(self, batch, verbose=0):
        return np.tile([0.8, 0.1, 0.1], (len(batch), 1))


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "best_model.keras").write_bytes(b"fixture")
    settings = {
        "sequence_length": 200,
        "class_names": ["promoter", "enhancer", "background"],
        "seed": 42,
        "data_source": "test fixture",
        "model_sha256": hashlib.sha256(b"fixture").hexdigest(),
    }
    (artifacts / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setattr(
        dna_classifier.inference, "load_artifacts", lambda folder: (DemoModel(), settings)
    )
    # Override only the artifact directory; execute the full, unchanged page implementation.
    source = APP.read_text(encoding="utf-8").replace(
        'ARTIFACT_DIR = PROJECT_ROOT / "artifacts"', f"ARTIFACT_DIR = Path({str(artifacts)!r})"
    )
    page = AppTest.from_string(source, default_timeout=15).run()
    yield page
    st.cache_resource.clear()


def test_prediction_and_example_selection(app):
    assert not app.exception
    assert len(app.text_area[0].value) == 200
    original = app.text_area[0].value
    app.button[0].click().run()
    assert not app.exception
    assert any("Predicted class: Promoter" in value.value for value in app.subheader)
    app.radio[0].set_value("Enhancer motif example").run()
    assert app.text_area[0].value != original
    assert not any("Predicted class:" in value.value for value in app.subheader)


def test_invalid_sequence_is_an_error_and_clears_old_result(app):
    app.button[0].click().run()
    app.text_area[0].set_value("ACGX").run()
    app.button[0].click().run()
    assert not app.exception
    assert "Invalid DNA" in app.error[0].value
    assert not any("Predicted class:" in value.value for value in app.subheader)


def test_padding_warning_and_result_persist_on_rerun(app):
    app.text_area[0].set_value("ACGT").run()
    app.button[0].click().run()
    assert not app.exception
    assert "Only 4 of 200" in app.warning[0].value
    app.run()
    assert any("Predicted class:" in value.value for value in app.subheader)


def test_unknown_only_sequence_is_not_classified(app):
    app.text_area[0].set_value("N" * 200).run()
    app.button[0].click().run()
    assert not app.exception
    assert "no known" in app.error[0].value
