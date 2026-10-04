"""Streamlit web interface for local DNA sequence predictions."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dna_classifier.data import clean_sequence
from dna_classifier.interpretation import influential_windows
from dna_classifier.preprocessing import encode_sequences

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@st.cache_resource
def load_model_and_settings():
    """Load the trained model once and reuse it between page interactions."""
    import tensorflow as tf

    settings = json.loads((ARTIFACT_DIR / "settings.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(ARTIFACT_DIR / "best_model.keras")
    return model, settings


def render_page() -> None:
    st.set_page_config(page_title="DNA Regulatory Classifier", page_icon="🧬", layout="centered")
    st.title("🧬 DNA Regulatory Classifier")
    st.write(
        "Enter a DNA sequence to estimate whether it resembles a promoter, enhancer, "
        "or background sequence."
    )
    st.info(
        "This is an educational model trained on synthetic motif data. Its results are not "
        "clinical or experimental evidence."
    )

    required_artifacts = (ARTIFACT_DIR / "best_model.keras", ARTIFACT_DIR / "settings.json")
    if not all(path.exists() for path in required_artifacts):
        st.error("No model was found. Close this page and run the local launcher again.")
        st.stop()

    model, settings = load_model_and_settings()
    example = "GCGCGCGCTATAAAGCTACGTACGTTAGCGCGCGC"
    sequence = st.text_area(
        "DNA sequence (A, C, G, T, or N)",
        value=example,
        height=140,
        help=f"Sequences are cropped or padded to {settings['sequence_length']} bases.",
    )

    if st.button("Classify sequence", type="primary", use_container_width=True):
        try:
            cleaned = clean_sequence(sequence)
        except ValueError as error:
            st.error(str(error))
            return

        encoded = encode_sequences([cleaned], settings["sequence_length"])
        probabilities = model.predict(encoded, verbose=0)[0]
        best_index = int(probabilities.argmax())
        best_label = settings["class_names"][best_index]
        st.success(f"Prediction: **{best_label.title()}** ({probabilities[best_index]:.1%})")

        st.subheader("Class probabilities")
        probability_rows = {
            name.title(): float(probability)
            for name, probability in zip(settings["class_names"], probabilities)
        }
        st.bar_chart(probability_rows, horizontal=True)

        st.subheader("Influential sequence windows")
        st.caption(
            "Masking these windows caused the largest drop in the predicted-class probability."
        )
        windows = influential_windows(model, cleaned, settings["sequence_length"])
        st.dataframe(windows, use_container_width=True, hide_index=True)

    with st.sidebar:
        st.header("About")
        st.write(f"Model input length: {settings['sequence_length']} bases")
        st.write("Classes: " + ", ".join(settings["class_names"]))
        metrics_path = ARTIFACT_DIR / "metrics.json"
        if metrics_path.exists():
            st.subheader("Demo test metrics")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for name, value in metrics.items():
                if value is not None:
                    st.metric(name.replace("_", " ").title(), f"{value:.3f}")


if __name__ == "__main__":
    render_page()
