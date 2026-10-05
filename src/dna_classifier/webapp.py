"""Local interview workbench for DNA predictions and held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dna_classifier.data import generate_demo_dataset
from dna_classifier.inference import load_artifacts, predict_sequence
from dna_classifier.interpretation import influential_windows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


@st.cache_resource
def load_model_and_settings(signature: tuple):
    """Reload when model or settings change, including retraining during a session."""
    return load_artifacts(ARTIFACT_DIR)


def artifact_signature() -> tuple:
    return tuple(
        (path.stat().st_mtime_ns, path.stat().st_size)
        for path in (ARTIFACT_DIR / "best_model.keras", ARTIFACT_DIR / "settings.json")
    )


def demo_examples(length: int) -> dict[str, str]:
    # Independently seeded examples are generated without selecting for model success.
    frame = generate_demo_dataset(samples_per_class=2, length=max(20, length), seed=2026)
    return {
        "Promoter motif example": frame.loc[frame.label == "promoter", "sequence"].iloc[0],
        "Enhancer motif example": frame.loc[frame.label == "enhancer", "sequence"].iloc[0],
        "Background control": frame.loc[frame.label == "background", "sequence"].iloc[0],
    }


def choose_example():
    selection = st.session_state["example_choice"]
    if selection in st.session_state["examples"]:
        st.session_state["sequence_input"] = st.session_state["examples"][selection]
    st.session_state.pop("prediction_result", None)


def render_prediction(model, settings: dict) -> None:
    length = settings["sequence_length"]
    examples = demo_examples(length)
    st.session_state["examples"] = examples
    st.session_state.setdefault("sequence_input", next(iter(examples.values())))
    st.radio(
        "Start with an example",
        [*examples, "Custom sequence"],
        key="example_choice",
        on_change=choose_example,
        horizontal=True,
        help="Generated with seed 2026, independently of training. Labels describe planted motifs.",
    )
    sequence = st.text_area(
        "DNA sequence",
        key="sequence_input",
        height=150,
        help="Paste A, C, G, T or N bases. Spaces and line breaks are accepted; FASTA headers are not.",
    )
    st.caption(f"Model input: {length} bases · A/C/G/T channels · N = unknown")
    if st.button("Classify sequence", type="primary", width="stretch"):
        st.session_state.pop("prediction_result", None)
        try:
            with st.spinner("Predicting and testing sequence windows…"):
                result = predict_sequence(model, settings, sequence)
                result["windows"] = influential_windows(model, sequence, length)
                result["submitted_sequence"] = sequence
                st.session_state["prediction_result"] = result
        except (ValueError, OSError) as error:
            st.error(str(error))
    result = st.session_state.get("prediction_result")
    if result is None or result["submitted_sequence"] != sequence:
        return
    label = result["prediction"]
    probability = result["probabilities"][label]
    st.subheader(f"Predicted class: {label.title()}")
    st.caption(
        f"Model score: {probability:.1%}. Softmax scores are not calibrated biological certainty."
    )
    if result["known_bases"] < length:
        st.warning(
            f"Only {result['known_bases']} of {length} model-input bases are known. "
            "Padding and ambiguous N bases can make this unlike the training examples."
        )
    if result["input_length"] > length:
        start = result["crop_start"]
        st.warning(
            f"Centre crop: only input bases [{start}, {start + length}) were used (0-based)."
        )
    elif result["input_length"] < length:
        left = result["padding_left"]
        right = length - result["input_length"] - left
        st.caption(f"Added {left} N bases on the left and {right} on the right.")
    chart = pd.DataFrame.from_dict(result["probabilities"], orient="index", columns=["Model score"])
    st.bar_chart(chart, horizontal=True)
    st.subheader("Which windows supported this prediction?")
    st.caption(
        "We replace each window with N and measure the change in this class's score. "
        "Positive values support the prediction; negative values oppose it. "
        "Coordinates are 0-based, end-exclusive, in the cropped/padded model input. "
        "This is a sensitivity check, not a validated regulatory annotation."
    )
    st.dataframe(result["windows"], hide_index=True, width="stretch")
    with st.expander("See the exact model input"):
        st.code(result["model_sequence"], language=None)
    export = {key: value for key, value in result.items() if key != "submitted_sequence"}
    export["model"] = settings
    st.download_button(
        "Download prediction report",
        json.dumps(export, indent=2),
        file_name="dna-prediction.json",
        mime="application/json",
        on_click="ignore",
    )


def render_evaluation(settings: dict) -> None:
    st.subheader("Held-out test results")
    st.write(
        "Training fits the weights; validation chooses the checkpoint; the held-out test "
        "set measures its performance once. These numbers describe this dataset only."
    )
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        st.info("Evaluation is unavailable. Run prepare-demo or evaluate to generate reports.")
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        columns = st.columns(3)
        for column, (label, key) in zip(
            columns,
            [
                ("Test accuracy", "accuracy"),
                ("Macro F1", "f1_macro"),
                ("Macro ROC-AUC", "roc_auc_ovr_macro"),
            ],
        ):
            value = metrics.get(key)
            column.metric(label, "Unavailable" if value is None else f"{value:.3f}")
        st.caption("Balanced three-class chance reference: approximately 0.333 accuracy.")
        report_path = ARTIFACT_DIR / "classification_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            st.dataframe(
                pd.DataFrame({name: report[name] for name in settings["class_names"]}).T,
                width="stretch",
            )
        matrix_path = ARTIFACT_DIR / "confusion_matrix.json"
        if matrix_path.exists():
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
            st.caption("Confusion matrix: rows = true class; columns = predicted class.")
            st.dataframe(
                pd.DataFrame(
                    matrix["matrix"], index=matrix["class_names"], columns=matrix["class_names"]
                ),
                width="stretch",
            )
        curve = ARTIFACT_DIR / "learning_curves.png"
        if curve.exists():
            st.image(str(curve), caption="Training and validation curves")
        st.subheader("Run provenance")
        st.json(settings, expanded=False)
        st.download_button(
            "Download evaluation summary",
            json.dumps({"metrics": metrics, "run": settings}, indent=2),
            file_name="dna-evaluation.json",
            mime="application/json",
            on_click="ignore",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Evaluation report could not be read: {error}. Regenerate the reports.")


def render_methodology() -> None:
    st.subheader("Sequence → representation → CNN → scores → sensitivity check")
    st.write(
        "A promoter is associated with transcription initiation; an enhancer can regulate "
        "transcription at a distance. Here they are teaching labels: promoter examples have "
        "a planted TATAAA motif, enhancer examples have CACGTG, and background examples "
        "are random DNA. Random sequences can contain either motif by chance. Real promoters "
        "and enhancers are more diverse and can have overlapping functions."
    )
    st.markdown(
        "1. **Prepare:** validate bases, centre-crop/pad and remove identical model inputs.\n"
        "2. **Split:** stratified train/validation/test partitions, using seed 42 for the demo.\n"
        "3. **Encode:** one row per base, four channels ordered A, C, G, T.\n"
        "4. **Learn:** two 1D convolutions, pooling, dropout and a three-class softmax.\n"
        "5. **Evaluate:** per-class precision/recall, macro F1, ROC-AUC and a confusion matrix.\n"
        "6. **Inspect:** mask windows and rank the changes in the predicted-class score."
    )
    st.subheader("What would make this a biological study?")
    st.write(
        "Use curated genomic annotations with a recorded genome assembly, cell type and "
        "source release. Match background GC content and length. Remove overlaps and related "
        "sequences, use chromosome-based or independent external test sets, compare with simple "
        "baselines and repeat across seeds. This project removes exact input duplicates; it "
        "does not automatically remove homologous, reverse-complement or overlapping loci. "
        "Scores and occlusion windows need experimental validation before functional claims."
    )


def render_page() -> None:
    st.set_page_config(page_title="DNA Regulatory Workbench", page_icon="🧬", layout="wide")
    st.title("DNA Regulatory Workbench")
    st.write("Explore promoter, enhancer and background predictions with a compact 1D CNN.")
    try:
        signature = artifact_signature()
        model, settings = load_model_and_settings(signature)
    except (OSError, ValueError, ImportError, KeyError, TypeError, RuntimeError) as error:
        st.error(f"The model could not be loaded: {error}")
        st.code("python -m dna_classifier.cli prepare-demo")
        st.stop()
    if st.session_state.get("model_signature") != signature:
        st.session_state.pop("prediction_result", None)
        st.session_state["model_signature"] = signature
    source = settings.get("data_source", "unknown (legacy model)")
    st.info(
        f"Training source: {source}. Educational prototype; biological validity is unestablished."
    )
    with st.sidebar:
        st.header("Demo checklist")
        st.write(
            "1. Compare the three full-length examples.\n"
            "2. Inspect influential windows.\n3. Review held-out metrics."
        )
        st.divider()
        st.write(f"Input length: **{settings['sequence_length']} bases**")
        st.write(f"Training seed: **{settings.get('seed', 'unrecorded')}**")
        st.caption("Runs locally. No external prediction API is called.")
    prediction, evaluation, methodology = st.tabs(
        ["Predict & explain", "Evaluation", "Method & limitations"]
    )
    with prediction:
        render_prediction(model, settings)
    with evaluation:
        render_evaluation(settings)
    with methodology:
        render_methodology()


if __name__ == "__main__":
    render_page()
