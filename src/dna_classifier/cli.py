"""Command-line entry point for the complete DNA classification workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import LABELS, generate_demo_dataset, load_csv
from .evaluation import save_evaluation, save_learning_curves
from .inference import ARTIFACT_VERSION, load_artifacts, predict_sequence, read_settings
from .preprocessing import encode_labels, encode_sequences, stratified_split


def set_random_seeds(seed: int) -> None:
    """Make data splitting and training as repeatable as practical."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass


def artifact_settings(artifact_dir: Path) -> dict:
    return read_settings(artifact_dir)


def command_demo_data(args: argparse.Namespace) -> None:
    frame = generate_demo_dataset(args.samples_per_class, args.length, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Created {len(frame)} balanced demo sequences at {output}")


def command_train(args: argparse.Namespace) -> None:
    if args.length < 18 or args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        raise ValueError("Use length >= 18, positive epochs/batch size, and a nonnegative seed")
    frame = load_csv(args.data)
    split = stratified_split(
        frame, args.test_size, args.validation_size, args.seed, sequence_length=args.length
    )
    artifact_dir = Path(args.artifact_dir)
    processed_dir = Path(args.processed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # A failed training run must not replace the previously usable model or splits.
    with tempfile.TemporaryDirectory(prefix="training-", dir=artifact_dir) as directory:
        staging = Path(directory)
        _train(args, split, staging)
        for name in ("train", "validation", "test"):
            getattr(split, name).to_csv(staging / f"{name}.csv", index=False)
        for path in staging.glob("*.csv"):
            shutil.copy2(path, processed_dir / path.name)
        for path in staging.iterdir():
            if path.suffix != ".csv" and path.name != "settings.json":
                path.replace(artifact_dir / path.name)
        (staging / "settings.json").replace(artifact_dir / "settings.json")
    print(f"Saved model and reports in {artifact_dir}")


def _train(args: argparse.Namespace, split, artifact_dir: Path) -> None:
    import tensorflow as tf

    from .model import build_cnn, training_callbacks

    set_random_seeds(args.seed)
    x_train = encode_sequences(split.train["sequence"], args.length)
    x_validation = encode_sequences(split.validation["sequence"], args.length)
    y_train = encode_labels(split.train["label"])
    y_validation = encode_labels(split.validation["label"])

    model_path = artifact_dir / "best_model.keras"
    model = build_cnn(args.length, len(LABELS))
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=training_callbacks(str(model_path)),
        verbose=2,
    )
    save_learning_curves(history.history, artifact_dir / "learning_curves.png")
    best_model = tf.keras.models.load_model(model_path)
    probabilities = best_model.predict(
        encode_sequences(split.test["sequence"], args.length), verbose=0
    )
    metrics = save_evaluation(
        encode_labels(split.test["label"]), probabilities, list(LABELS), artifact_dir
    )
    settings = {
        "artifact_version": ARTIFACT_VERSION,
        "sequence_length": args.length,
        "class_names": list(LABELS),
        "seed": args.seed,
        "data_source": args.data_source,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "tensorflow_version": tf.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_requested": args.epochs,
        "epochs_run": len(history.history["loss"]),
        "batch_size": args.batch_size,
        "test_fraction": args.test_size,
        "validation_fraction": args.validation_size,
        "split_counts": {
            name: len(getattr(split, name)) for name in ("train", "validation", "test")
        },
        "class_counts": {
            name: getattr(split, name)["label"].value_counts().to_dict()
            for name in ("train", "validation", "test")
        },
    }
    (artifact_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))


def command_prepare_demo(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.artifact_dir)
    try:
        settings = read_settings(artifact_dir)
        if not args.force and settings.get("artifact_version") == ARTIFACT_VERSION:
            load_artifacts(artifact_dir)
            print(f"Existing validated model is ready in {artifact_dir}")
            return
    except (OSError, ValueError, KeyError):
        pass
    data = Path(args.processed_dir).parent / "raw" / "demo_sequences.csv"
    command_demo_data(
        argparse.Namespace(output=str(data), samples_per_class=600, length=200, seed=42)
    )
    command_train(
        argparse.Namespace(
            data=str(data),
            length=200,
            epochs=30,
            batch_size=32,
            test_size=0.15,
            validation_size=0.15,
            seed=42,
            artifact_dir=args.artifact_dir,
            processed_dir=args.processed_dir,
            data_source="synthetic planted motifs (educational)",
        )
    )


def command_evaluate(args: argparse.Namespace) -> None:
    artifact_dir = Path(args.artifact_dir)
    model, settings = load_artifacts(artifact_dir)
    frame = load_csv(args.data)
    probabilities = model.predict(
        encode_sequences(frame["sequence"], settings["sequence_length"]), verbose=0
    )
    metrics = save_evaluation(
        encode_labels(frame["label"], settings["class_names"]),
        probabilities,
        settings["class_names"],
        Path(args.output_dir) if args.output_dir else artifact_dir / "evaluation",
    )
    print(json.dumps(metrics, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    from .interpretation import influential_windows

    artifact_dir = Path(args.artifact_dir)
    model, settings = load_artifacts(artifact_dir)
    result = predict_sequence(model, settings, args.sequence)
    print(f"Prediction: {result['prediction']}")
    print(f"Known model-input bases: {result['known_bases']}/{settings['sequence_length']}")
    for name, probability in result["probabilities"].items():
        print(f"  {name:10s} {probability:.2%}")
    print("Most influential windows (0-based positions):")
    for window in influential_windows(model, args.sequence, settings["sequence_length"]):
        print(
            f"  {window['start']:3d}-{window['end']:3d}  {window['sequence']}  "
            f"score={window['importance']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify promoter, enhancer, and background DNA")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo-data", help="create a balanced synthetic teaching dataset")
    demo.add_argument("--output", default="data/raw/demo_sequences.csv")
    demo.add_argument("--samples-per-class", type=int, default=300)
    demo.add_argument("--length", type=int, default=200)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(function=command_demo_data)

    train = subparsers.add_parser("train", help="split data, train the CNN, and evaluate it")
    train.add_argument("--data", required=True, help="CSV with sequence and label columns")
    train.add_argument("--length", type=int, default=200)
    train.add_argument("--epochs", type=int, default=15)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--validation-size", type=float, default=0.15)
    train.add_argument("--test-size", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--artifact-dir", default="artifacts")
    train.add_argument("--processed-dir", default="data/processed")
    train.add_argument("--data-source", default="user-provided (unverified)")
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate the saved model on labeled CSV data"
    )
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--artifact-dir", default="artifacts")
    evaluate.add_argument("--output-dir", help="report folder; defaults to artifacts/evaluation")
    evaluate.set_defaults(function=command_evaluate)

    predict = subparsers.add_parser("predict", help="classify one DNA sequence")
    predict.add_argument("--sequence", required=True)
    predict.add_argument("--artifact-dir", default="artifacts")
    predict.set_defaults(function=command_predict)
    prepare = subparsers.add_parser(
        "prepare-demo", help="prepare or reuse the local interview model"
    )
    prepare.add_argument("--artifact-dir", default="artifacts")
    prepare.add_argument("--processed-dir", default="data/processed")
    prepare.add_argument(
        "--force", action="store_true", help="replace the model with a freshly trained demo"
    )
    prepare.set_defaults(function=command_prepare_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.function(args)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
