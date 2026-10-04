"""Command-line entry point for the complete DNA classification workflow."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from .data import LABELS, generate_demo_dataset, load_csv
from .evaluation import save_evaluation, save_learning_curves
from .preprocessing import encode_labels, encode_sequences, stratified_split


def set_random_seeds(seed: int) -> None:
    """Make data splitting and training as repeatable as practical."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def artifact_settings(artifact_dir: Path) -> dict:
    settings_path = artifact_dir / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"Missing {settings_path}. Run the train command first.")
    return json.loads(settings_path.read_text(encoding="utf-8"))


def command_demo_data(args: argparse.Namespace) -> None:
    frame = generate_demo_dataset(args.samples_per_class, args.length, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Created {len(frame)} balanced demo sequences at {output}")


def command_train(args: argparse.Namespace) -> None:
    import tensorflow as tf

    from .model import build_cnn, training_callbacks

    set_random_seeds(args.seed)
    artifact_dir = Path(args.artifact_dir)
    processed_dir = Path(args.processed_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    frame = load_csv(args.data)
    split = stratified_split(frame, args.test_size, args.validation_size, args.seed)
    for name, part in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        part.to_csv(processed_dir / f"{name}.csv", index=False)

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
    settings = {"sequence_length": args.length, "class_names": list(LABELS), "seed": args.seed}
    (artifact_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    best_model = tf.keras.models.load_model(model_path)
    probabilities = best_model.predict(
        encode_sequences(split.test["sequence"], args.length), verbose=0
    )
    metrics = save_evaluation(
        encode_labels(split.test["label"]), probabilities, list(LABELS), artifact_dir
    )
    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model and reports in {artifact_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    import tensorflow as tf

    artifact_dir = Path(args.artifact_dir)
    settings = artifact_settings(artifact_dir)
    frame = load_csv(args.data)
    model = tf.keras.models.load_model(artifact_dir / "best_model.keras")
    probabilities = model.predict(
        encode_sequences(frame["sequence"], settings["sequence_length"]), verbose=0
    )
    metrics = save_evaluation(
        encode_labels(frame["label"]), probabilities, settings["class_names"], artifact_dir
    )
    print(json.dumps(metrics, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    import tensorflow as tf

    from .interpretation import influential_windows

    artifact_dir = Path(args.artifact_dir)
    settings = artifact_settings(artifact_dir)
    model = tf.keras.models.load_model(artifact_dir / "best_model.keras")
    encoded = encode_sequences([args.sequence], settings["sequence_length"])
    probabilities = model.predict(encoded, verbose=0)[0]
    best_index = int(probabilities.argmax())
    print(f"Prediction: {settings['class_names'][best_index]}")
    for name, probability in zip(settings["class_names"], probabilities):
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
    train.set_defaults(function=command_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate the saved model on labeled CSV data"
    )
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--artifact-dir", default="artifacts")
    evaluate.set_defaults(function=command_evaluate)

    predict = subparsers.add_parser("predict", help="classify one DNA sequence")
    predict.add_argument("--sequence", required=True)
    predict.add_argument("--artifact-dir", default="artifacts")
    predict.set_defaults(function=command_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
