# DNA Promoter / Enhancer Classifier

A beginner-friendly deep-learning project that classifies fixed-length DNA sequences as
**promoter**, **enhancer**, or **background**. It includes data preparation, a 1D convolutional
neural network (CNN), evaluation, prediction, and simple model interpretation.

> **Educational project:** the included demo data contains planted motifs. Results from it are a
> software check, not biological evidence. Use curated genomic data and careful experimental
> validation before drawing scientific conclusions.

## What the pipeline does

1. Loads labeled sequences from CSV, or creates a balanced demo dataset.
2. Cleans bases, centre-crops long sequences, and pads short sequences with `N`.
3. Makes stratified train/validation/test splits.
4. One-hot encodes bases in `A, C, G, T` channels (`N` becomes all zeros).
5. Trains a compact 1D CNN to recognize local sequence patterns.
6. Reports accuracy, precision, recall, F1, multiclass ROC-AUC, and a confusion matrix.
7. Finds influential sequence windows using a simple occlusion analysis.

## Quick start

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

dna-classifier demo-data --output data/raw/demo_sequences.csv
dna-classifier train --data data/raw/demo_sequences.csv --epochs 15
dna-classifier evaluate --data data/processed/test.csv
dna-classifier predict --sequence ACGTTATAAAGCTACGTACGTACGT
```

All commands also work as `python -m dna_classifier.cli ...`. Training writes the fitted model,
class names, run settings, learning curves, and test splits under `artifacts/` and `data/processed/`.

## Input data

Provide a CSV with exactly the following logical fields (extra columns are ignored):

```csv
sequence,label
ACGTTATAAAGCTACGT,promoter
TTGCCACGTGAAAATC,enhancer
GCTAGCTAGCTAGCTA,background
```

Labels must be `promoter`, `enhancer`, or `background`. Sequences may contain `A`, `C`, `G`, `T`,
and `N` (case-insensitive). See [docs/data-guide.md](docs/data-guide.md) for responsible sources and
conversion advice.

## Repository layout

```text
src/dna_classifier/   Python package and CLI
tests/                Fast unit tests
docs/                 Data and methodology notes
data/                 Local raw/processed data (ignored by Git)
artifacts/             Models, metrics, and plots (ignored by Git)
```

## Reproducibility

The default random seed is `42`. Exact deep-learning results can still differ slightly by hardware
and TensorFlow version. Run `pytest` for unit tests and `ruff check .` for lint checks.

## License

MIT
