#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Creating the Python environment..."
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --quiet --upgrade pip
.venv/bin/python -m pip install --disable-pip-version-check --quiet -r requirements.txt
.venv/bin/python -m pip install --disable-pip-version-check --quiet -e .

if [[ ! -f artifacts/best_model.keras || ! -f artifacts/settings.json ]]; then
  echo "Preparing the demo model (first launch only)..."
  .venv/bin/python -m dna_classifier.cli demo-data \
    --output data/raw/demo_sequences.csv \
    --samples-per-class 120
  .venv/bin/python -m dna_classifier.cli train \
    --data data/raw/demo_sequences.csv \
    --epochs 8
fi

echo "Codespace setup complete."
