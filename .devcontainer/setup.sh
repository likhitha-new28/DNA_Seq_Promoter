#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Creating the Python environment..."
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --quiet --upgrade pip
.venv/bin/python -m pip install --disable-pip-version-check --quiet -r requirements.txt
.venv/bin/python -m pip install --disable-pip-version-check --quiet -e .

.venv/bin/python -m dna_classifier.cli prepare-demo

echo "Codespace setup complete."
