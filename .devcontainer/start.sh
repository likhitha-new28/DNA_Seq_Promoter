#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Avoid starting a second server when a Codespace is resumed more than once.
if pgrep -f "streamlit run src/dna_classifier/webapp.py" >/dev/null; then
  echo "DNA Classifier is already running."
  exit 0
fi

echo "Starting DNA Classifier on port 8501..."
nohup .venv/bin/python -m streamlit run src/dna_classifier/webapp.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --browser.gatherUsageStats false \
  > /tmp/dna-classifier.log 2>&1 &
