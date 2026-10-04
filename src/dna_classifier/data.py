"""Dataset loading, validation, standardisation, and demo-data generation."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

VALID_BASES = set("ACGTN")
LABELS = ("promoter", "enhancer", "background")


def clean_sequence(sequence: str) -> str:
    """Return an uppercase DNA sequence and reject unsupported characters."""
    cleaned = "".join(str(sequence).upper().split())
    invalid = set(cleaned) - VALID_BASES
    if invalid:
        raise ValueError(f"Invalid DNA base(s): {', '.join(sorted(invalid))}")
    if not cleaned:
        raise ValueError("DNA sequence cannot be empty")
    return cleaned


def standardize_sequence(sequence: str, length: int) -> str:
    """Centre-crop long sequences and pad short ones equally with N bases."""
    sequence = clean_sequence(sequence)
    if len(sequence) >= length:
        start = (len(sequence) - length) // 2
        return sequence[start : start + length]
    missing = length - len(sequence)
    left = missing // 2
    return "N" * left + sequence + "N" * (missing - left)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV containing sequence and label columns and validate it."""
    frame = pd.read_csv(path)
    required = {"sequence", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain 'sequence' and 'label' columns")
    frame = frame[["sequence", "label"]].dropna().copy()
    frame["sequence"] = frame["sequence"].map(clean_sequence)
    frame["label"] = frame["label"].str.lower().str.strip()
    unknown = set(frame["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown label(s): {', '.join(sorted(unknown))}")
    return frame


def load_fasta(path: str | Path, label: str) -> pd.DataFrame:
    """Read a basic FASTA file and attach one label to every record."""
    if label not in LABELS:
        raise ValueError(f"Label must be one of: {', '.join(LABELS)}")
    records: list[str] = []
    current: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(">"):
            if current:
                records.append(clean_sequence("".join(current)))
                current = []
        elif line:
            current.append(line)
    if current:
        records.append(clean_sequence("".join(current)))
    if not records:
        raise ValueError("FASTA file contains no sequences")
    return pd.DataFrame({"sequence": records, "label": label})


def generate_demo_dataset(samples_per_class: int = 300, length: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a balanced teaching dataset with planted regulatory motifs.

    This data demonstrates the pipeline; it is not suitable for biological claims.
    """
    if samples_per_class < 2 or length < 20:
        raise ValueError("Use at least 2 samples per class and sequence length >= 20")
    rng = random.Random(seed)
    motifs = {"promoter": "TATAAA", "enhancer": "CACGTG"}
    rows: list[dict[str, str]] = []
    for label in LABELS:
        for _ in range(samples_per_class):
            bases = [rng.choice("ACGT") for _ in range(length)]
            if label in motifs:
                motif = motifs[label]
                position = rng.randint(length // 4, 3 * length // 4 - len(motif))
                bases[position : position + len(motif)] = motif
            rows.append({"sequence": "".join(bases), "label": label})
    rng.shuffle(rows)
    return pd.DataFrame(rows)

