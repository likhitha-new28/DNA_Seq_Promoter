import pandas as pd
import pytest

from dna_classifier.data import (
    clean_sequence,
    generate_demo_dataset,
    load_csv,
    standardize_sequence,
)


def test_clean_sequence_accepts_case_and_whitespace():
    assert clean_sequence(" acg\ntN ") == "ACGTN"


def test_clean_sequence_rejects_invalid_bases():
    with pytest.raises(ValueError, match="Invalid DNA"):
        clean_sequence("ACGX")


def test_standardize_sequence_pads_and_crops_from_centre():
    assert standardize_sequence("AC", 6) == "NNACNN"
    assert standardize_sequence("AACCGG", 4) == "ACCG"


def test_demo_dataset_is_balanced_and_reproducible():
    first = generate_demo_dataset(samples_per_class=4, length=30, seed=7)
    second = generate_demo_dataset(samples_per_class=4, length=30, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["label"].value_counts().to_dict() == {
        "promoter": 4,
        "enhancer": 4,
        "background": 4,
    }


def test_load_csv_validates_columns_and_labels(tmp_path):
    valid = tmp_path / "valid.csv"
    valid.write_text("sequence,label\nacgt,promoter\n", encoding="utf-8")
    assert load_csv(valid).iloc[0].to_dict() == {"sequence": "ACGT", "label": "promoter"}

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("sequence,label\nACGT,unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown label"):
        load_csv(invalid)
