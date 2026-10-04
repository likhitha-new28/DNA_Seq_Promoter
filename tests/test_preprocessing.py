import numpy as np

from dna_classifier.data import generate_demo_dataset
from dna_classifier.preprocessing import encode_labels, one_hot_encode, stratified_split


def test_one_hot_encoding_and_unknown_base():
    encoded = one_hot_encode("ACGTN", 5)
    np.testing.assert_array_equal(
        encoded,
        np.asarray(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
            dtype=np.float32,
        ),
    )


def test_label_order_is_stable():
    np.testing.assert_array_equal(encode_labels(["promoter", "enhancer", "background"]), [0, 1, 2])


def test_split_sizes_and_class_balance():
    frame = generate_demo_dataset(samples_per_class=20, length=30)
    split = stratified_split(frame, test_size=0.2, validation_size=0.2)
    assert (len(split.train), len(split.validation), len(split.test)) == (36, 12, 12)
    for part in (split.train, split.validation, split.test):
        assert part["label"].value_counts().nunique() == 1
