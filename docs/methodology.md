# Methodology

## Representation

Every sequence is converted to a matrix with one row per position and four columns ordered
`A, C, G, T`. The matching base receives 1 and the others 0. An ambiguous or padded `N` becomes
`[0, 0, 0, 0]`. Long sequences are centre-cropped and short sequences are padded equally on both
sides.

Before splitting, inputs are standardized to the model length and exact duplicates
are removed. Identical inputs with conflicting labels raise an error. Each split
must contain all three classes; undersized datasets produce a clear validation error.
This handles exact input duplication only, not homologous, reverse-complement or
overlapping genomic regions. Genomic studies need stronger locus-aware splitting.

## Network

The network uses two one-dimensional convolution layers. These filters slide across the sequence
and can learn short recurring patterns similar to regulatory motifs. Max pooling and global max
pooling retain strong responses while reducing dimensionality. Dropout discourages memorisation,
and a softmax output returns one probability for each of the three classes.

Training uses Adam and sparse categorical cross-entropy. Early stopping restores the weights from
the lowest validation loss, while learning-rate reduction makes smaller updates after a plateau.

## Metrics

- **Accuracy**: overall fraction classified correctly.
- **Macro precision/recall/F1**: calculates each class separately and weights classes equally.
- **One-vs-rest macro ROC-AUC**: measures probability ranking for each class.
- **Confusion matrix**: shows which classes are confused with one another.

No single metric proves biological usefulness. Report variability across seeds or cross-validation,
compare against simple baselines, and test on independently collected data.

Macro precision, recall and F1 include every configured class, including absent
classes (zero score). ROC-AUC is reported as unavailable when any class is missing
from the evaluation set. Reports include class support and a machine-readable
confusion matrix. Running `evaluate` saves separate reports under
`artifacts/evaluation/` so it cannot silently replace the original training test metrics.

## Interpretation

Prediction uses occlusion: each non-overlapping window is masked, then the drop in the predicted
class probability is measured. A large drop means the window supported that prediction. This is a
helpful diagnostic, not proof that a region is functional or causal.

All windows, including a shorter final window, are tested together in one masked
prediction batch. Coordinates are 0-based and end-exclusive in the standardized
input. For a centre crop, add the recorded `crop_start` to recover input-string
positions. For padding, subtract `padding_left`; padded positions have no original
base. No genomic coordinate mapping is inferred. Negative importance means masking
increased the predicted-class score.

## Reproducibility and artifacts

`prepare-demo` generates 600 sequences per class (200 bases, seed 42), creates
70/15/15 stratified partitions, and trains for up to 30 epochs with early stopping.
It reuses a valid current model on later launches. The independent UI examples use
seed 2026 and are not selected by model predictions. TATAAA and CACGTG are planted
teaching patterns; random background can also contain them by chance.

Settings record data/model SHA-256 hashes, source description, seed, TensorFlow
version, epochs, batch size and split/class counts. TensorFlow deterministic operations
are enabled for training; exact results can still vary across hardware and versions.
Predictions verify the model hash and input/output shapes against its settings.
Training runs in a temporary artifact directory so a failed fit leaves the previous
model usable. Publication of a completed run is not a multi-file atomic transaction;
avoid concurrent training jobs targeting the same artifact directory.
