# Methodology

## Representation

Every sequence is converted to a matrix with one row per position and four columns ordered
`A, C, G, T`. The matching base receives 1 and the others 0. An ambiguous or padded `N` becomes
`[0, 0, 0, 0]`. Long sequences are centre-cropped and short sequences are padded equally on both
sides.

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

## Interpretation

Prediction uses occlusion: each non-overlapping window is masked, then the drop in the predicted
class probability is measured. A large drop means the window supported that prediction. This is a
helpful diagnostic, not proof that a region is functional or causal.
