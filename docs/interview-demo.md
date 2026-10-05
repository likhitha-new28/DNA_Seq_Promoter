# Five-minute interview walkthrough

## Prepare before the interview

Run `run-local.cmd -PrepareOnly` at least once while online, then launch the app
and try all three examples. Keep the repository and Python environment on the
laptop you will present from. Do not plan to install TensorFlow during the interview.
Use `run-local.cmd` to start; the app is at http://127.0.0.1:8501. Stop with Ctrl+C.

## A short opening in your own words

“I built an end-to-end DNA sequence classification prototype. It validates the data,
converts bases into four numerical channels, trains a small convolutional network,
and lets me inspect both predictions and held-out evaluation. For this demonstration
I use synthetic sequences with planted motifs, so I can test the workflow in a
reproducible way. I would need curated genomic data and independent validation before
claiming that it recognizes real promoters or enhancers.”

## Show the workflow

| Time | Show | Explain |
| --- | --- | --- |
| 0:00–1:00 | Predict the promoter example | This is full-length DNA with a planted TATAAA motif. Show all three scores, not just the winning class. |
| 1:00–2:00 | Influential windows and exact model input | Occlusion replaces windows with N and measures a score change. Positions refer to the 200-base model input, not genomic coordinates. |
| 2:00–3:00 | Enhancer example and background control | The enhancer example contains CACGTG. Background is random DNA; chance motifs and mistakes remain possible. These examples use a separate seed from training. |
| 3:00–4:00 | Evaluation | Explain validation versus test data, macro F1, class support and the confusion matrix. A useful model must handle all classes, not just promoters. |
| 4:00–5:00 | Method & limitations | Explain synthetic-data limits, leakage prevention and what a real biological study would require. Download a report if the panel wants an artifact. |

If useful, paste `ACGX` to show validation, then `ACGT` to demonstrate the padding
warning. An all-N input is rejected because it contains no known bases. Return to a
full-length example before continuing. These checks make preprocessing visible.

## Questions bioinformaticians may ask

**What is the biological task?** Promoters are associated with transcription
initiation; enhancers can regulate transcription at a distance. The classifier uses
three exclusive teaching labels. Real regulatory functions can overlap and depend
on cellular context; the planted motifs are not universal definitions of these classes.

**Why a CNN?** Convolution filters can detect local patterns across the sequence.
Pooling summarizes strong responses. The model is small enough for a CPU demo.
It cannot model chromatin accessibility, tissue context or long-range interactions
from these sequence windows alone.

**How do you prevent data leakage?** The pipeline cleans and centre-crops/pads first,
then removes identical inputs before stratified splitting. Conflicting labels for
identical inputs raise an error. This does not resolve related sequences or overlapping
genomic coordinates. With real data, preserve locus identifiers, remove overlaps and
use chromosome-based or external test sets before fitting.

**What does 94.8% accuracy mean?** In the recorded local run, 256 of 270 held-out
synthetic sequences were correctly classified. It demonstrates learning in this
generator's distribution. It does not estimate performance on human genomic DNA.
Macro F1 and the per-class report expose failures hidden by overall accuracy.

**What would you compare against?** A majority-class reference (one third here),
simple motif matching and a k-mer logistic-regression model, followed by repeated
seeds and external validation. Those are proposed comparisons; this repository has
not established superiority over them.

**Are the scores calibrated?** No calibration analysis is implemented. A 90% softmax
score is the model's relative score, not a 90% probability of biological function.

**Does occlusion discover functional motifs?** It identifies sensitivity to replacing
a window with zeros. That replacement can be outside the training distribution and
can disrupt several patterns at once. It supports debugging, not causal annotation.
Positive importance supports the selected class; negative importance opposes it.

**How would you move to real data?** Start with well-documented annotations such as
EPDnew promoters or ENCODE candidate regulatory elements. Fix genome build and cell
context, define labels, and match background length and GC content. Record provenance,
perform stronger splits, assess imbalance and domain shift, and validate externally.
See the [data guide](data-guide.md) for source links and cautions.

## If something goes wrong

- Port occupied: close the previous app or use `run-local.cmd -Port 8502`.
- Browser did not open: enter the displayed local URL manually.
- Model/settings mismatch: run `run-local.cmd -Retrain` before the interview.
- Dependency errors: use `-RefreshDependencies` while online; avoid changing the
  environment immediately before presenting.
- Unexpected prediction: show the scores and limitations. Do not claim every motif
  guarantees its intended label; the model is a learned approximation.
