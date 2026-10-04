# Data guide

## Suggested public resources

Real regulatory-region experiments need carefully chosen, consistently annotated data. Useful
starting points include:

- **EPDnew** for experimentally validated eukaryotic promoters.
- **FANTOM5** and **ENCODE SCREEN** for candidate enhancers.
- **Ensembl** or **UCSC Genome Browser** for reference genome sequences and coordinates.

Check each provider's license and citation requirements. Keep downloaded files under `data/raw/`;
that directory is intentionally ignored because genomic datasets can be large and may be subject to
specific redistribution terms.

## Avoiding misleading results

- Use one reference genome assembly across every class (for example, GRCh38).
- Match background regions to regulatory regions by length and, where practical, GC content.
- Remove overlapping and duplicate regions before splitting. Otherwise nearly identical sequences
  can leak into training and test sets.
- Consider chromosome-based splits for a stronger generalisation test.
- Treat database labels as operational annotations, not perfect biological ground truth.
- Record the source, release, genome build, query date, and filtering steps.

## Converting FASTA files

The package exposes `load_fasta` for simple one-label FASTA files. A short conversion script is:

```python
import pandas as pd
from dna_classifier.data import load_fasta

frames = [
    load_fasta("promoters.fa", "promoter"),
    load_fasta("enhancers.fa", "enhancer"),
    load_fasta("background.fa", "background"),
]
pd.concat(frames, ignore_index=True).to_csv("data/raw/sequences.csv", index=False)
```

Inspect class counts and sequence lengths before training. The CLI centre-crops and pads sequences,
but choosing biologically meaningful windows is part of the experimental design.

