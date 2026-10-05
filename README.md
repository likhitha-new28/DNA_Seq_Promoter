# DNA Promoter / Enhancer Classifier

A local DNA classification workbench by **MU Likhitha**. It uses a small 1D CNN to
explore promoter, enhancer and background predictions, then shows which sequence
windows influenced the result. You can inspect held-out metrics and download a
prediction report from the same app.

**The default model learns planted motifs in synthetic DNA.** Its accuracy is a
software demonstration, not evidence that it identifies functional regulatory
regions in a real genome.

## Launch and use it locally

Install Python (3.11 is a good starting point), clone this repository, and open its
folder. On Windows, double-click **`run-local.cmd`**. The first launch installs the
dependencies and trains the demo model, so allow a few minutes and an internet
connection. Later launches reuse the prepared environment and model.

```powershell
git clone https://github.com/likhitha-new28/DNA_Seq_Promoter.git
cd DNA_Seq_Promoter
.\run-local.cmd
```

Open **http://127.0.0.1:8501** if your browser doesn't open automatically. Choose an
example or paste your own A/C/G/T/N sequence, then click **Classify sequence**. The
app shows all three model scores, the exact cropped or padded input, and influential
windows. Open **Evaluation** for the test metrics and confusion matrix, or
**Method & limitations** for the scientific context. Keep the terminal open while
using the app; press **Ctrl+C** there to stop it.

Before an interview, run `.\run-local.cmd -PrepareOnly` while online. Once prepared,
the local demo needs no dependency downloads or external prediction service.

```powershell
.\run-local.cmd -SkipBrowser          # open the URL yourself
.\run-local.cmd -Port 8502            # if another app uses port 8501
.\run-local.cmd -Retrain              # explicitly replace the model with a fresh demo
.\run-local.cmd -RefreshDependencies  # repair/reinstall the Python dependencies
```

On macOS/Linux, from the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m dna_classifier.cli prepare-demo
python -m streamlit run src/dna_classifier/webapp.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Python 3.13 / TensorFlow 2.21 were also tested on Windows. Other versions require a
compatible TensorFlow wheel. No GPU is required. If PowerShell blocks scripts, use
the `.cmd` launcher; it sets execution policy for that process only.

## Presenting the project

Follow the [five-minute interview walkthrough](docs/interview-demo.md). It covers
the demo order, likely bioinformatics questions and the limits of synthetic data.
The [recorded local validation](docs/demo-validation.md) includes measured results;
future training runs can differ by hardware and package version.

[![Launch in Codespaces](https://img.shields.io/badge/Launch-GitHub_Codespaces-2ea44f?logo=github)](https://codespaces.new/likhitha-new28/DNA_Seq_Promoter?quickstart=1)

Codespaces runs remotely and needs internet. Its setup prepares the same synthetic
demo and forwards port 8501. The local launcher is the preferred interview option.

## Train on your own labeled data

Provide a CSV with `sequence,label` columns. Labels are `promoter`, `enhancer`, or
`background`. Missing values, invalid bases and conflicting duplicate labels are
rejected; identical model inputs are deduplicated before splitting.

```bash
python -m dna_classifier.cli train --data path/to/sequences.csv --data-source "your dataset release and genome assembly"
python -m dna_classifier.cli evaluate --data data/processed/test.csv
python -m dna_classifier.cli predict --sequence ACGTTATAAAGCTACGTACGTACGT
```

Evaluation writes to `artifacts/evaluation/`, preserving the original training test
report. Models, generated data and local reports are ignored by Git. Custom trained
models with current, valid artifacts are reused by the launcher; `-Retrain` replaces
them with the synthetic demo. Short example sequences are heavily padded and may
produce unreliable predictions.

Read the [data guide](docs/data-guide.md) and [methodology](docs/methodology.md) before
using genomic annotations. Exact duplicate removal alone does not prevent leakage
from homologous sequences, overlapping loci or reverse complements.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check src tests
```

The tests cover preprocessing, splitting, metrics, prediction validation, batched
occlusion and Streamlit interactions. GitHub Actions runs the same checks and a
small TensorFlow training/evaluation smoke test. Licensed under [MIT](LICENSE).
