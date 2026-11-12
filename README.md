# SystS

SystS is a PySide6 desktop application for systematic literature reviews. It
searches academic databases, screens papers, extracts claims and statistics,
synthesizes evidence, performs meta-analysis and robustness checks, and exports
source-linked PRISMA/GRADE reports.

This is the app-only poster release. It contains application source and required
configuration assets, not research data, credentials, model weights, training
tools, development tests, or the development repository's Git history.

## Install and run

Use Python 3.13 or newer. Run these commands from this directory on macOS/Linux:

```sh
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/python main.py
```

On Windows:

```powershell
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe main.py
```

`requirements.txt` contains the runtime dependencies. A virtual environment and
model downloads can be large; they are intentionally not part of this repository.
PyMuPDF and pypdf provide the standard PDF readers. The commented requirements
offer additional, optional PDF fallbacks.

## Start a review

1. Create a project with **File > New Project**. Store review projects outside
   this source directory so downloaded papers and outputs stay separate.
2. Enter your research question and search terms. Configure provider credentials,
   endpoint URLs, and model choices in **File > Settings** and the stage tabs.
3. Optionally adapt [the blank template](templates/blank_template.yaml) in your
   project directory and load it through Settings. Replace its placeholder review
   text before running; it is an example, not an automatic startup configuration.
4. Configure Stage 2 as described below before running the full pipeline.

No personal research configuration is bundled. API keys are supplied locally and
are excluded from exported configuration. **Remember API keys** controls their
storage in the operating system's QSettings store. Another SystS installation on
the same user account shares that preferences store; a separate source copy does
not create a separate application identity.

## Models and providers

The GUI opens without screening weights, but model-backed stages need a working
provider and the selected models. Supported providers are Ollama, OpenAI,
Anthropic, and custom endpoints using OpenAI-compatible, Anthropic-compatible,
or LM Studio REST formats. Local servers must be running; hosted services require
your own credentials and may incur charges. Searches and model downloads require
network access, and some literature databases require institutional access.

### Default Laya + BRT-BERT screening

New projects default to **Laya -> BRT-BERT**. The Python runtime is installed from
`requirements.txt`, but the calibrated model package is distributed separately.
This repository does not include that package or provide an automatic download.
Obtain the compatible package and its distribution terms from its distributor.

The runtime needs the package's original `SHA256SUMS.txt` and all files listed
under `model/laya/` and `model/brt-bert/`. Keep that structure intact. Training
code, evidence reports, and other package source files are not needed by this app.
Use the conventional local layout below, or set `laya_screener_dir` to the
package's absolute directory in your project's YAML/JSON configuration:

```text
laya-screener-pw4/
  SHA256SUMS.txt
  model/
    laya/
    brt-bert/
```

Every model file is verified against the manifest before scoring. Missing or
modified files stop the run; do not disable this integrity check. The default
gates are Laya probability >= 0.6 and packaged BRT-BERT probability >= 0.236.
Raw probabilities and both gate decisions are retained. Validate screening
against human-labelled examples from your own review; these thresholds do not
guarantee performance on a new subject or corpus.

### Screening without the packaged weights

In Settings, enable **Show advanced LLM options**. In Stage 2 select
**Existing BERT / LLM scorer**, then choose a working LLM provider and model,
not BRT-BERT. Selecting the strategy alone leaves the existing BERT choice in
place. This is an alternative scoring method, not an equivalent implementation
of the calibrated cascade. Other model-backed stages also need provider setup.

## Pipeline

| Stage | Purpose |
| --- | --- |
| 1 | Search academic providers and deduplicate results |
| 2 | Screen relevance and support human review |
| 3 | Extract structured claims |
| 4 | Review methods and validate spans |
| 5 | Audit evidence grounding |
| 6 | Cluster and synthesize evidence |
| 7 | Retrieve full text and extract metrics |
| 8 | Run robustness and configured network analyses |
| 9 | Generate the final source-linked review and internal checks |
| 10 | Compare saved review snapshots |

**Run All** ends at Final Review. Compare Runs is launched separately. Historical
module numbers and compatibility helpers are retained where application imports
need them. Statistical checks and generated reports support researcher review;
they do not independently establish scientific validity.

## Contents

`main.py` starts the app. `acquire/`, `analysis/`, `core/`, `gui/`, `process/`, and
`stages/` contain its runtime modules. `templates/domain_patterns/` supplies the
general and neuroscience vocabulary packs used by the application. The GUI's
built-in help covers stage controls and workflows.

The development test suite is maintained separately rather than shipped in this
app-only copy. No application or model license grant is specified by this copy;
confirm the relevant permissions and licensing before public distribution.
