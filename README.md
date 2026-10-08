# pdeeg

Characterising and classifying Parkinson's disease (PD) from resting-state EEG: PD against healthy
controls, and PD off against on medication. The main features come from Multifractal Detrended
Fluctuation Analysis (MFDFA); band power from the power spectrum is the baseline.

**Status:** scaffold. The package installs, the configuration loads and `pdeeg config` prints it.
The pipeline stages (preprocessing, features, models, statistics, figures) are not written yet.

## Setup

Requires Python 3.11 or 3.12.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
ruff check .
pytest
```

## Data

The data is [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778/versions/1.0.5), version
1.0.5: 16 controls with one session each and 15 patients with an off- and an on-medication session,
recorded at UC San Diego on a 32-channel BioSemi system at 512 Hz. It is not tracked in git. Put it
in `data/raw/ds002778/`, for example with:

```
openneuro-py download --dataset ds002778 --tag 1.0.5 --target-dir data/raw/ds002778
```

Read `data/raw/ds002778/README` before publishing anything from this data. The curators ask to be
contacted before a manuscript is submitted, and they caution specifically against presenting
machine-learning classification of patients against controls as a diagnostic result, because 15
patients are too few for reliable test statistics.

## Configuration

Every setting, including every path, lives in `configs/`. Nothing is hard-coded in the package.

| File | Covers |
|---|---|
| `configs/data.yaml` | Dataset identity, data and report directories, session-to-condition mapping |
| `configs/preprocessing.yaml` | Referencing, filtering, ICA, epoching |
| `configs/features/mfdfa.yaml` | MFDFA moment orders, scales, detrending |
| `configs/features/psd.yaml` | Spectral estimation and frequency bands |
| `configs/model.yaml` | Classification tasks, cross-validation, experiment tracking |

The config directory is taken from `--config-dir`, then the `PDEEG_CONFIG_DIR` environment
variable, then `./configs`. Relative paths in the YAML files are resolved against the parent of the
config directory. Every key is required and unknown keys are rejected. To see the resolved result:

```
pdeeg config
```

## Layout

```
configs/        YAML configuration
src/pdeeg/      the package: config.py, cli.py, and data/, preprocessing/, features/,
                models/, stats/, viz/
scripts/        one-off scripts
tests/          pytest tests
notebooks/      exploratory notebooks that import pdeeg
reports/        generated figures and tables
data/           raw/, interim/, processed/ (contents git-ignored)
legacy/         the original notebooks, kept for reference; see legacy/README.md
```
