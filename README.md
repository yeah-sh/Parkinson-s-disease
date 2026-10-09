# pdeeg

Characterising and classifying Parkinson's disease (PD) from resting-state EEG: PD against healthy
controls, and PD off against on medication. The main features come from Multifractal Detrended
Fluctuation Analysis (MFDFA); band power from the power spectrum is the baseline.

**Status:** data access, preprocessing, and the MFDFA algorithm. Downloading, indexing, loading
and cleaning the recordings work, and the MFDFA code is validated on synthetic signals. It is not
yet applied to the EEG; the later stages (features, models, statistics, figures) are not written.

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
recorded at UC San Diego on a 32-channel BioSemi system at 512 Hz. It is not tracked in git.

```
python scripts/download_data.py     # fetch into data/raw/ds002778; re-running only verifies
pdeeg recordings                    # write data/interim/recordings.parquet, one row per recording
```

In code, `pdeeg.data.bids.list_recordings` builds that table and `load_raw` reads one of its rows
as 32 scalp EEG channels with the montage applied.

Things in the dataset that later stages have to allow for:

- **Two on-medication sessions are not raw recordings.** `sub-pd6` and `sub-pd16` `ses-on` were
  exported from already preprocessed EEGLAB data (see `notes` in `participants.tsv`): they are
  high-pass filtered and have no mains peak, unlike every other recording, including the same
  patients' off-medication sessions.
- **Mains frequency is 60 Hz.** Nine sidecars say 50, which the signals do not support, so use
  `dataset.line_freq` from `configs/data.yaml` rather than `raw.info["line_freq"]`.
- **`EXG1`-`EXG8` are typed EEG** in every `channels.tsv` and are undocumented. `load_raw` drops
  them because the montage has no position for them.
- **An undocumented trigger with value 1** is in 44 of the 46 recordings: 2-29 s in, except 96 s
  in for `sub-pd6` `ses-on`. At least 181 s of data follow it. `sub-hc4` and `sub-pd16` `ses-on`
  do not have it.

Read `data/raw/ds002778/README` before publishing anything from this data. The curators ask to be
contacted before a manuscript is submitted, and they caution specifically against presenting
machine-learning classification of patients against controls as a diagnostic result, because 15
patients are too few for reliable test statistics.

## Preprocessing

```
pdeeg preprocess --config configs/preprocessing.yaml    # also takes --n-jobs N and --force
```

Each recording becomes one continuous, cleaned file; nothing is cut into epochs or cut out,
because MFDFA needs continuous data. The steps, all set in `configs/preprocessing.yaml`:

1. Crop to 180 s, starting at trigger 1 where the recording has it.
2. Band-pass 0.5-50 Hz (zero-phase FIR) and re-reference to the average.
3. Fit ICA on a 1-100 Hz copy with as many components as the data rank, classify the components
   with ICLabel, and remove those it calls eye, muscle, heart or line noise with a probability of
   at least 0.8.
4. Mark the remaining bad stretches as `BAD_peak` and `BAD_flat` annotations.

Outputs:

- `data/interim/clean/sub-XX_ses-YY_clean.fif`: the cleaned recording. Read it with
  `pdeeg.preprocessing.pipeline.read_clean`.
- `data/interim/clean/sub-XX_ses-YY_clean.json`: its log, with the crop window, mains frequency,
  removed components and their ICLabel probabilities, bad stretches and per-channel statistics.
- `reports/qc_preprocessing.md` and `reports/figures/qc_preprocessing/`: the QC report, with a
  spectrum and the removed components for every recording and a list of flagged recordings.

A recording is skipped when its outputs are newer than its input files and were written with the
same settings, so re-running after a change only redoes what the change affects. The seed is
fixed in the configuration; re-running two recordings in a single process reproduced the
six-worker run byte for byte.

## MFDFA

`pdeeg.features.mfdfa` implements multifractal detrended fluctuation analysis as in Kantelhardt et
al., Physica A 316 (2002) 87-114. It takes any one-dimensional series:

```python
from pdeeg.features.mfdfa import make_qs, make_scales, mfdfa, spectrum_features

qs = make_qs(-5.0, 5.0, 0.5)  # moment orders
scales = make_scales(len(x), 16, 0.1, 20)  # 20 log-spaced window sizes, 16 samples to N / 10
result = mfdfa(x, scales, qs, order=1)  # Fq, h, h_r2, tau, alpha, f_alpha
features = spectrum_features(result, qs)  # h2, delta_h, delta_alpha, alpha0, asymmetry, min_r2
```

`pdeeg.features.surrogates` makes shuffled and IAAFT surrogates of a series, and
`pdeeg.features.synthetic` the test signals with known exponents: fractional Gaussian noise and
the binomial multiplicative cascade.

`tests/test_mfdfa.py` and `notebooks/01_mfdfa_validation.ipynb` check the code against those
signals and against the PyPI `MFDFA` package, which returns the same fluctuation functions to 11
digits. With the values in `configs/features/mfdfa.yaml` and 65536 samples:

- `h(2)` of a monofractal signal has a bias of 0.003 or less and a standard deviation of 0.01 to
  0.02.
- A monofractal signal still shows a spectrum width `delta_alpha` of 0.03 to 0.06 (0.12 for a
  random walk), because the estimator widens the spectrum at finite length. Judge a width against
  surrogates, not against zero.
- For the binomial cascade the width is within 4 % of the exact 1.57. `h(q)` is up to 0.065 too
  low, because the cascade reaches its scaling only at large scales and log-spaced windows
  straddle its dyadic boxes; fitted from 256 samples on dyadic scales it is 0.011 low at every q.
- Detrending of order m removes a polynomial trend of degree m from the profile, which is degree
  m - 1 in the series itself.

To re-run the notebook: `jupyter execute --inplace notebooks/01_mfdfa_validation.ipynb`.

## Configuration

Every setting, including every path, lives in `configs/`. Nothing is hard-coded in the package.

| File | Covers |
|---|---|
| `configs/data.yaml` | Dataset identity and montage, data and report paths, what each session label means |
| `configs/preprocessing.yaml` | Cropping, filtering, referencing, ICA, bad-stretch marking, QC limits |
| `configs/features/mfdfa.yaml` | MFDFA moment orders, scales, detrending, fit range, surrogates |
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
```
