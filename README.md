# pdeeg

Characterising and classifying Parkinson's disease (PD) from resting-state EEG: PD against healthy
controls, and PD off against on medication. The main features come from Multifractal Detrended
Fluctuation Analysis (MFDFA); band power from the power spectrum is the baseline.

**Status:** data access, preprocessing, the MFDFA algorithm, and the features. Downloading,
indexing, loading and cleaning the recordings work; the MFDFA code is validated on synthetic
signals; the fit ranges were chosen without sight of the group labels; and MFDFA features with
surrogates and band-power features are extracted for every recording, with a QC report. The later
stages (models, statistics) are not written, and nothing has been compared between groups.

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

### Where the EEG scales

MFDFA is run on two kinds of series per channel: the cleaned EEG itself ("broadband"), and the
Hilbert amplitude envelope of the EEG band-passed to theta, alpha or beta
(`pdeeg.features.envelope`). Before any exponent is fitted,

```
pdeeg mfdfa-scaling                 # also takes --n-jobs N
```

plots `log Fq(s)` against `log s` and its local slope for 8 recordings picked at random, next to
noise put through the same filters, and writes `reports/mfdfa_scaling.md`. The recordings appear
as R1 to R8: the command does not load the group labels, so the fit ranges read from the report
cannot be tuned towards a group difference. The ranges and the reasons for them are in
`configs/features/mfdfa.yaml`. In short:

- **Broadband** has no single scaling regime. The 50 Hz low-pass bends it below about 0.05 s, the
  alpha rhythm between 0.09 and 0.4 s, and the 0.5 Hz high-pass above about 1 s. Two short ranges
  are left, 0.045-0.09 s and 0.3-1 s. On the first, filtered monofractal noise is as "multifractal"
  as the EEG, so only `h2` means anything there.
- **Envelopes** are fitted from 2 s to a tenth of the series (17.8 s). Below 2 s the band-pass
  filter alone gives the envelope of white noise a memory.

## Features

```
pdeeg mfdfa-features                # also takes --n-jobs N and --force
pdeeg psd-features                  # also takes --n-jobs N
```

`pdeeg mfdfa-features` runs MFDFA on every channel of every recording, for each series (the
broadband EEG and the theta, alpha and beta envelopes) and each fit range, and writes

- `data/processed/mfdfa_features_ds002778-1.0.5.parquet`: the long table, one value per row, with
  the columns `subject`, `session`, `group`, `level` (`channel` or `region`), `name` (the channel
  or region), `variant` (`broadband` or `envelope`), `band`, `fit_range`, `segments`, `feature` and
  `value`;
- `data/processed/mfdfa_features_ds002778-1.0.5_wide.parquet`: one row per recording and
  `segments`, one column per feature, named `<series>_<fit range>_<feature>_<channel or region>`,
  for modelling. It holds the features listed under `wide_features` in the configuration: all
  five for the long broadband range and the envelopes, `h2` alone for the short broadband range;
- `reports/qc_mfdfa.md`: the QC report.

What the long table holds for each channel, series and fit range:

| `feature` | What it is |
|---|---|
| `h2`, `delta_h`, `delta_alpha`, `alpha0`, `asymmetry` | The MFDFA features (`pdeeg.features.mfdfa.spectrum_features`) |
| `min_r2` | Worst R² of the h(q) fits: a quality measure, not a feature |
| `n_scales` | Scales the fit used: 12, or fewer when bad stretches left too few windows |
| `delta_alpha_iaaft_mean`, `_sd` | Mean and standard deviation of `delta_alpha` over 20 IAAFT surrogates |
| `delta_alpha_iaaft_z` | (`delta_alpha` - that mean) / that standard deviation |
| `delta_alpha_iaaft_p` | Share of the surrogates, the original counted in, at least as wide as the original; 1/21 when it is wider than all 20 |
| `h2_iaaft_mean` | Mean `h2` of the surrogates, which should equal `h2`: a check on the surrogates |
| `..._shuffle_...` | The same five for 20 shuffled surrogates |

Three things about these tables:

- **`segments`.** Every feature is computed on the whole recording (`full`) and again without
  the windows that touch a stretch annotated bad (`clean`). Windows are dropped where they stand;
  data is never cut out and joined. Surrogates are made from the whole series, so their rows are
  `full` only. For the 22 recordings without a bad stretch the two are equal.
- **Regions.** A region's value is the mean of its channels' values. The five regions (frontal,
  central, temporal, parietal, occipital) are defined in `configs/data.yaml`.
- **Runtime.** The surrogates take most of the time: 117,760 IAAFT surrogates of 92,160 samples,
  about 5 minutes per recording on 18 processes. Every MFDFA run is stored per recording in
  `data/interim/mfdfa/`, a recording is skipped when its stored runs are current, and the tables
  are rebuilt from the stored runs in seconds.

`pdeeg psd-features` writes the baseline, `data/processed/psd_features_ds002778-1.0.5.parquet`
(long, with `band` in place of the MFDFA columns) and its `_wide` companion: log10 relative band
power from the Welch spectrum, delta to gamma, per channel and region, again `full` and `clean`.
Run it before `pdeeg mfdfa-features` to have the band powers included in the QC report.

## Configuration

Every setting, including every path, lives in `configs/`. Nothing is hard-coded in the package.

| File | Covers |
|---|---|
| `configs/data.yaml` | Dataset identity and montage, data and report paths, what each session label means, scalp regions |
| `configs/preprocessing.yaml` | Cropping, filtering, referencing, ICA, bad-stretch marking, QC limits |
| `configs/features/mfdfa.yaml` | MFDFA moment orders, detrending, envelope bands, fit ranges and why, bad stretches, surrogates, outputs, the scaling inspection |
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
