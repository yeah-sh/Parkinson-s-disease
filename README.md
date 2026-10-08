# pdeeg

Characterising and classifying Parkinson's disease (PD) from resting-state EEG: PD against healthy
controls, and PD off against on medication. The main features come from Multifractal Detrended
Fluctuation Analysis (MFDFA); band power from the power spectrum is the baseline.

**Status:** data access and preprocessing. Downloading, indexing, loading and cleaning the
recordings work. The later stages (features, models, statistics, figures) are not written yet.

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

## Configuration

Every setting, including every path, lives in `configs/`. Nothing is hard-coded in the package.

| File | Covers |
|---|---|
| `configs/data.yaml` | Dataset identity and montage, data and report paths, what each session label means |
| `configs/preprocessing.yaml` | Cropping, filtering, referencing, ICA, bad-stretch marking, QC limits |
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
```
