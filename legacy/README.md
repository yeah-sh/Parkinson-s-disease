# Legacy notebooks

These are the project's original notebooks. They were moved here unchanged on 2026-10-08
(byte-identical to commit `1ab0029`) and are kept for reference only. The pipeline is being
rewritten as the `pdeeg` package under `src/`; nothing here is maintained, linted or tested.

## What each notebook does

| Notebook | Purpose |
|---|---|
| `processing3.ipynb` | Walkthrough on one recording (`sub-pd26`, OFF): load BDF, 0.5-50 Hz filter, ICA, 1 s epochs, multitaper band power, phase-locking value |
| `main3.ipynb` | The same preprocessing over all subjects; writes one band's epoch-level band power (epochs x 32 channels) to a pickle per run |
| `svm.ipynb` | Linear SVM on one band's epoch-level features |
| `cnn3.ipynb` | 2-D CNN on five bands x 32 channels per epoch |
| `1d_cnn3.ipynb` | 1-D CNN on one band's epoch-level features |
| `check_data.ipynb` | Loads a feature pickle and prints its shape and label counts |
| `subject_independent/processing3.ipynb` | Copy of `processing3.ipynb` with `../` paths |
| `subject_independent/main3.ipynb` | Writes per-subject feature arrays (subjects x epochs x channels), one label per subject |
| `subject_independent/svm.ipynb` | RBF SVM on flattened per-subject features, split by subject |
| `subject_independent/cnn3.ipynb` | 2-D CNN on per-subject feature images, split by subject |

## Why they were replaced

- **Data leakage.** `svm.ipynb`, `cnn3.ipynb` and `1d_cnn3.ipynb` split and cross-validate over
  1-second epochs pooled across subjects, so the same person's epochs are in both train and test.
  The accuracies saved in those notebooks are not estimates of performance on new people and
  should not be quoted.
- **Leakage in the subject-level variants too.** `subject_independent/` splits by subject, but the
  scaler is still fitted on all subjects before the split, and the test set is used as the
  validation set during training.
- **Duplicated code.** The same preprocessing functions (`set_montage`, `bandpass_filter`,
  `find_ecg_via_temporal_channels`, `apply_ica`, `segment_data`, `compute_psd`) are copy-pasted
  into every processing notebook.
- **No re-referencing.** BioSemi recordings are reference-free; the notebooks drop the external
  channels and never set a reference.
- **Pickles as the data format**, written one frequency band per manual re-run.

## Why they will not run as they are

- They read `Dataset/`, `new_data/` and `IowaDataset/` relative to the notebook's working
  directory. The raw data now lives in `data/raw/ds002778/`, the feature pickles were deleted, and
  the Iowa data was never in this repository.
- They were run on a Python 3.9 kernel with TensorFlow and seaborn, which are not part of this
  project's environment.

## Other things to know

- **Iowa dataset.** The project used a second dataset from the University of Iowa, which has been
  dropped, and the three notebooks dedicated to it were deleted. Three notebooks kept here were
  nonetheless last run against Iowa-derived features (29 channels, 120 s per subject):
  `svm.ipynb`, `1d_cnn3.ipynb` and `subject_independent/cnn3.ipynb`.
- **Saved outputs** in the two `processing3.ipynb` notebooks contain absolute paths from the
  machine they were originally run on.
