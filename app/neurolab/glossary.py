"""Short definitions of the terms the app uses, as this project uses them.

The MFDFA feature definitions follow pdeeg.features.mfdfa.spectrum_features; the evaluation
terms follow the evaluation code (src/pdeeg/models/evaluate.py) and configs/model.yaml.
Settings are not repeated here: the About page reads them from the configuration.
"""

from __future__ import annotations

TERMS: dict[str, str] = {
    "Sampling frequency": (
        "How many times per second each channel was measured, in hertz (Hz). The models were "
        "trained on recordings at one rate; a file at another rate is resampled to it, which "
        "the models never saw, so the app says so."
    ),
    "Montage": (
        "The named electrode positions on the scalp. The models need every channel of the "
        "dataset's montage, in its order, so a file is checked for each one by name."
    ),
    "ICA artifact removal": (
        "Independent component analysis splits the EEG into components that are as "
        "statistically independent as possible. ICLabel, a trained classifier, names each "
        "component (brain, eye blink, muscle, heart, line noise, channel noise, other); "
        "components it confidently calls an artefact are removed and the EEG is rebuilt from "
        "the rest."
    ),
    "Bad windows": (
        "Stretches of the cleaned recording where some channel's peak-to-peak amplitude in a "
        "short window is implausibly large (an artefact ICA left) or near zero (a flat or "
        "disconnected channel). They are marked, not cut out, and every window touching one is "
        "left out of the features computed on clean data."
    ),
    "Multifractal analysis": (
        "Multifractal detrended fluctuation analysis (MFDFA) asks how the size of a signal's "
        "fluctuations grows with the time scale it is looked at, separately for small and "
        "large fluctuations. If one exponent describes every size the signal is monofractal; "
        "if small and large fluctuations scale differently it is multifractal."
    ),
    "Fluctuation function": (
        "Fq(s): the signal is integrated, cut into windows of s samples, a polynomial trend "
        "is removed from each window, and the leftover variances are averaged as a q-th "
        "order mean. "
        "Negative q weights the small fluctuations, positive q the large ones. A straight "
        "line on log-log axes means the signal scales over that range."
    ),
    "Generalised Hurst exponent": (
        "h(q): the slope of log Fq(s) against log s over a fit range. h(2) is the exponent of "
        "ordinary detrended fluctuation analysis."
    ),
    "Singularity spectrum": (
        "f(α) against α, derived from h(q). α is a local scaling exponent and f(α) how much of "
        "the signal has it. A monofractal signal collapses to a point; a wider arch means more "
        "multifractal. h(q) has to fall as q rises for the usual arch shape."
    ),
    "h2": "h(2), the Hurst exponent of ordinary DFA, on the fit range.",
    "delta_h": "Δh = h(q_min) − h(q_max): how much the exponent changes across the moment orders.",
    "delta_alpha": "Δα = max(α) − min(α): the width of the singularity spectrum.",
    "alpha0": "α₀: the α at which f(α) peaks, the most common local exponent.",
    "asymmetry": (
        "(left − right) / (left + right), with left = α₀ − min(α) and right = max(α) − α₀. "
        "0 for a symmetric spectrum; positive when the left branch (q > 0, large "
        "fluctuations) is the longer one."
    ),
    "min_r2": ("The worst R² among the h(q) fits: a quality measure of the fits, not a feature."),
    "n_scales": (
        "How many of the fit range's window sizes the fit used; a size with too few clean "
        "windows is left out."
    ),
    "Band power": (
        "Power of the spectrum (Welch's method) within a frequency band, as a share of the "
        "total power and on a log10 scale: the baseline the MFDFA features are held against."
    ),
    "Balanced accuracy": (
        "The mean of sensitivity and specificity, so that a model cannot score well by "
        "favouring the larger class. Chance is 0.5 for two classes."
    ),
    "Confidence interval": (
        "The 95 % interval of a metric from resampling participants (bootstrap) on the "
        "out-of-fold predictions. A wide interval means the estimate is uncertain."
    ),
    "Cross-validation by participant": (
        "The data are split into folds so that all recordings of one person are in the same "
        "fold. Each fold is predicted by a model fitted on the others, so no prediction comes "
        "from a model that saw that person. Hyperparameters are tuned in an inner loop inside "
        "each training set (nested cross-validation)."
    ),
    "Out-of-fold prediction": (
        "The prediction a recording got in cross-validation, from the model of the fold that "
        "left its participant out. The evaluation metrics are computed on these."
    ),
    "Permutation test": (
        "The labels are shuffled and the whole nested cross-validation is run again, many "
        "times. The p-value is the share of shuffles that scored at least as well as the real "
        "labels: how often chance alone does this well."
    ),
    "q-value": (
        "A p-value adjusted (Benjamini-Hochberg) for having picked the best of several "
        "feature-set and model combinations on the same cross-validation."
    ),
    "Better than chance": (
        "This project's rule for a stage: its p- or q-value (as configured) below the "
        "configured alpha and the whole 95 % interval of balanced accuracy above chance. A "
        "stage that fails it is greyed out."
    ),
    "Uncertain band": (
        "Probabilities for which the stage gives no label. It is derived from the out-of-fold "
        "probabilities: a probability keeps its label only if what it turned out to mean in "
        "cross-validation is reliably on the same side of the threshold."
    ),
    "Model score": (
        "The Platt-scaled probability a stage outputs: how the model's score maps to the "
        "classes it was trained on. It is not the probability that a person has Parkinson's "
        "disease or is off medication."
    ),
}


def define(term: str) -> str:
    return TERMS[term]
