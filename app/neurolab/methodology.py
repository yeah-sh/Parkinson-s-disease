"""The pipeline as it is configured, for the About page and the report.

Every value is read from the configuration (configs/) or the saved models' metadata; nothing
here is typed in by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne

from pdeeg.config import Config


@dataclass(frozen=True)
class Section:
    title: str
    summary: str
    rows: list[tuple[str, str]]


def _ranges(ranges: dict[str, tuple[float, float]]) -> str:
    return "; ".join(f"{name} {low:g}–{high:g} s" for name, (low, high) in ranges.items())


def sections(config: Config, evidence: dict[str, dict[str, Any]]) -> list[Section]:
    prep, mfdfa, psd, model, predict = (
        config.preprocessing,
        config.mfdfa,
        config.psd,
        config.model,
        config.predict,
    )
    montage = config.data.dataset.montage
    n_channels = len(mne.channels.make_standard_montage(montage).ch_names)
    crop = prep.crop
    ica = prep.ica
    out = [
        Section(
            "1 · Input checks",
            "Every file is read and checked before anything is computed from it; a failed check "
            "stops the analysis with the reason.",
            [
                ("Channels", f"all {n_channels} of the {montage} montage, matched by name"),
                ("Sampling rate", f"{predict.sfreq:g} Hz (others resampled, with a note)"),
                ("Length", f"at least {crop.duration:g} s from the crop start"),
                (
                    "Amplitude",
                    f"median channel SD (1–40 Hz) within {predict.amplitude_uv[0]:g}–"
                    f"{predict.amplitude_uv[1]:g} µV",
                ),
            ],
        ),
        Section(
            "2 · Preprocessing",
            "One continuous window per recording is kept and cleaned; nothing is cut out, "
            "because MFDFA needs continuous data. Bad stretches are only marked.",
            [
                (
                    "Window",
                    f"{crop.duration:g} s from trigger “{crop.start_event}”, else from the file "
                    "start"
                    if crop.start_event is not None
                    else f"{crop.duration:g} s from the file start",
                ),
                (
                    "Band-pass",
                    f"{prep.filter.l_freq:g}–{prep.filter.h_freq:g} Hz, {prep.filter.phase}-phase "
                    f"FIR ({prep.filter.fir_window}), transitions {prep.filter.l_trans_bandwidth:g}"
                    f" / {prep.filter.h_trans_bandwidth:g} Hz",
                ),
                (
                    "Reference",
                    prep.reference
                    if isinstance(prep.reference, str)
                    else ", ".join(prep.reference),
                ),
                (
                    "ICA",
                    f"{'extended ' if ica.extended else ''}{ica.method}, "
                    + (
                        "as many components as the data's rank"
                        if ica.n_components == "rank"
                        else f"{ica.n_components} components"
                    )
                    + f", fitted on a {ica.fit_l_freq:g}–{ica.fit_h_freq:g} Hz copy"
                    + (
                        f" notched at {config.data.dataset.line_freq:g} Hz" if ica.fit_notch else ""
                    ),
                ),
                (
                    "Removed",
                    f"components ICLabel calls {', '.join(ica.exclude_labels)} with probability "
                    f"≥ {ica.threshold:g}",
                ),
                (
                    "Bad windows",
                    f"{prep.bad_segments.window:g} s windows where any channel exceeds "
                    f"{prep.bad_segments.peak_to_peak_uv:g} µV peak-to-peak or stays below "
                    f"{prep.bad_segments.flat_uv:g} µV",
                ),
            ],
        ),
        Section(
            "3 · Multifractal analysis (MFDFA)",
            "Run per channel on the cleaned EEG and on the amplitude envelope of each band; "
            f"the fit ranges were chosen from {mfdfa.inspection.n_recordings} randomly picked "
            "recordings with no label seen.",
            [
                (
                    "Moment orders",
                    f"q from {mfdfa.q_min:g} to {mfdfa.q_max:g} in steps of {mfdfa.q_step:g}",
                ),
                ("Detrending", f"polynomial of order {mfdfa.detrend_order} per window"),
                (
                    "Window sizes",
                    f"{mfdfa.scales_per_range} per fit range, log-spaced, at most "
                    f"{mfdfa.scale_max_frac:g} of the series",
                ),
                ("Broadband fit ranges", _ranges(mfdfa.broadband.fit_ranges)),
                (
                    "Envelopes",
                    ", ".join(
                        f"{b} {lo:g}–{hi:g} Hz" for b, (lo, hi) in mfdfa.envelope.bands.items()
                    )
                    + f"; fit range {_ranges(mfdfa.envelope.fit_ranges)}",
                ),
                (
                    "Bad windows",
                    f"windows touching one are dropped; a scale needs "
                    f"{mfdfa.bad_windows.min_windows} clean windows and a fit "
                    f"{mfdfa.bad_windows.min_scales} scales",
                ),
                ("Features", ", ".join(mfdfa.features)),
            ],
        ),  # fmt: skip
        Section(
            "4 · Band power",
            "The baseline the MFDFA features are held against.",
            [
                (
                    "Spectrum",
                    f"{psd.method}, {psd.window_sec:g} s segments, {psd.overlap:.0%} overlap",
                ),
                (
                    "Bands",
                    ", ".join(f"{b} {lo:g}–{hi:g} Hz" for b, (lo, hi) in psd.bands.items()),
                ),
                (
                    "Scale",
                    ("relative to " if psd.relative else "absolute, ")
                    + f"{psd.fmin:g}–{psd.fmax:g} Hz"
                    + (", log10" if psd.log else ""),
                ),
            ],
        ),
        Section(
            "5 · Models and validation",
            "Classification is exploratory. Every split is by participant, so no person is in "
            "both the training and the test part of a fold.",
            [
                (
                    "Model inputs",
                    f"{model.segments} windows, {model.level}-level features "
                    f"({', '.join(config.data.regions)})",
                ),
                (
                    "Cross-validation",
                    f"{model.cv.outer.replace('_', ' ')} with {model.cv.outer_folds} outer folds "
                    f"grouped by {model.cv.group_key}; {model.cv.inner_folds} inner folds tune the "
                    "hyperparameters",
                ),
                (
                    "Intervals",
                    f"{model.cv.confidence:.0%}, {model.cv.n_bootstrap} bootstrap "
                    "resamples of participants on the out-of-fold predictions",
                ),
                (
                    "Permutation test",
                    f"{model.cv.n_permutations} label shuffles, each repeating "
                    "the whole nested cross-validation",
                ),
                (
                    "Better than chance",
                    f"{predict.better_than_chance.p_value} below "
                    f"{predict.better_than_chance.alpha:g} and the whole 95 % interval of "
                    "balanced accuracy above 0.5",
                ),
            ],
        ),  # fmt: skip
    ]
    rows = []
    for info in evidence.values():
        low, high = info["uncertain_band"]
        rows.append(
            (
                info["title"],
                f"{info['model']}; threshold {info['threshold']:g}, no label from {low:g} to "
                f"{high:g}",
            )
        )
    if rows:
        out.append(
            Section(
                "6 · The final models",
                "Each stage is the best of the configurations compared on the same "
                "cross-validation, refitted on every eligible participant with Platt scaling. "
                f"Uncertain bands come from {predict.band.n_bootstrap} bootstrap recalibrations "
                f"of the out-of-fold scores at {predict.band.confidence:.0%} confidence.",
                rows,
            )
        )
    return out


CURATORS = (
    "We strongly advise against using any such approach because it would mislead patients and "
    "people who are interested in knowing if they have Parkinson's disease."
)
