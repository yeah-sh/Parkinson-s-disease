"""Report of the scaling inspection: figures, local slopes, and a check of the fit ranges.

The report states what was measured. Which ranges to fit is a decision; it and its reasons are
written in ``configs/features/mfdfa.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pdeeg.config import Config, MfdfaConfig
from pdeeg.features.scaling import (
    BROADBAND,
    Inspection,
    fit_ranges,
    local_slopes,
    null_name,
    scales_in_range,
    series_length,
)
from pdeeg.viz.scaling import plot_fluctuations, plot_local_slopes
from pdeeg.viz.style import save

# Slope range drawn for the broadband series and for the envelopes. Slopes of the shortest
# scales lie above it and run out of the frame.
_YLIM_BROADBAND = (0.0, 2.2)
_YLIM_ENVELOPE = (0.3, 1.5)


def _title(name: str, config: MfdfaConfig) -> str:
    if name == BROADBAND:
        return "Broadband EEG"
    low, high = config.envelope.bands[name]
    return f"{name.capitalize()} envelope ({low:g}-{high:g} Hz)"


def _image(path: Path, alt: str, base: Path) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, base)).as_posix()})"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _spread(values: pd.Series) -> str:
    low, median, high = np.percentile(values, [10, 50, 90])
    return f"{median:.2f} [{low:.2f}, {high:.2f}]"


def _slope_table(inspection: Inspection, name: str, show_qs: list[float], half_width: int):
    """Local slopes of one series at every scale: the EEG, then each noise."""
    series = inspection.curves[name]
    index = [list(inspection.qs).index(q) for q in show_qs]
    slopes = local_slopes(series.scales, series.fq, half_width)
    median = np.median(slopes, axis=(0, 1))
    low, high = np.percentile(slopes[:, :, index[1]], [10, 90], axis=(0, 1))
    noises = {
        noise: local_slopes(series.scales, values, half_width).mean(axis=0)
        for noise, values in series.null.items()
    }
    header = ["s (s)", "samples", *(f"EEG q={q:g}" for q in show_qs), "EEG q=2, 10-90 %"]
    header += [f"{noise} q={q:g}" for noise in noises for q in show_qs]
    rows = []
    for j in np.flatnonzero(np.isfinite(median[index[1]])):
        row = [f"{series.scales[j] / inspection.sfreq:.3g}", str(series.scales[j])]
        row += [f"{median[k, j]:.2f}" for k in index]
        row.append(f"{low[j]:.2f} to {high[j]:.2f}")
        row += [f"{values[k, j]:.2f}" for values in noises.values() for k in index]
        rows.append(row)
    return _markdown_table(header, rows)


def _fit_table(inspection: Inspection, config: MfdfaConfig) -> list[str]:
    """Fits over the configured ranges, summarised per series, range and source."""
    header = [
        "Series",
        "Fit range",
        "Scales (samples)",
        "Source",
        "Fits",
        f"h({config.q_min:g})",
        "h(2)",
        f"h({config.q_max:g})",
        "delta_alpha",
        "min R²",
        f"min R² < {config.poor_fit_r2:g}",
    ]
    rows = []
    for name in inspection.curves:
        n_samples = series_length(name, inspection.n_times, inspection.sfreq, config)
        for range_name, fit_range in fit_ranges(config, name).items():
            scales = scales_in_range(fit_range, inspection.sfreq, config.scales_per_range)
            fits = inspection.fits
            fits = fits[(fits["series"] == name) & (fits["fit_range"] == range_name)]
            for source, group in fits.groupby("source", sort=False):
                rows.append(
                    [
                        name,
                        f"{range_name}: {fit_range[0]:g} to {fit_range[1]:g} s",
                        f"{scales[0]} to {scales[-1]} ({len(scales)}; series {n_samples})",
                        source,
                        str(len(group)),
                        _spread(group["h_q_min"]),
                        _spread(group["h2"]),
                        _spread(group["h_q_max"]),
                        _spread(group["delta_alpha"]),
                        _spread(group["min_r2"]),
                        f"{100 * (group['min_r2'] < config.poor_fit_r2).mean():.0f} %",
                    ]
                )
    return _markdown_table(header, rows)


def write_report(config: Config, inspection: Inspection, n_recordings_total: int) -> Path:
    """Write the Markdown report and its figures; returns the report path.

    ``n_recordings_total`` is the number of recordings the inspected ones were picked from.
    """
    mfdfa_config = config.mfdfa
    settings = mfdfa_config.inspection
    report = settings.report
    base = report.parent
    n_recordings = next(iter(inspection.curves.values())).fq.shape[0]
    qs = [float(q) for q in inspection.qs]
    show_qs = [qs[0], 2.0, qs[-1]]
    titles = {name: _title(name, mfdfa_config) for name in inspection.curves}
    ranges = {name: fit_ranges(mfdfa_config, name) for name in inspection.curves}
    seconds = inspection.n_times / inspection.sfreq
    prep = config.preprocessing.filter
    noises = ", ".join(null_name(beta) for beta in settings.null_exponents)

    slopes_figure = settings.figures_dir / "local_slopes.png"
    save(
        plot_local_slopes(
            inspection.curves,
            inspection.sfreq,
            qs,
            show_qs,
            settings.slope_half_width,
            ranges,
            titles,
            {
                name: _YLIM_BROADBAND if name == BROADBAND else _YLIM_ENVELOPE
                for name in inspection.curves
            },
            f"Local slope of log Fq(s) against log s: {n_recordings} recordings, "
            f"{len(inspection.channels)} channels each",
            "A power law is a level stretch. Slopes above the frame belong to the shortest "
            "scales. The thin horizontal line is 0.5, the slope of a series without memory.",
        ),
        slopes_figure,
    )
    fluctuation_figures = {}
    for name, series in inspection.curves.items():
        fluctuation_figures[name] = settings.figures_dir / f"fq_{name}.png"
        save(
            plot_fluctuations(
                series,
                inspection.sfreq,
                qs,
                list(inspection.channels),
                list(settings.channels),
                ranges[name],
                f"{titles[name]}: fluctuation functions Fq(s)",
                f"One row per recording, one column per channel. Detrending order "
                f"{mfdfa_config.detrend_order}. The lowest curve is q = {qs[0]:g}, the highest "
                f"q = {qs[-1]:g}.",
            ),
            fluctuation_figures[name],
        )

    envelope = mfdfa_config.envelope
    bands = envelope.bands.items()
    lines = [
        "# MFDFA scaling inspection",
        "",
        f"Written by `pdeeg mfdfa-scaling`. {n_recordings} of the {n_recordings_total} "
        f"recordings were picked at random (seed {settings.seed}) and are called R1 to "
        f"R{n_recordings}. No subject, session or group label was read to choose them or appears "
        "below, so nothing here can be tuned towards a group difference.",
        "",
        "This report states what was measured. The fit ranges chosen from it, and the reasons, "
        "are in `configs/features/mfdfa.yaml`; the shaded bands in the figures are those ranges.",
        "",
        "## What was computed",
        "",
        f"- **Recordings:** {seconds:g} s at {inspection.sfreq:g} Hz ({inspection.n_times} "
        f"samples), {len(inspection.channels)} channels, whole recording including the stretches "
        "marked bad.",
        f"- **Broadband:** the cleaned EEG as it is ({prep.l_freq:g} to {prep.h_freq:g} Hz "
        "band-pass from preprocessing).",
        "- **Envelope:** "
        + "; ".join(f"{band} {low:g} to {high:g} Hz" for band, (low, high) in bands)
        + f". Zero-phase FIR band-pass with {envelope.trans_bandwidth:g} Hz transitions "
        f"({3.3 / envelope.trans_bandwidth:.2f} s long), modulus of the analytic signal, "
        f"{envelope.edge_trim:g} s dropped from each end.",
        f"- **MFDFA:** detrending order {mfdfa_config.detrend_order}, "
        f"q = {', '.join(f'{q:g}' for q in qs)}, {settings.n_scales} log-spaced window sizes from "
        f"{settings.broadband_scale_min:g} s (broadband) or {settings.envelope_scale_min:g} s "
        f"(envelope) to {mfdfa_config.scale_max_frac:g} of the series.",
        f"- **Local slope:** least-squares slope of log Fq against log s through "
        f"{2 * settings.slope_half_width + 1} neighbouring scales.",
        f"- **Filtered noise:** {settings.null_realisations} realisations each. For broadband, "
        f"{noises} put through the preprocessing band-pass; unfiltered, such noise has the slope "
        "(beta + 1) / 2 at every scale and every q, so any departure is the filter's doing. For "
        "an envelope, white noise put through the same band-pass and Hilbert transform; it has "
        "no memory of its own, so any slope above 0.5 is the filter's doing.",
        "",
        "## Local slopes",
        "",
        _image(slopes_figure, "Local slopes of every series", base),
        "",
    ]
    for name in inspection.curves:
        lines += [
            f"### {titles[name]}",
            "",
            "Median over every channel of every recording; for the noises, the mean over "
            "realisations.",
            "",
            *_slope_table(inspection, name, show_qs, settings.slope_half_width),
            "",
        ]
    lines += [
        "## Fits over the configured ranges",
        "",
        f"h(q) fitted for q from {mfdfa_config.q_min:g} to {mfdfa_config.q_max:g} in steps of "
        f"{mfdfa_config.q_step:g} on {mfdfa_config.scales_per_range} scales per range. Each cell "
        "is the median with the 10th and 90th percentile in brackets, over the channels of all "
        "recordings (EEG) or over the realisations (noise). min R² is the worst R² over q.",
        "",
        *_fit_table(inspection, mfdfa_config),
        "",
        "## Fluctuation functions",
        "",
    ]
    for name, path in fluctuation_figures.items():
        lines += [_image(path, f"Fluctuation functions, {titles[name]}", base), ""]

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report
