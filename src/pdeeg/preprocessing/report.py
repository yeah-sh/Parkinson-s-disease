"""QC report for the preprocessing stage: a table of every recording, figures and outlier flags."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdeeg.config import Config, QcConfig
from pdeeg.preprocessing import qc
from pdeeg.preprocessing.pipeline import config_hash, output_paths, read_log
from pdeeg.viz.style import save

# Signal metrics compared across recordings: the column, the side on which an extreme value is
# a problem, and the reason given, formatted with the recording's row.
# Bad time is not in this list. Most recordings have none, so a distance from the median says
# little; it has fixed limits instead (see flag_outliers).
_OUTLIER_METRICS = (
    ("n_removed", "high", "{n_removed:.0f} ICA components removed"),
    (
        "variance_removed_percent",
        "high",
        "ICA removed {variance_removed_percent:.0f}% of the variance",
    ),
    ("median_sd_uv", "both", "median channel SD after cleaning is {median_sd_uv:.1f} µV"),
    (
        "max_sd_ratio",
        "high",
        "{noisiest_channel} has {max_sd_ratio:.1f} times the median channel SD",
    ),
    (
        "raw_slow_db",
        "low",
        "power below 0.5 Hz before filtering is {raw_slow_db:.0f} dB relative to 1-4 Hz, so the "
        "file was already high-passed",
    ),
    (
        "residual_hf_db",
        "high",
        "power above 30 Hz after cleaning is {residual_hf_db:.1f} dB, high for this dataset, "
        "which points to muscle activity that ICA did not remove",
    ),
)
# Lower edge of the band used for residual_hf_db; it runs up to the end of the pass band.
_HF_FROM_HZ = 30.0


def _longest_clean_stretch(segments: list[dict[str, Any]], duration: float) -> float:
    """Longest time, in seconds, between bad annotations (or the ends of the recording)."""
    longest, cursor = 0.0, 0.0
    for segment in sorted(segments, key=lambda segment: segment["onset_s"]):
        longest = max(longest, segment["onset_s"] - cursor)
        cursor = max(cursor, segment["onset_s"] + segment["duration_s"])
    return max(longest, duration - cursor)


def collect(config: Config, recordings: pd.DataFrame) -> pd.DataFrame:
    """One row of QC metrics per recording, read from the logs of the current configuration.

    A recording with no current log gets ``status`` ``missing`` and empty metrics.
    """
    digest = config_hash(config)
    classes = config.preprocessing.ica.exclude_labels
    band = config.preprocessing.filter
    hf_to_hz = band.h_freq - band.h_trans_bandwidth
    rows = []
    for recording in recordings.itertuples():
        paths = output_paths(config, recording.subject, recording.session)
        row: dict[str, Any] = {
            "subject": recording.subject,
            "session": recording.session,
            "group": recording.group,
            "label": f"sub-{recording.subject} {recording.session}",
            "status": "missing",
            "psd_figure": None,
            "ica_figure": None,
        }
        log = read_log(paths["log"])
        if log is not None and log.get("config_hash") == digest:
            removed = [component["label"] for component in log["ica"]["removed"]]
            channels = log["channels"]
            most_often_bad = max(channels, key=lambda name: channels[name]["peak_percent"])
            freqs = np.asarray(log["psd"]["freqs_hz"])
            high = (freqs >= _HF_FROM_HZ) & (freqs <= hf_to_hz)
            row |= {
                "status": "ok",
                "start_s": log["crop"]["start_s"],
                "anchor": log["crop"]["anchor"],
                "line_detected_hz": log["line_noise"]["detected_hz"],
                "line_sidecar_hz": log["line_noise"]["sidecar_hz"],
                "line_configured_hz": log["line_noise"]["configured_hz"],
                "rank": log["ica"]["rank"],
                "n_removed": len(removed),
                **{qc.count_column(name): removed.count(name) for name in classes},
                "n_brain": log["ica"]["label_counts"]["brain"],
                "variance_removed_percent": log["ica"]["variance_removed_percent"],
                "bad_percent": log["bad_segments"]["percent"],
                "longest_clean_s": _longest_clean_stretch(
                    log["bad_segments"]["segments"], log["crop"]["duration_s"]
                ),
                "worst_channel": most_often_bad,
                "worst_channel_percent": channels[most_often_bad]["peak_percent"],
                "noisiest_channel": max(channels, key=lambda name: channels[name]["sd_uv"]),
                "residual_hf_db": float(np.mean(np.asarray(log["psd"]["after_db"])[high])),
                **log["summary"],
                "psd_figure": paths["psd"],
                "ica_figure": paths["ica"] if log["figures"]["ica"] else None,
                "psd": log["psd"],
            }
        rows.append(row)
    return pd.DataFrame(rows)


def robust_z(values: pd.Series) -> pd.Series:
    """Distance from the median in robust standard deviations (MAD, scaled for a normal)."""
    deviation = values - values.median()
    spread = 1.4826 * deviation.abs().median()
    if not spread > 0:  # more than half the values are identical; fall back to the mean
        spread = 1.2533 * deviation.abs().mean()
    if not spread > 0:
        return deviation * 0.0
    return deviation / spread


def flag_outliers(table: pd.DataFrame, config: QcConfig) -> list[list[str]]:
    """For every row of :func:`collect`'s table, the reasons it should be looked at."""
    reasons: list[list[str]] = [[] for _ in range(len(table))]
    ok = (table["status"] == "ok").to_numpy()
    for i in np.flatnonzero(~ok):
        reasons[i].append("no cleaned output for the current configuration")
    done = table[ok]
    if done.empty:
        return reasons
    for i in done.index[done["bad_percent"] > config.max_bad_percent]:
        row = done.loc[i]
        reasons[i].append(
            f"{row['bad_percent']:.1f}% of the time is marked bad (limit "
            f"{config.max_bad_percent:g}%); {row['worst_channel']} is over the amplitude limit "
            f"most often, in {row['worst_channel_percent']:.1f}%"
        )
    for i in done.index[done["longest_clean_s"] < config.min_clean_stretch_s]:
        reasons[i].append(
            f"longest stretch with no bad mark is {done.at[i, 'longest_clean_s']:.0f} s "
            f"(minimum {config.min_clean_stretch_s:g} s)"
        )
    for metric, side, wording in _OUTLIER_METRICS:
        z = robust_z(done[metric].astype(float))
        extreme = {"high": z > config.outlier_z, "low": z < -config.outlier_z}.get(
            side, z.abs() > config.outlier_z
        )
        for i in done.index[extreme]:
            reasons[i].append(f"{wording.format(**done.loc[i])} (robust z {z[i]:+.1f})")
    mismatch = done["line_detected_hz"].notna() & (
        done["line_detected_hz"] != done["line_configured_hz"]
    )
    for i in done.index[mismatch]:
        reasons[i].append(
            f"mains peak at {done.at[i, 'line_detected_hz']:g} Hz, but the configuration "
            f"says {done.at[i, 'line_configured_hz']:g} Hz"
        )
    return reasons


def _link(path: Path | None, text: str, base: Path) -> str:
    if path is None:
        return ""
    return f"[{text}]({Path(os.path.relpath(path, base)).as_posix()})"


def _image(path: Path, alt: str, base: Path) -> str:
    return f"![{alt}]({Path(os.path.relpath(path, base)).as_posix()})"


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def write_report(config: Config, recordings: pd.DataFrame) -> Path:
    """Write the Markdown QC report and its summary figures; returns the report path."""
    prep = config.preprocessing
    report = prep.qc.report
    base = report.parent
    table = collect(config, recordings)
    reasons = flag_outliers(table, prep.qc)
    table["flagged"] = [bool(r) for r in reasons]
    done = table[table["status"] == "ok"].reset_index(drop=True)
    classes = list(prep.ica.exclude_labels)
    reference = prep.reference if isinstance(prep.reference, str) else ", ".join(prep.reference)

    lines = [
        "# Preprocessing QC",
        "",
        f"Written by `pdeeg preprocess`. {len(done)} of {len(table)} recordings have cleaned "
        f"output for configuration `{config_hash(config)[:12]}`; "
        f"{int(table['flagged'].sum())} are flagged below.",
        "",
        "## What was done",
        "",
        f"- **Crop:** {prep.crop.duration:g} s per recording, starting at "
        + (
            f"trigger `{prep.crop.start_event}` where the recording has it, otherwise at the "
            "file start."
            if prep.crop.start_event is not None
            else "the file start."
        ),
        f"- **Band-pass:** {prep.filter.l_freq:g} to {prep.filter.h_freq:g} Hz, zero-phase FIR.",
        f"- **Reference:** {reference}.",
        f"- **ICA:** {prep.ica.method}, as many components as the data rank, fitted on a "
        f"{prep.ica.fit_l_freq:g} to {prep.ica.fit_h_freq:g} Hz copy. Components that ICLabel "
        f"calls {', '.join(classes)} with probability of at least {prep.ica.threshold:g} are "
        "removed.",
        f"- **Bad stretches:** {prep.bad_segments.window:g} s windows with a peak-to-peak "
        f"amplitude above {prep.bad_segments.peak_to_peak_uv:g} µV or below "
        f"{prep.bad_segments.flat_uv:g} µV in any channel are annotated. Nothing is cut out.",
        "",
    ]

    if not done.empty:
        anchors = done["anchor"].value_counts()
        detected = done["line_detected_hz"].map(lambda hz: "none" if pd.isna(hz) else f"{hz:g} Hz")
        sidecar = done["line_sidecar_hz"].map(lambda hz: "none" if pd.isna(hz) else f"{hz:g} Hz")
        lines += [
            "## Crop and mains frequency",
            "",
            "- **Window start:** "
            + "; ".join(f"{count} recordings from the {name}" for name, count in anchors.items())
            + ".",
            "- **Mains peak found in the signal:** "
            + "; ".join(f"{name} in {count}" for name, count in detected.value_counts().items())
            + ".",
            "- **Mains frequency in the sidecar files:** "
            + "; ".join(f"{name} in {count}" for name, count in sidecar.value_counts().items())
            + f". The pipeline uses {config.data.dataset.line_freq:g} Hz from the configuration.",
            "",
        ]

    lines += ["## Flagged recordings", ""]
    if not table["flagged"].any():
        lines += ["None.", ""]
    else:
        lines += _markdown_table(
            ["Recording", "Why"],
            [
                [row.label, "; ".join(reasons[i])]
                for i, row in enumerate(table.itertuples())
                if reasons[i]
            ],
        )
        lines += [
            "",
            "A recording is flagged when:",
            "",
            f"- more than {prep.qc.max_bad_percent:g}% of it is marked bad;",
            f"- its longest stretch with no bad mark is shorter than "
            f"{prep.qc.min_clean_stretch_s:g} s;",
            f"- a signal metric is more than {prep.qc.outlier_z:g} robust standard deviations "
            "(median and MAD across recordings) from the median, on the side that matters. The "
            "metrics are the number of components removed, the variance ICA removed, the median "
            "channel SD, the noisiest channel relative to the median, slow power before "
            f"filtering, and power from {_HF_FROM_HZ:g} Hz to the end of the pass band after "
            "cleaning;",
            "- its mains peak disagrees with the configuration.",
            "",
            "A flag is a reason to look, not a verdict.",
            "",
        ]

    if not done.empty:
        prep.qc.figures_dir.mkdir(parents=True, exist_ok=True)
        overview = prep.qc.figures_dir / "overview.png"
        save(qc.plot_overview(done, prep), overview)
        spectra = prep.qc.figures_dir / "psd_overview.png"
        save(
            qc.plot_psd_overview(
                np.asarray(done["psd"][0]["freqs_hz"]),
                np.array([psd["before_db"] for psd in done["psd"]]),
                np.array([psd["after_db"] for psd in done["psd"]]),
                done["flagged"].to_numpy(),
                prep,
            ),
            spectra,
        )
        lines += [
            "## Overview",
            "",
            _image(overview, "Bad time, longest clean stretch and ICA components removed", base),
            "",
            _image(spectra, "Power spectra of all recordings before and after cleaning", base),
            "",
            "## All recordings",
            "",
        ]
        rows = []
        for row in done.to_dict("records"):
            detected = row["line_detected_hz"]
            figures = [
                _link(row["psd_figure"], "spectrum", base),
                _link(row["ica_figure"], "components", base),
            ]
            rows.append(
                [
                    f"**{row['label']}**" if row["flagged"] else row["label"],
                    row["group"],
                    f"{row['start_s']:.1f}",
                    "none" if pd.isna(detected) else f"{detected:g}",
                    " / ".join(str(row[qc.count_column(name)]) for name in classes),
                    str(row["n_brain"]),
                    f"{row['variance_removed_percent']:.0f}",
                    f"{row['bad_percent']:.1f}",
                    f"{row['longest_clean_s']:.0f}",
                    f"{row['median_sd_uv']:.1f}",
                    f"{row['noisiest_channel']} {row['max_sd_ratio']:.1f}",
                    " ".join(link for link in figures if link),
                ]
            )
        lines += _markdown_table(
            [
                "Recording",
                "Group",
                "Start (s)",
                "Mains (Hz)",
                "Removed: " + " / ".join(name.split()[0] for name in classes),
                "Brain components",
                "Variance removed (%)",
                "Bad time (%)",
                "Longest clean stretch (s)",
                "Median SD (µV)",
                "Noisiest channel (x median SD)",
                "Figures",
            ],
            rows,
        )
        ranks = sorted(set(done["rank"]))
        lines += [
            "",
            "Flagged recordings are in bold. Start is where the kept window begins in the "
            "original file. Brain components is how many of the ICA components ICLabel calls "
            f"brain. Data rank before ICA: {', '.join(str(rank) for rank in ranks)}.",
            "",
        ]

        flagged = done[done["flagged"]]
        if not flagged.empty:
            lines += ["## Figures for the flagged recordings", ""]
            for row in flagged.to_dict("records"):
                lines += [
                    f"### {row['label']}",
                    "",
                    _image(row["psd_figure"], "Spectrum", base),
                    "",
                ]
                if row["ica_figure"] is not None:
                    lines += [_image(row["ica_figure"], "Removed components", base), ""]

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report
