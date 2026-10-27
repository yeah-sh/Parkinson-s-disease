"""The research report: one self-contained, printable HTML file.

It holds what the app shows, from the same sources: the checks (pdeeg.predict.inputs), the
cleaning log and QC figures of the pipeline, the MFDFA of pdeeg.demo drawn with the project's
report figures (pdeeg.viz), the result of predict_files and the saved models' evidence.
Nothing in it is computed here except the layout. Uploaded names and messages are escaped.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import pdeeg
from neurolab.engine import Analysis, Described
from neurolab.methodology import CURATORS, sections
from pdeeg import demo
from pdeeg.config import Config
from pdeeg.features.scaling import BROADBAND
from pdeeg.viz.demo import plot_fluctuations, plot_spectrum
from pdeeg.viz.predict import plot_contributions
from pdeeg.viz.stats import Topography, montage_info, plot_topographies
from pdeeg.viz.style import ACCENT, MUTED, new_figure, styled

SIGNED = ("delta_h", "asymmetry")
TITLES = {
    "stage1": "Stage 1: Parkinson's vs healthy",
    "stage2_single": "Stage 2: OFF vs ON medication",
    "stage2_paired": "Stage 2, paired: which recording is OFF medication",
}
ENDS = {"stage1": ("healthy", "Parkinson's"), "stage2_single": ("ON medication", "OFF medication")}

_CSS = """
:root { color-scheme: light; }
body { font: 14px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif; color: #14181f;
  background: #f6f7f9; margin: 0; }
main { max-width: 980px; margin: 0 auto; padding: 32px 40px 64px; background: #fff; }
h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 34px 0 10px; padding-top: 14px; border-top: 1px solid #e3e6eb; }
h3 { font-size: 14px; margin: 18px 0 6px; }
.eyebrow { font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #0e7490;
  font-weight: 700; }
.muted { color: #5b6473; }
.box { border: 1px solid #f0c36d; background: #fff8e8; border-left: 4px solid #d9930d;
  padding: 10px 14px; border-radius: 6px; margin: 14px 0; }
.weak { border: 1px solid #f0c36d; background: #fff8e8; padding: 8px 12px; border-radius: 6px; }
.ok { border: 1px solid #b6dce6; background: #f0f9fb; padding: 8px 12px; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 6px 0 10px; }
th, td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e8eaee;
  vertical-align: top; }
th { color: #5b6473; font-weight: 600; background: #fafbfc; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 6px 0; }
dt { color: #5b6473; } dd { margin: 0; overflow-wrap: anywhere; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0 4px; }
.cap { font-size: 12px; color: #5b6473; margin: 0 0 12px; }
.failed { color: #b42318; font-weight: 600; } .note { color: #a15c07; font-weight: 600; }
.passed { color: #0e7490; font-weight: 600; }
@media print { body { background: #fff; } main { padding: 0; max-width: none; }
  h2 { break-after: avoid; } img, table { break-inside: avoid; } }
"""


@dataclass(frozen=True)
class Upload:
    name: str
    size: int


def _png(figure: Figure, alt: str) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    return _img(buffer.getvalue(), alt)


def _img(data: bytes, alt: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{escape(alt)}">'


def _dl(rows: Sequence[tuple[str, str]]) -> str:
    return "<dl>" + "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows) + "</dl>"


def _table(
    header: Sequence[str], rows: Sequence[Sequence[Any]], numeric: Sequence[int] = ()
) -> str:
    head = "".join(
        f'<th class="num">{escape(h)}</th>' if i in numeric else f"<th>{escape(h)}</th>"
        for i, h in enumerate(header)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{escape(str(c))}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


@styled
def waveform_figure(
    described: Described, seconds: float, title: str, n_channels: int = 16
) -> Figure:
    """The first ``seconds`` of the analysed window, before (grey) and after (blue) cleaning."""
    sfreq = described.series.sfreq
    channels = described.series.channels[:n_channels]
    stop = int(round(seconds * sfreq))
    before = described.before_uv[:n_channels, :stop].astype(float)
    cleaned = described.cleaned_uv[:n_channels, :stop].astype(float)
    before -= before.mean(axis=1, keepdims=True)
    cleaned -= cleaned.mean(axis=1, keepdims=True)
    spacing = 5 * float(np.median(cleaned.std(axis=1))) or 50.0
    figure = new_figure(
        7.2,
        0.32 * len(channels) + 1.4,
        title,
        f"First {seconds:g} s of the analysed window, {len(channels)} channels, "
        f"{spacing:.0f} µV between rows. Grey: before cleaning; blue: cleaned.",
    )
    axes = figure.subplots()
    time = np.arange(stop) / sfreq
    for row in range(len(channels)):
        offset = (len(channels) - 1 - row) * spacing
        axes.plot(time, before[row] + offset, color=MUTED, lw=0.6)
        axes.plot(time, cleaned[row] + offset, color=ACCENT, lw=0.7)
    axes.set_yticks([(len(channels) - 1 - i) * spacing for i in range(len(channels))], channels)
    axes.set_xlim(0, seconds)
    axes.set_xlabel("Time in the analysed window (s)")
    axes.grid(axis="y", visible=False)
    axes.tick_params(axis="y", length=0, labelsize=7)
    return figure


def _checks(report: Mapping[str, Any]) -> str:
    rows = [
        [escape(c["name"]), f'<span class="{c["status"]}">{escape(c["status"])}</span>',
         escape(c["detail"])]
        for c in report["checks"]
    ]  # fmt: skip
    verdict = (
        '<p class="passed">Every check passed.</p>'
        if report["error"] is None
        else f'<p class="failed">Refused: {escape(report["error"])}</p>'
    )
    return _table(["Check", "Result", "What was found"], rows) + verdict


def _cleaning(described: Described, config: Config) -> str:
    log = described.log
    prep = config.preprocessing
    crop, ica, bad, line = log["crop"], log["ica"], log["bad_segments"], log["line_noise"]
    mains = "none detected" if line["detected_hz"] is None else f"{line['detected_hz']:g} Hz"
    rows = [
        ("Window", f"{crop['duration_s']:g} s from {crop['anchor']} ({crop['start_s']:.1f} s in)"),
        ("Band-pass", f"{prep.filter.l_freq:g}–{prep.filter.h_freq:g} Hz"),
        ("Mains peak", mains),
        (
            "ICA",
            f"{ica['n_components']} components; removed {len(ica['removed'])} "
            f"({ica['variance_removed_percent']:.1f} % of the variance)",
        ),
        (
            "Bad windows",
            f"{bad['percent']:.1f} % of the time ({bad['peak_percent']:.1f} % too large, "
            f"{bad['flat_percent']:.1f} % flat), {len(bad['segments'])} stretch(es)",
        ),
        ("Median channel SD", f"{log['summary']['median_sd_uv']:.2f} µV after cleaning"),
    ]
    html = _dl(rows)
    if ica["removed"]:
        html += _table(
            ["Component", "ICLabel class", "Probability"],
            [[f"IC{c['index']:02d}", escape(c["label"]), f"{c['probability']:.2f}"]
             for c in ica["removed"]],
            numeric=(2,),
        )  # fmt: skip
    for kind, caption in (
        ("psd", "Spectrum before and after cleaning (the pipeline's QC figure)."),
        ("ica", "Scalp maps of the components ICA removed (the pipeline's QC figure)."),
    ):
        if kind in described.qc:
            html += _img(described.qc[kind], caption) + f'<p class="cap">{escape(caption)}</p>'
    return html


def _mfdfa(described: Described, config: Config, choice: tuple[str, str, str]) -> str:
    name, fit_range, channel = choice
    label = "broadband EEG" if name == BROADBAND else f"{name} envelope"
    ranges = demo.ranges_in_seconds(name, config)
    scales, qs, fq = demo.fluctuation_functions(described.series, name, channel, config)
    html = _png(plot_fluctuations(scales, qs, fq, ranges, f"{label}, {channel}"), "Fq(s)")
    curves = demo.spectrum(described.series, name, fit_range, channel, config)
    if curves["h"].isna().all():
        html += '<p class="cap">Too few clean windows on this range to fit h(q).</p>'
    else:
        html += _png(
            plot_spectrum(curves, f"{label}, {channel}, {fit_range} range"), "h(q) and f(alpha)"
        )
    table = demo.channel_features(described.series, name, fit_range, config)
    section = config.mfdfa.broadband if name == BROADBAND else config.mfdfa.envelope
    panels = []
    for feature in section.wide_features[fit_range]:
        values = table[feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if feature in SIGNED:
            top = float(np.abs(finite).max()) if finite.size else 1.0
            panels.append(Topography(values, feature, "t", (-top, top)))
        else:
            limits = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            panels.append(Topography(values, feature, "mean", limits))
    info = montage_info(described.series.channels, config.data.dataset.montage)
    html += _png(
        plot_topographies(
            info,
            [("", panels)],
            f"{label}, {fit_range} range",
            "One value per channel, bad windows left out. Each map has its own scale; delta_h "
            "and asymmetry are centred on 0.",
        ),
        "Scalp maps",
    )
    columns = ["channel", *config.mfdfa.features, "min_r2", "n_scales"]
    rows = [
        [escape(str(r["channel"]))]
        + [("NaN" if pd.isna(r[c]) else f"{r[c]:.4f}") for c in columns[1:-1]]
        + [str(int(r["n_scales"]))]
        for _, r in table.iterrows()
    ]
    html += f"<h3>Features of every channel ({escape(label)}, {escape(fit_range)} range)</h3>"
    html += _table(columns, rows, numeric=tuple(range(1, len(columns))))
    return html


def _stage(stage: Mapping[str, Any], names: list[str]) -> str:
    name = stage["stage"]
    where = stage.get("recording") or ", ".join(stage.get("recordings", [])) or "the pair"
    html = f"<h3>{escape(TITLES[name])} · {escape(where)}</h3>"
    if "skipped" in stage:
        return html + f"<p>Not run: {escape(stage['skipped'])}.</p>"
    cv = stage["cross_validated"]
    low, high = stage["uncertain_band"]
    kind = "out of fold (participant held out)" if stage.get("out_of_fold") else (
        "final model on a recording not recognised as training data")  # fmt: skip
    rows = [
        ("Result", "uncertain: no label" if stage["uncertain"] else stage["label"]),
        ("Model score", f"{stage['probability']:.3f} (not a probability of disease)"),
        ("Threshold and band", f"{stage['threshold']:g}; no label from {low:g} to {high:g}"),
        ("Kind of prediction", kind),
    ]
    if stage.get("out_of_fold"):
        rows.append(("True label", f"{stage['true_label']} (fold {stage['fold']})"))
    rows += [
        (
            "Cross-validated balanced accuracy",
            f"{cv['balanced_accuracy']:.2f} [{cv['ci95'][0]:.2f}, {cv['ci95'][1]:.2f}], chance "
            f"{cv['chance']:g}",
        ),
        ("Permutation p / q", f"{cv['permutation_p']:.3f} / {cv['q']:.3f}"),
        (
            "Split",
            f"by participant: {cv['n_participants']} people, {cv['n_recordings']} recordings",
        ),
    ]
    html += _dl(rows)
    if cv["better_than_chance"]:
        html += (
            f'<p class="ok">Better than chance by the project\'s rule: {escape(cv["test"])}.</p>'
        )
    else:
        html += (
            '<p class="weak"><b>Not better than chance</b>: '
            f"{escape('; '.join(cv['not_better_because']))}. Its output should not be read as "
            "saying anything about this recording.</p>"
        )
    html += f'<p class="cap">{escape(cv["note"])}</p>'
    if "top_features" in stage:
        ends = (
            (f"{names[1]} is OFF", f"{names[0]} is OFF") if name == "stage2_paired" else ENDS[name]
        )
        html += _png(
            plot_contributions(
                stage["top_features"],
                ends,
                f"{TITLES[name]}: {where}",
                greyed=not cv["better_than_chance"],
            ),
            "Features that moved the score",
        )
    return html


def build(
    uploads: Sequence[Upload],
    checks: Mapping[str, Any],
    analysis: Analysis,
    evidence: Mapping[str, Mapping[str, Any]],
    config: Config,
    choice: tuple[str, str, str],
    generated: str,
) -> str:
    """The report of one analysed upload. ``choice`` is the (series, fit range, channel) whose
    MFDFA figures are included."""
    names = [u.name for u in uploads]
    parts = [
        '<p class="eyebrow">NeuroLab · pdeeg research report</p>',
        f"<h1>{escape(' + '.join(names))}</h1>",
        f'<p class="muted">Generated {escape(generated)} with pdeeg {escape(pdeeg.__version__)}; '
        f"configuration {escape(str(config.config_dir))}.</p>",
        f'<div class="box"><b>{escape(config.predict.disclaimer)}.</b> A research '
        "demonstration on one public dataset; nothing here is validated for deciding whether a "
        "person has Parkinson's disease or is on medication.</div>",
        "<h2>1. Recording</h2>",
    ]
    result = analysis.result
    for index, (upload, report) in enumerate(zip(uploads, checks["reports"], strict=True)):
        header = report.get("header") or {}
        rows = [("File", upload.name), ("Size", f"{upload.size / 1e6:.2f} MB"),
                ("Format", report["format"])]  # fmt: skip
        if result:
            rows.append(("SHA-256", result["inputs"][index]["sha256"]))
        if header:
            rows += [
                ("Channels in the file", str(header["n_channels"])),
                ("Sampling rate", f"{header['sfreq']:g} Hz"),
                ("Length", f"{header['duration_s']:.1f} s"),
            ]
        parts += [f"<h3>{escape(upload.name)}</h3>", _dl(rows)]
    parts.append("<h2>2. Input checks</h2>")
    for report in checks["reports"]:
        parts += [f"<h3>{escape(report['name'])}</h3>", _checks(report)]
    parts.append("<h2>3. Preprocessing</h2>")
    for name, described in zip(names, analysis.described, strict=True):
        parts.append(f"<h3>{escape(name)}</h3>")
        parts.append(
            "<p>Could not be cleaned.</p>" if described is None else _cleaning(described, config)
        )
    parts.append("<h2>4. Signal</h2>")
    for name, described in zip(names, analysis.described, strict=True):
        if described is not None:
            parts.append(_png(waveform_figure(described, 10.0, name), f"EEG of {name}"))
    parts.append("<h2>5. Multifractal analysis</h2>")
    for name, described in zip(names, analysis.described, strict=True):
        if described is not None:
            parts += [f"<h3>{escape(name)}</h3>", _mfdfa(described, config, choice)]
    parts.append("<h2>6. Model evaluation</h2>")
    if result is None:
        parts.append(f"<p>No prediction: {escape(analysis.error or '')}</p>")
    else:
        if result["training_recording"]:
            parts.append(f'<p class="weak">{escape(result["explanation"])}</p>')
        else:
            parts.append(
                "<p>Kind of prediction: the final models on a recording not recognised as one "
                "of their training recordings (by file hash, signal hash or BIDS name). This is "
                "not an external validation: one recording says nothing about how well a model "
                "works.</p>"
            )
        parts += [_stage(stage, names) for stage in result["stages"]]
    parts.append("<h2>7. Methodology</h2>")
    for part in sections(config, dict(evidence)):
        parts += [
            f"<h3>{escape(part.title)}</h3>",
            f"<p>{escape(part.summary)}</p>",
            _dl(part.rows),
        ]
    parts.append("<h2>8. Limitations</h2>")
    stage1 = evidence.get("stage1")
    limits = stage1["model_card"]["limitations"] if stage1 else []
    parts.append(
        "<ul>"
        + "".join(f"<li>{escape(item)}</li>" for item in limits)
        + "<li>A model score is not the probability that a person has Parkinson's disease or "
        "is off medication.</li><li>No external test set: every evaluation number comes from "
        "cross-validation within one dataset.</li></ul>"
    )
    parts.append(f"<p>The dataset's curators: “{escape(CURATORS)}”</p>")
    parts.append(f'<div class="box"><b>{escape(config.predict.disclaimer)}.</b></div>')
    title = f"NeuroLab report: {' + '.join(names)}"
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main>{''.join(parts)}</main></body></html>"
    )
