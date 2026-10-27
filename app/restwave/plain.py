"""Everything the page says in plain words: feature names, brain-wave bands, input checks.

The one place to change wording. Feature names are put together from their parts (what was
measured, of which series, where on the head), so every column a model uses has a name; the
frequency ranges and time scales in them are read from the configuration, not written here.
"""

from __future__ import annotations

import re
from typing import Any

from pdeeg.config import Config

# --- where on the head -------------------------------------------------------------------------

REGIONS = {
    "frontal": "front of head",
    "central": "top of head",
    "temporal": "sides of head",
    "parietal": "upper back of head",
    "occipital": "back of head",
}

# --- brain-wave bands ----------------------------------------------------------------------------

BAND_WORDS = {
    "delta": "slowest waves",
    "theta": "slow waves",
    "alpha": "the resting rhythm",
    "beta": "fast waves",
    "gamma": "fastest waves",
}


def hz(low: float, high: float) -> str:
    return f"{low:g}–{high:g} Hz"


def band_label(band: str, config: Config) -> str:
    """'Alpha · 8–13 Hz', for an axis."""
    low, high = config.psd.bands[band]
    return f"{band.capitalize()} · {hz(low, high)}"


# --- feature names -------------------------------------------------------------------------------

# What an MFDFA feature measures, longest name first so that "delta_alpha" is not read as "alpha".
MEASURES = {
    "delta_alpha": "Irregularity of {what}",
    "asymmetry": "Balance of calm and bursty stretches in {what}",
    "delta_h": "Spread of persistence in {what}",
    "alpha0": "Typical roughness of {what}",
    "h2": "Persistence of {what}",
}
_POWER = re.compile(r"^log_relative_power_(?P<band>[a-z]+)_(?P<where>[A-Za-z0-9]+)$")


def _where(token: str) -> str:
    return REGIONS.get(token, f"near electrode {token}")


def _series(series: str, fit_range: str, config: Config) -> str | None:
    """'beta rhythm' or 'the brain signal over 0.3–1 s', or None for an unknown series."""
    if series == "broadband":
        ranges = config.mfdfa.broadband.fit_ranges
        if fit_range not in ranges:
            return None
        low, high = ranges[fit_range]
        pace = "slow" if fit_range == max(ranges, key=lambda r: ranges[r][1]) else "fast"
        return f"the brain signal's {pace} swings ({low:g}–{high:g} s)"
    if series.startswith("envelope_"):
        band = series.removeprefix("envelope_")
        if band in config.mfdfa.envelope.bands and fit_range in config.mfdfa.envelope.fit_ranges:
            return f"{band} rhythm"
    return None


def feature_words(name: str, config: Config) -> str | None:
    """A model column in plain words, or None when its name has an unknown form.

    ``log_relative_power_beta_frontal`` -> "Share of beta waves (13–30 Hz), front of head";
    ``envelope_beta_main_h2_frontal`` -> "Persistence of beta rhythm, front of head".
    """
    power = _POWER.match(name)
    if power:
        band = power["band"]
        if band not in config.psd.bands:
            return None
        return f"Share of {band} waves ({hz(*config.psd.bands[band])}), {_where(power['where'])}"
    head, _, where = name.rpartition("_")
    for measure, template in MEASURES.items():
        if head.endswith(f"_{measure}"):
            series, _, fit_range = head.removesuffix(f"_{measure}").rpartition("_")
            what = _series(series, fit_range, config)
            if what is None:
                return None
            return f"{template.format(what=what)}, {_where(where)}"
    return None


def feature(name: str, config: Config) -> str:
    """:func:`feature_words`, falling back on the name itself."""
    return feature_words(name, config) or name.replace("_", " ")


def series_words(series: str, config: Config) -> str:
    """'the beta rhythm (13–30 Hz)' or 'the brain signal', for a caption."""
    if series in config.mfdfa.envelope.bands:
        return f"the {series} rhythm ({hz(*config.mfdfa.envelope.bands[series])})"
    return "the brain signal"


def cleaning_words(removed: int, bad_percent: float) -> str:
    """What the cleaning did, in a sentence."""
    if removed == 0:
        took = "Cleaning found no eye blinks, heartbeats or muscle noise to take out"
    else:
        sources = "source" if removed == 1 else "sources"
        took = f"Cleaning took out {removed} {sources} of noise (eye blinks, heartbeats or muscle)"
    if bad_percent < 0.5:
        return f"{took}, and no stretch was too noisy to use."
    return f"{took}, and {bad_percent:.0f}% of the time was too noisy to use."


def region_of(name: str) -> str | None:
    """The scalp region a model column belongs to, if it is a region's."""
    token = name.rpartition("_")[2]
    return token if token in REGIONS else None


# --- the input checks ----------------------------------------------------------------------------

CHIPS = ("channels", "sampling rate", "duration", "amplitude")
CHIP_TITLES = {
    "channels": "Channels",
    "sampling rate": "Sampling rate",
    "duration": "Length",
    "amplitude": "Signal level",
}


def minutes(seconds: float) -> str:
    whole = int(round(seconds))
    m, s = divmod(whole, 60)
    if not m:
        return f"{s} s"
    return f"{m} min" + (f" {s} s" if s else "")


def chip(name: str, check: dict[str, Any] | None, report: dict[str, Any], config: Config) -> str:
    """What one chip says: the check's outcome in a few words."""
    if check is None:
        return "Not checked"
    status, facts = check["status"], report.get("facts") or {}
    if name == "channels":
        if status == "failed":
            return "Electrodes missing"
        n = facts.get("n_channels")
        return f"All {n} electrodes" if n else "All electrodes"
    if name == "sampling rate":
        target = config.predict.sfreq
        return f"Resampled to {target:g} Hz" if status == "note" else f"{target:g} Hz"
    if name == "duration":
        if status == "failed":
            return "Too short"
        available = facts.get("available_s")
        return minutes(available) if available else "Long enough"
    if name == "amplitude":
        return "Out of range" if status == "failed" else "Normal"
    return check["detail"]


def problem(report: dict[str, Any], config: Config) -> tuple[str, str]:
    """(What is wrong, how to fix it) for a file that failed its checks, in plain words."""
    failed = next((c for c in report["checks"] if c["status"] == "failed"), None)
    name = failed["name"] if failed else "file"
    error = report["error"] or ""
    needed = config.preprocessing.crop.duration
    if name == "channels":
        found = re.search(r"missing: ([^.]+)\.", error)
        which = f" ({found.group(1)})" if found else ""
        return (
            f"Some electrodes are missing{which}.",
            f"The analysis needs all electrodes of the {config.data.dataset.montage} cap, named "
            "as usual (Fp1, AF3, Cz …). Older names such as T3 and an “EEG ” prefix are fine. "
            "Export the recording again with every scalp channel.",
        )
    if name == "duration":
        found = re.search(r"([\d.]+) s of EEG", error)
        have = f": it has {minutes(float(found.group(1)))} of EEG" if found else ""
        return (
            f"This recording is too short{have}, and the analysis needs {minutes(needed)}.",
            f"Upload a longer resting-state recording, at least {minutes(needed)} after the "
            "start marker.",
        )
    if name == "amplitude":
        found = re.search(r"amplitude is ([^ ]+) µV", error)
        level = f" ({found.group(1)} µV)" if found else ""
        return (
            f"The signal is far too large or too small{level}.",
            "This usually means the units in the file are wrong (volts saved as microvolts, or "
            "the reverse). Check the export settings and upload the file again.",
        )
    return (
        "We could not open this file as an EEG recording.",
        "Upload a .bdf or .edf file. Most EEG software can export one.",
    )
