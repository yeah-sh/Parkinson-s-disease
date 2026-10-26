"""A recording given to ``pdeeg predict``: read, checked and put in the form training used.

Nothing is computed from a recording before :func:`read_recording` has accepted it. It refuses
(:class:`InputError`, with the reason) a file it cannot read, EEG channels that do not cover the
montage, and a recording too short for the crop or with implausible amplitudes. It resamples a
recording sampled at another rate, with a note.

A recording is also fingerprinted, so that a recording the final models were trained on can be
recognised (:func:`identify_training`): by its BIDS subject and session in the training
dataset, by the SHA-256 of its file, or by a hash of its signal, which survives renaming the
file or editing its header.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from pdeeg.config import Config
from pdeeg.preprocessing.pipeline import crop_window
from pdeeg.stages import file_digest

READERS = {".bdf": mne.io.read_raw_bdf, ".edf": mne.io.read_raw_edf}
# A reference or montage suffix some systems add to channel names, and a type prefix.
_SUFFIX = re.compile(r"[-_ ](ref|le|avg|ar|cz)$", re.IGNORECASE)
_PREFIX = re.compile(r"^eeg[-_ ]?", re.IGNORECASE)
_BIDS_NAME = re.compile(r"^sub-([^_]+)_ses-([^_]+)_.*_eeg\.[a-z0-9]+$", re.IGNORECASE)
# The signal hash rounds to 0.01 µV: below what any EEG system resolves, above float noise.
_HASH_VOLTS = 1e-8
# How much of the recording the amplitude check looks at, from the crop start.
_AMPLITUDE_S = 60.0


class InputError(Exception):
    """The recording cannot be used as it is; the message says why."""


@dataclass
class Recording:
    """A recording that passed the checks.

    ``raw`` holds the scalp channels with the montage's names, order and positions, loaded and
    at the configured sampling rate. ``notes`` says what was changed to get there.
    ``available_s`` is the length from the crop start. ``bids`` holds ``subject``,
    ``session``, ``root``, ``bids_path`` and ``doi`` when the file sits in a BIDS dataset.
    """

    path: Path
    raw: mne.io.BaseRaw
    file_sha256: str
    signal_sha256: str
    available_s: float
    median_amplitude_uv: float
    notes: list[str] = field(default_factory=list)
    bids: dict[str, str] | None = None


def signal_digest(raw: mne.io.BaseRaw, montage: str) -> str:
    """SHA-256 of the montage channels' signal, in montage order, rounded to 0.01 µV."""
    names = mne.channels.make_standard_montage(montage).ch_names
    data = raw.get_data(picks=names)
    quantised = np.round(data / _HASH_VOLTS).astype(np.int64)
    digest = hashlib.sha256(str(quantised.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(quantised).tobytes())
    return digest.hexdigest()


def bids_entities(path: Path) -> dict[str, str] | None:
    """Subject, session, dataset root and DOI of a file in a BIDS dataset, else None."""
    match = _BIDS_NAME.match(path.name)
    if match is None:
        return None
    for root in list(path.resolve().parents)[:4]:
        description = root / "dataset_description.json"
        if description.is_file():
            try:
                doi = json.loads(description.read_text(encoding="utf-8")).get("DatasetDOI", "")
            except (OSError, ValueError):
                doi = ""
            return {
                "subject": match.group(1),
                "session": match.group(2),
                "root": str(root),
                "bids_path": path.resolve().relative_to(root).as_posix(),
                "doi": str(doi),
            }
    return None


def _read(path: Path, bids: dict[str, str] | None) -> mne.io.BaseRaw:
    if bids is not None:
        from mne_bids import get_bids_path_from_fname, read_raw_bids

        bids_path = get_bids_path_from_fname(path).update(root=bids["root"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return read_raw_bids(bids_path, verbose="error")
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise InputError(
            f"{path.name}: expected a .bdf or .edf file, or the EEG file of a BIDS dataset"
        )
    raw = reader(path, preload=False, verbose="error")
    stim = mne.pick_types(raw.info, eeg=False, stim=True)
    if stim.size and not len(raw.annotations):
        # Triggers as annotations, named by their value, as a BIDS events.tsv gives them: the
        # crop starts at the configured trigger in both.
        events = mne.find_events(
            raw, stim_channel=raw.ch_names[stim[0]], shortest_event=1, verbose="error"
        )
        if len(events):
            raw.set_annotations(
                mne.Annotations(
                    onset=(events[:, 0] - raw.first_samp) / raw.info["sfreq"],
                    duration=0.0,
                    description=[str(value) for value in events[:, 2]],
                )
            )
    return raw


def map_channels(
    names: list[str], montage_names: list[str], aliases: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Which EEG channel is which montage channel, and the channels that are none of them.

    Names are compared ignoring case, an "EEG" prefix and a reference suffix such as "-REF";
    ``aliases`` maps older names (T3) to the montage's (T7). Raises :class:`InputError` when a
    montage channel is missing or two channels claim the same one.
    """
    by_lower = {name.lower(): name for name in montage_names}
    alias_lower = {key.lower(): value for key, value in aliases.items()}
    unknown = sorted(set(aliases.values()) - set(montage_names))
    if unknown:
        raise InputError(f"predict.channel_aliases: {', '.join(unknown)} are not montage channels")
    mapping, ignored, claimed = {}, [], {}
    for name in names:
        bare = _SUFFIX.sub("", _PREFIX.sub("", name.strip())).lower()
        standard = by_lower.get(bare) or alias_lower.get(bare)
        if standard is None:
            ignored.append(name)
            continue
        if standard in claimed:
            raise InputError(
                f"channels {claimed[standard]!r} and {name!r} would both be {standard}; "
                "rename one of them"
            )
        claimed[standard] = name
        mapping[name] = standard
    missing = [name for name in montage_names if name not in claimed]
    if missing:
        raise InputError(
            f"{len(missing)} of the {len(montage_names)} montage channels are missing: "
            f"{', '.join(missing)}. The models need every one of them, named as in the "
            "montage (an 'EEG ' prefix, a '-REF' suffix and the old names T3-T6 are accepted)"
        )
    return mapping, ignored


def median_amplitude_uv(raw: mne.io.BaseRaw, start: int) -> float:
    """Median over channels of the standard deviation after a 1-40 Hz band-pass, in µV."""
    sfreq = raw.info["sfreq"]
    stop = min(raw.n_times, start + int(_AMPLITUDE_S * sfreq))
    data = raw.get_data(start=start, stop=stop)
    sos = butter(4, [1.0, min(40.0, 0.45 * sfreq)], btype="bandpass", fs=sfreq, output="sos")
    return float(np.median(sosfiltfilt(sos, data, axis=1).std(axis=1)) * 1e6)


def read_recording(path: str | Path, config: Config) -> Recording:
    """Read ``path`` and check it, before anything else is done with it."""
    path = Path(path)
    if not path.is_file():
        raise InputError(f"{path}: no such file")
    bids = bids_entities(path)
    try:
        raw = _read(path, bids)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"{path.name}: cannot be read as EEG ({exc})") from exc
    notes = []
    montage = config.data.dataset.montage
    layout = mne.channels.make_standard_montage(montage)
    kinds = raw.get_channel_types()
    eeg = [name for name, kind in zip(raw.ch_names, kinds, strict=True) if kind == "eeg"]
    mapping, ignored = map_channels(eeg, layout.ch_names, config.predict.channel_aliases)
    if ignored:
        notes.append(f"ignored EEG channels that are not in the montage: {', '.join(ignored)}")
    renamed = {old: new for old, new in mapping.items() if old != new}
    if renamed:
        notes.append(
            "renamed channels: " + ", ".join(f"{old} -> {new}" for old, new in renamed.items())
        )
    raw.pick(list(mapping))
    raw.rename_channels(renamed)
    raw.reorder_channels(layout.ch_names)
    raw.load_data(verbose="error")
    raw.set_montage(layout)
    signal = signal_digest(raw, montage)

    sfreq, target = float(raw.info["sfreq"]), config.predict.sfreq
    if not np.isclose(sfreq, target):
        notes.append(
            f"resampled from {sfreq:g} Hz to {target:g} Hz: the models were trained on "
            f"{target:g} Hz recordings, and features of a resampled recording can differ"
        )
        raw.resample(target, verbose="error")

    crop = config.preprocessing.crop
    start, anchor = crop_window(raw, crop)
    available = (raw.n_times - start) / raw.info["sfreq"]
    if available < crop.duration:
        raise InputError(
            f"{path.name}: {available:.1f} s of EEG from the {anchor}; at least "
            f"{crop.duration:g} s are needed"
        )
    amplitude = median_amplitude_uv(raw, start)
    low, high = config.predict.amplitude_uv
    if not low <= amplitude <= high:
        raise InputError(
            f"{path.name}: the median channel amplitude is {amplitude:.3g} µV (1-40 Hz), "
            f"outside {low:g}-{high:g} µV; are the units in the file right?"
        )
    if anchor == "file start" and crop.start_event is not None:
        notes.append(
            f"no trigger {crop.start_event!r}: the {crop.duration:g} s analysed start at the "
            "beginning of the file"
        )
    return Recording(
        path=path,
        raw=raw,
        file_sha256=file_digest(path),
        signal_sha256=signal,
        available_s=float(available),
        median_amplitude_uv=amplitude,
        notes=notes,
        bids=bids,
    )


# --- recordings the final models were trained on ------------------------------------------------


REGISTRY_COLUMNS = ("subject", "session", "bids_path", "file_sha256", "signal_sha256")


def identify_training(
    recording: Recording, registry: pd.DataFrame, config: Config
) -> dict[str, Any] | None:
    """The training recording ``recording`` is, and how it was recognised, or None.

    ``registry`` lists the training recordings (:data:`REGISTRY_COLUMNS`).
    """
    dataset = config.data.dataset
    bids = recording.bids
    if bids is not None:
        in_training_dataset = Path(bids["root"]).resolve() == config.data.paths.raw.resolve() or (
            dataset.id in bids["doi"]
        )
        if in_training_dataset:
            hit = registry[
                (registry["subject"] == bids["subject"]) & (registry["session"] == bids["session"])
            ]
            if len(hit):
                return dict(hit.iloc[0]) | {"matched_by": "BIDS subject and session"}
    for column, how in (
        ("file_sha256", "SHA-256 of the file"),
        ("signal_sha256", "hash of the signal"),
    ):
        hit = registry[registry[column] == getattr(recording, column)]
        if len(hit):
            return dict(hit.iloc[0]) | {"matched_by": how}
    return None
