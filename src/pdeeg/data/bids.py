"""Index and load the EEG recordings of a BIDS dataset."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne
import pandas as pd
from mne_bids import find_matching_paths, get_bids_path_from_fname, read_raw_bids
from mne_bids.config import ALLOWED_DATATYPE_EXTENSIONS

from pdeeg.config import SessionConfig

logger = logging.getLogger(__name__)

# Names fixed by the BIDS specification, not by this project.
_DATATYPE = "eeg"
_PARTICIPANTS_FILE = "participants.tsv"
_PARTICIPANT_KEY = "participant_id"
_MISSING = "n/a"


class DatasetError(Exception):
    """The dataset on disk does not have the layout or metadata this package expects."""


def _read_participants(root: Path) -> pd.DataFrame:
    path = root / _PARTICIPANTS_FILE
    if not path.is_file():
        raise DatasetError(f"{path} not found; is {root} a BIDS dataset?")
    # BIDS marks missing values with "n/a" only. pandas' default list would also blank out
    # legitimate text such as "NA" or "None".
    participants = pd.read_csv(path, sep="\t", na_values=[_MISSING], keep_default_na=False)
    if _PARTICIPANT_KEY not in participants.columns:
        raise DatasetError(f"{path} has no {_PARTICIPANT_KEY} column")
    repeated = participants.loc[participants[_PARTICIPANT_KEY].duplicated(), _PARTICIPANT_KEY]
    if not repeated.empty:
        raise DatasetError(f"{path} lists more than once: {', '.join(sorted(set(repeated)))}")
    return participants


def list_recordings(
    root: str | Path,
    sessions: Mapping[str, SessionConfig],
    *,
    rename_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return one row per EEG recording under BIDS dataset ``root``.

    Columns are ``subject``, ``session``, ``group``, ``condition`` and ``bids_path`` (relative to
    ``root``, so the table holds no machine-specific paths), followed by every column of
    ``participants.tsv``, renamed according to ``rename_columns``.

    ``group`` and ``condition`` are looked up from each recording's BIDS session label in
    ``sessions``; subject identifiers are never parsed. The dataset is rejected if a session
    label is not in ``sessions``, if one participant's sessions disagree about the group, if a
    participant has two recordings for one session, or if a recording has no row in
    ``participants.tsv``.

    Rows follow the order of ``participants.tsv``, then session label.
    """
    root = Path(root)
    participants = _read_participants(root).rename(columns=dict(rename_columns or {}))

    bids_paths = find_matching_paths(
        root,
        datatypes=_DATATYPE,
        suffixes=_DATATYPE,
        extensions=ALLOWED_DATATYPE_EXTENSIONS[_DATATYPE],
    )
    if not bids_paths:
        raise DatasetError(f"no EEG recordings found under {root}")

    rows = []
    for bids_path in bids_paths:
        if bids_path.session not in sessions:
            raise DatasetError(
                f"{bids_path.basename}: session {bids_path.session!r} is not a configured "
                f"session ({', '.join(sessions)})"
            )
        session = sessions[bids_path.session]
        rows.append(
            {
                "subject": bids_path.subject,
                "session": bids_path.session,
                "group": session.group,
                "condition": session.condition,
                "bids_path": bids_path.fpath.relative_to(bids_path.root).as_posix(),
                _PARTICIPANT_KEY: f"sub-{bids_path.subject}",
            }
        )
    recordings = pd.DataFrame(rows)

    groups = recordings.groupby("subject")["group"].nunique()
    mixed = sorted(groups.index[groups > 1])
    if mixed:
        raise DatasetError(f"sessions assign more than one group to: {', '.join(mixed)}")
    repeated = recordings.loc[recordings.duplicated(["subject", "session"]), "bids_path"]
    if not repeated.empty:
        raise DatasetError(f"more than one recording for a session: {', '.join(repeated)}")
    clashes = sorted((set(recordings.columns) & set(participants.columns)) - {_PARTICIPANT_KEY})
    if clashes:
        raise DatasetError(
            f"{_PARTICIPANTS_FILE} column(s) {', '.join(clashes)} clash with the recordings "
            "table; rename them with rename_columns"
        )
    unlisted = sorted(set(recordings[_PARTICIPANT_KEY]) - set(participants[_PARTICIPANT_KEY]))
    if unlisted:
        raise DatasetError(f"recordings with no row in {_PARTICIPANTS_FILE}: {', '.join(unlisted)}")
    unrecorded = sorted(set(participants[_PARTICIPANT_KEY]) - set(recordings[_PARTICIPANT_KEY]))
    if unrecorded:
        logger.warning("%s rows with no recording: %s", _PARTICIPANTS_FILE, ", ".join(unrecorded))

    participants = participants.assign(_order=range(len(participants)))
    recordings = recordings.merge(participants, on=_PARTICIPANT_KEY, validate="many_to_one")
    recordings = recordings.sort_values(["_order", "session"]).drop(columns="_order")
    return recordings.reset_index(drop=True)


def load_raw(row: Any, root: str | Path, montage: str) -> mne.io.BaseRaw:
    """Read one recording as scalp EEG with ``montage`` applied; the data is not preloaded.

    ``row`` is a row of :func:`list_recordings` (anything with a ``bids_path`` attribute) and
    ``root`` is the BIDS dataset it was listed from.

    Channels are selected in two steps, neither of which names a channel. First by type: only
    channels typed EEG are kept, which removes triggers, EOG and the like. Then by position:
    EEG-typed channels that ``montage`` has no position for are dropped, because a sidecar may
    type auxiliary electrodes as EEG (ds002778 does this for EXG1-8). A recording that lacks a
    channel of the montage is rejected.
    """
    bids_path = get_bids_path_from_fname(Path(root) / row.bids_path).update(root=root)
    with warnings.catch_warnings():
        # mne-bids warns that it cannot store the non-standard participants.tsv columns on the
        # Raw object. They are carried by the recordings table instead.
        warnings.filterwarnings(
            "ignore", message="Unable to map the following column", category=RuntimeWarning
        )
        raw = read_raw_bids(bids_path)
    raw.pick("eeg")

    layout = mne.channels.make_standard_montage(montage)
    absent = [name for name in layout.ch_names if name not in raw.ch_names]
    if absent:
        raise DatasetError(
            f"{row.bids_path}: no EEG channel named {', '.join(absent)} ({montage} montage)"
        )
    unplaced = [name for name in raw.ch_names if name not in layout.ch_names]
    if unplaced:
        logger.info(
            "%s: dropping EEG-typed channels with no %s position: %s",
            row.bids_path,
            montage,
            ", ".join(unplaced),
        )
        raw.drop_channels(unplaced)
    raw.set_montage(layout)
    return raw
