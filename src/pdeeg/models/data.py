"""The data of one classification task: a row per recording, its label, and whose it is.

Features come from the wide tables (one column per feature and site) at the level of
``model.level``: the five regions, or the channels. Age and sex are the confound features.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pdeeg.config import Config
from pdeeg.stats.confounds import covariate_matrix

CONFOUNDS = "confounds"
CONFOUND_COLUMNS = ("age", "sex")
_INDEX = ["subject", "session"]


class TaskError(Exception):
    """The recordings or tables do not allow the task to be built."""


@dataclass(frozen=True)
class TaskData:
    """Everything a classifier sees of one task and feature set.

    ``X`` is recordings x features and ``y`` is 1 for the positive conditions. ``groups`` holds
    the participant of every row: splits never separate rows with the same group. ``rows`` has
    ``subject``, ``session`` and ``condition`` for every row, ``features`` the column names, and
    ``left_out`` the participants dropped for a missing value. ``weights`` is None when every
    recording counts once, else each row's weight (a participant's rows sum to 1). ``within``
    says that both classes come from the same participants, so labels are permuted within them.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    rows: pd.DataFrame
    features: list[str]
    paired: bool
    left_out: list[str]
    weights: np.ndarray | None = None
    within: bool = False


def site_of(column: str) -> str:
    """The channel or region a wide-table column belongs to: the part after the last ``_``.

    Empty for a column without one, such as ``age``.
    """
    head, _, site = column.rpartition("_")
    return site if head else ""


def feature_type(column: str) -> str:
    """A wide-table column without its site, e.g. ``envelope_beta_main_h2``; ``age`` stays."""
    head, _, _ = column.rpartition("_")
    return head or column


def wide_features(wide: pd.DataFrame, config: Config) -> pd.DataFrame:
    """The columns of a wide table at ``model.level``, one row per recording (subject, session).

    Only the rows of ``model.segments`` are used.
    """
    model = config.model
    rows = wide[wide["segments"] == model.segments]
    regions = set(config.data.regions)
    keys = {"subject", "session", "group", "segments"}
    columns = [
        column
        for column in rows.columns
        if column not in keys and (site_of(column) in regions) == (model.level == "region")
    ]
    if not columns:
        raise TaskError(f"no {model.level} columns in the {model.segments} rows of a wide table")
    return rows.set_index(_INDEX)[columns]


def within_participants(config: Config, task: str) -> bool:
    """Whether both classes of ``task`` come from one group of data.yaml, i.e. the same people."""
    groups = {session.condition: session.group for session in config.data.sessions.values()}
    settings = config.model.tasks[task]
    return len({groups[c] for c in (*settings.positive, *settings.negative)}) == 1


def applicable(config: Config, task: str, feature_set: str) -> bool:
    """Whether a feature set says anything in a task: age and sex alone do not, within people."""
    tables = config.model.feature_sets[feature_set]
    return not (within_participants(config, task) and set(tables) == {CONFOUNDS})


def participant_weights(groups: np.ndarray) -> np.ndarray:
    """One over the number of rows of each row's participant: everyone weighs 1 in total."""
    groups = pd.Series(np.asarray(groups))
    return 1.0 / groups.map(groups.value_counts()).to_numpy(dtype=float)


def task_rows(config: Config, recordings: pd.DataFrame, task: str) -> pd.DataFrame:
    """The recordings of ``task`` with a ``label`` column, in the order of ``recordings``.

    A paired task has, for every participant, one recording of each side. An unpaired task has
    one recording per participant, unless its classes come from the same participants or its
    weights are per participant.
    """
    settings = config.model.tasks[task]
    conditions = (*settings.positive, *settings.negative)
    dropped = set(settings.exclude_recordings)
    recording = pd.Series(list(zip(recordings["subject"], recordings["session"], strict=True)))
    rows = recordings[
        recordings["condition"].isin(conditions)
        & ~recordings["subject"].isin(settings.exclude_subjects)
        & ~recording.isin(dropped).to_numpy()
    ].copy()
    rows["label"] = rows["condition"].isin(settings.positive).astype(int)
    per_person = rows.groupby("subject", sort=False)["label"].agg(["size", "sum"])
    if settings.paired:
        complete = per_person[(per_person["size"] == 2) & (per_person["sum"] == 1)].index
        rows = rows[rows["subject"].isin(complete)]
    elif (
        settings.weights == "per_recording"
        and not within_participants(config, task)
        and (per_person["size"] > 1).any()
    ):
        repeated = per_person.index[per_person["size"] > 1].tolist()
        raise TaskError(f"{task}: more than one recording for {', '.join(repeated)}")
    return rows.reset_index(drop=True)


def centre_within(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``X`` less the mean of the rows of the same group: each row against its own participant.

    Nothing but the rows of one participant goes into that participant's values.
    """
    X = np.asarray(X, dtype=float)
    centred = np.empty_like(X)
    for group in np.unique(groups):
        rows = groups == group
        centred[rows] = X[rows] - X[rows].mean(axis=0)
    return centred


def task_data(
    config: Config,
    recordings: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame],
    task: str,
    feature_set: str,
) -> TaskData:
    """The rows, labels, groups and features of ``task`` with ``feature_set``.

    ``tables`` maps ``psd`` and ``mfdfa`` to their wide tables. A participant with a missing
    value in any feature is left out with all their recordings. In a paired task the features
    are centred within participants.
    """
    model = config.model
    settings = model.tasks[task]
    rows = task_rows(config, recordings, task)
    index = pd.MultiIndex.from_frame(rows[_INDEX])
    parts = []
    for table in model.feature_sets[feature_set]:
        if table == CONFOUNDS:
            values = covariate_matrix(rows, CONFOUND_COLUMNS)
            parts.append(pd.DataFrame(values, index=index, columns=list(CONFOUND_COLUMNS)))
            continue
        if table not in tables:
            raise TaskError(f"no {table} table")
        features = wide_features(tables[table], config)
        absent = index.difference(features.index)
        if len(absent):
            raise TaskError(f"{table}: no features for {', '.join(map(str, absent))}")
        parts.append(features.reindex(index))
    frame = pd.concat(parts, axis=1)
    complete = frame.notna().all(axis=1).to_numpy()
    groups = rows[model.cv.group_key].to_numpy()
    missing = sorted(set(groups[~complete]))
    keep = ~np.isin(groups, missing)
    X = frame.to_numpy(dtype=float)[keep]
    groups = groups[keep]
    if settings.paired:
        X = centre_within(X, groups)
    return TaskData(
        X=X,
        y=rows["label"].to_numpy()[keep],
        groups=groups,
        rows=rows.loc[keep, ["subject", "session", "condition"]].reset_index(drop=True),
        features=list(frame.columns),
        paired=settings.paired,
        left_out=missing,
        weights=participant_weights(groups) if settings.weights == "per_participant" else None,
        within=within_participants(config, task),
    )
