"""ICA decomposition and ICLabel classification of the components."""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel.config import ICLABEL_NUMERICAL_TO_STRING
from mne_icalabel.iclabel import iclabel_label_components

from pdeeg.config import IcaConfig

# ICLabel's seven classes, in the column order of its probability matrix.
CLASSES = tuple(ICLABEL_NUMERICAL_TO_STRING[i] for i in range(len(ICLABEL_NUMERICAL_TO_STRING)))


def fit_ica(raw: mne.io.BaseRaw, config: IcaConfig) -> tuple[ICA, int]:
    """Fit ICA on ``raw``; return it with the data rank that was measured.

    With ``n_components: rank`` the number of components is the numerical rank of the data,
    which is one less than the channel count after an average reference. Asking for as many
    components as channels would make ICA split off a component that carries no data.
    """
    rank = int(mne.compute_rank(raw, rank=None)["eeg"])
    n_components = rank if config.n_components == "rank" else config.n_components
    fit_params: dict[str, Any] = {"extended": config.extended}
    if config.method == "picard":
        fit_params["ortho"] = False  # with extended=True this is extended infomax
    ica = ICA(
        n_components=n_components,
        method=config.method,
        fit_params=fit_params,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    ica.fit(raw, picks="eeg")
    return ica, rank


def classify_components(raw: mne.io.BaseRaw, ica: ICA) -> np.ndarray:
    """ICLabel class probabilities, shape (components, 7), columns ordered as ``CLASSES``."""
    return np.asarray(iclabel_label_components(raw, ica, inplace=True), dtype=float)


def select_components(probabilities: np.ndarray, config: IcaConfig) -> list[dict[str, Any]]:
    """Components to remove: most probable class is excluded and at least ``threshold`` sure."""
    removed = []
    for index, row in enumerate(probabilities):
        best = int(np.argmax(row))
        if CLASSES[best] in config.exclude_labels and row[best] >= config.threshold:
            removed.append(
                {
                    "index": index,
                    "label": CLASSES[best],
                    "probability": float(row[best]),
                    "probabilities": {name: float(p) for name, p in zip(CLASSES, row, strict=True)},
                }
            )
    return removed
