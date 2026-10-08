import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, write_raw_bids

from pdeeg.config import SessionConfig, load_config
from pdeeg.data.bids import DatasetError, list_recordings, load_raw

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

MONTAGE = "biosemi32"
SFREQ = 256.0
SESSIONS = {
    "hc": SessionConfig(group="HC", condition="HC"),
    "off": SessionConfig(group="PD", condition="PD_OFF"),
    "on": SessionConfig(group="PD", condition="PD_ON"),
}
# Subject labels that say nothing true about the group: "pd9" is the control. Any code that
# reads the group out of the label gets this dataset wrong.
RECORDINGS = [("pd9", "hc"), ("x2", "off"), ("x2", "on")]
# Laid out like ds002778: "gender" rather than "sex", no group column, "n/a" for missing.
# The order differs from the alphabetical order of the folders on disk.
PARTICIPANTS = pd.DataFrame(
    {
        "participant_id": ["sub-x2", "sub-pd9"],
        "age": [70, 61],
        "gender": ["m", "f"],
        "disease_duration": [4, "n/a"],
        "notes": ["n/a", "None"],  # "None" is text and must not be read as missing
    }
)


def _synthetic_raw(rng: np.random.Generator) -> mne.io.RawArray:
    """Two seconds laid out like a ds002778 recording, plus one EOG channel."""
    scalp = mne.channels.make_standard_montage(MONTAGE).ch_names
    # As in ds002778, the external electrodes are typed EEG although they are not scalp channels.
    names = [*scalp, "EXG1", "EXG2", "VEOG", "Status"]
    types = ["eeg"] * (len(scalp) + 2) + ["eog", "stim"]
    data = rng.normal(scale=1e-5, size=(len(names), int(2 * SFREQ)))
    data[-1] = 0.0
    data[-1, 10:20] = 1.0
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, types), verbose="error")
    raw.info["line_freq"] = 60.0
    return raw


@pytest.fixture(scope="module")
def bids_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bids")
    rng = np.random.default_rng(0)
    for subject, session in RECORDINGS:
        bids_path = BIDSPath(
            subject=subject, session=session, task="rest", datatype="eeg", root=root
        )
        write_raw_bids(
            _synthetic_raw(rng), bids_path, format="BDF", allow_preload=True, verbose="error"
        )
    PARTICIPANTS.to_csv(root / "participants.tsv", sep="\t", index=False)
    return root


def test_list_recordings_takes_group_from_session_not_subject_label(bids_root: Path):
    recordings = list_recordings(bids_root, SESSIONS, rename_columns={"gender": "sex"})

    assert list(recordings.columns[:5]) == ["subject", "session", "group", "condition", "bids_path"]
    # participants.tsv order, then session.
    assert list(zip(recordings.subject, recordings.session, recordings.group, strict=True)) == [
        ("x2", "off", "PD"),
        ("x2", "on", "PD"),
        ("pd9", "hc", "HC"),
    ]
    assert list(recordings.condition) == ["PD_OFF", "PD_ON", "HC"]
    assert not any(Path(path).is_absolute() for path in recordings.bids_path)
    assert all((bids_root / path).is_file() for path in recordings.bids_path)

    assert "gender" not in recordings.columns
    assert list(recordings.sex) == ["m", "m", "f"]
    assert list(recordings.age) == [70, 70, 61]
    control = recordings.iloc[2]
    assert pd.isna(control.disease_duration)
    assert control.notes == "None"
    assert recordings.notes.isna().tolist() == [True, True, False]


def test_list_recordings_rejects_sessions_that_do_not_fit_the_layout(bids_root: Path):
    without_on = {label: SESSIONS[label] for label in ("hc", "off")}
    with pytest.raises(DatasetError, match="not a configured session"):
        list_recordings(bids_root, without_on)

    on_as_control = {**SESSIONS, "on": SessionConfig(group="HC", condition="HC")}
    with pytest.raises(DatasetError, match="more than one group to: x2"):
        list_recordings(bids_root, on_as_control)


def test_list_recordings_rejects_a_recording_without_a_participants_row(
    bids_root: Path, tmp_path: Path
):
    root = tmp_path / "bids"
    shutil.copytree(bids_root, root)
    PARTICIPANTS.iloc[:1].to_csv(root / "participants.tsv", sep="\t", index=False)

    with pytest.raises(DatasetError, match="no row in participants.tsv: sub-pd9"):
        list_recordings(root, SESSIONS)


def test_recordings_survive_a_parquet_round_trip(bids_root: Path, tmp_path: Path):
    recordings = list_recordings(bids_root, SESSIONS, rename_columns={"gender": "sex"})
    path = tmp_path / "recordings.parquet"

    recordings.to_parquet(path, index=False)

    pd.testing.assert_frame_equal(pd.read_parquet(path), recordings)


def test_load_raw_returns_only_scalp_eeg_with_the_montage_applied(bids_root: Path):
    recordings = list_recordings(bids_root, SESSIONS)
    scalp = mne.channels.make_standard_montage(MONTAGE).ch_names

    raw = load_raw(recordings.iloc[0], bids_root, MONTAGE)

    # EXG1-2 (typed EEG, no scalp position), VEOG and Status are gone.
    assert raw.ch_names == scalp
    assert set(raw.get_channel_types()) == {"eeg"}
    assert raw.info["sfreq"] == SFREQ
    assert all(np.isfinite(channel["loc"][:3]).all() for channel in raw.info["chs"])
    # Rows from itertuples() work as well as Series rows.
    assert load_raw(next(recordings.itertuples()), bids_root, MONTAGE).ch_names == scalp


def test_load_raw_rejects_a_recording_that_lacks_montage_channels(bids_root: Path):
    recordings = list_recordings(bids_root, SESSIONS)

    with pytest.raises(DatasetError, match="no EEG channel named"):
        load_raw(recordings.iloc[0], bids_root, "biosemi64")


def test_real_dataset_counts_channels_and_sampling_rate():
    """Integration test against the downloaded dataset; skipped until it has been downloaded."""
    data = load_config(CONFIG_DIR).data
    if not (data.paths.raw / "participants.tsv").is_file():
        pytest.skip(f"raw dataset not found at {data.paths.raw}; run scripts/download_data.py")

    recordings = list_recordings(
        data.paths.raw, data.sessions, rename_columns=data.participants_rename
    )

    assert len(recordings) == 46
    assert recordings.groupby("group").subject.nunique().to_dict() == {"HC": 16, "PD": 15}
    sessions = recordings.groupby(["group", "subject"]).session.agg(frozenset)
    assert set(sessions["HC"]) == {frozenset({"hc"})}
    assert set(sessions["PD"]) == {frozenset({"off", "on"})}
    # Independent of how group is derived: only patients have a disease duration.
    assert (recordings.disease_duration.notna() == (recordings.group == "PD")).all()
    assert {"age", "sex"} <= set(recordings.columns)

    scalp = mne.channels.make_standard_montage(data.dataset.montage).ch_names
    for row in recordings.itertuples():
        raw = load_raw(row, data.paths.raw, data.dataset.montage)
        assert raw.ch_names == scalp, row.bids_path
        assert raw.info["sfreq"] == 512.0, row.bids_path
        assert raw.n_times / raw.info["sfreq"] >= 180.0, row.bids_path
