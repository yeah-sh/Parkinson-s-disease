import dataclasses
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from pdeeg.config import Config, load_config
from pdeeg.features.mfdfa import make_scales, mfdfa
from pdeeg.features.scaling import (
    BROADBAND,
    EEG,
    analysis_series,
    local_slopes,
    null_series,
    run_inspection,
    scales_in_range,
    select_recordings,
)
from pdeeg.features.scaling_report import write_report
from pdeeg.features.synthetic import power_law_noise
from pdeeg.preprocessing.pipeline import output_paths

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

SFREQ = 256.0
DURATION = 60.0
CHANNELS = ["Fz", "Cz", "Pz", "Oz", "C3", "C4"]
# Nothing a report could contain by accident.
SUBJECTS = ["zebra", "yak", "xerus", "wombat", "vole"]
GROUP = "SECRETGROUP"


# --- scales and slopes -------------------------------------------------------------------------


def test_scales_in_range_are_whole_samples_inside_the_range():
    scales = scales_in_range((0.045, 0.09), 512.0, 12)

    # 0.045 s is 23.04 samples and 0.09 s is 46.08: the ends are rounded inwards.
    assert scales[0] == 24 and scales[-1] == 46
    assert len(scales) == 12
    assert np.all(np.diff(scales) > 0)
    # A range whose ends are whole samples keeps them.
    assert list(scales_in_range((2.0, 17.8), 512.0, 3)) == [1024, 3055, 9113]
    # Only six whole numbers lie between 24 and 29.
    assert list(scales_in_range((0.046, 0.057), 512.0, 12)) == [24, 25, 26, 27, 28, 29]


def test_scales_in_range_rejects_ranges_that_do_not_fit():
    with pytest.raises(ValueError, match="empty"):
        scales_in_range((0.0101, 0.0102), 512.0, 12)
    with pytest.raises(ValueError, match="beyond the largest scale"):
        scales_in_range((2.0, 17.8), 512.0, 12, max_scale=9000)
    assert scales_in_range((2.0, 17.8), 512.0, 12, max_scale=9113)[-1] == 9113


def test_local_slopes_of_power_laws_are_their_exponents():
    scales = make_scales(2**16, 16, 0.1, 20)
    exponents = np.array([0.5, 1.0, 1.7])
    fq = 3.0 * scales ** exponents[:, None]

    slopes = local_slopes(scales, fq, 2)

    assert slopes.shape == fq.shape
    assert np.isnan(slopes[:, :2]).all() and np.isnan(slopes[:, -2:]).all()
    np.testing.assert_allclose(slopes[:, 2:-2], np.broadcast_to(exponents[:, None], (3, 16)))


def test_local_slopes_show_a_crossover_where_it_is():
    scales = np.unique(np.round(np.geomspace(10, 10000, 40)).astype(int))
    knee = 300.0
    fq = np.where(scales < knee, scales**1.5, knee * scales**0.5)

    slopes = local_slopes(scales, fq, 2)

    # A fit through five scales is pure when all five lie on one side of the knee.
    position = np.searchsorted(scales, knee)
    np.testing.assert_allclose(slopes[2 : position - 2], 1.5)
    np.testing.assert_allclose(slopes[position + 2 : -2], 0.5)
    assert np.all(np.diff(slopes[position - 3 : position + 3]) < 0)
    with pytest.raises(ValueError, match="half_width"):
        local_slopes(scales, fq, 0)
    assert np.isnan(local_slopes(scales[:4], fq[:4], 2)).all()


def test_recordings_are_selected_from_the_table_size_alone():
    chosen = select_recordings(46, 8, seed=3)

    assert len(set(chosen)) == 8
    assert chosen.min() >= 0 and chosen.max() < 46
    assert np.array_equal(chosen, select_recordings(46, 8, seed=3))
    assert not np.array_equal(chosen, select_recordings(46, 8, seed=4))
    # The order is random too, so R1 says nothing about where a recording is in the table.
    assert not np.array_equal(chosen, np.sort(chosen))
    assert sorted(select_recordings(5, 5, seed=0)) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="cannot pick"):
        select_recordings(5, 6, seed=0)


# --- the series and the noise they are compared with --------------------------------------------


def test_analysis_series_are_the_data_and_one_trimmed_envelope_per_band():
    config = load_config(CONFIG_DIR).mfdfa
    data = np.random.default_rng(0).standard_normal((2, 8192))

    series = analysis_series(data, 512.0, config)

    assert list(series) == [BROADBAND, "theta", "alpha", "beta"]
    assert series[BROADBAND] is data
    for band in config.envelope.bands:
        assert series[band].shape == (2, 8192 - 2 * int(config.envelope.edge_trim * 512))
        assert np.all(series[band] >= 0)


def test_filtered_noise_shows_what_the_preprocessing_band_pass_does():
    """1/f noise has slope 1 at every scale; band-passed, it keeps it only in between."""
    config = load_config(CONFIG_DIR)
    sfreq, n_times = 512.0, 92160
    trim = int(config.mfdfa.envelope.edge_trim * sfreq)

    noise = null_series(n_times, sfreq, config, np.random.default_rng(0))

    assert list(noise) == [BROADBAND, "theta", "alpha", "beta"]
    assert list(noise[BROADBAND]) == ["1/f noise", "1/f² noise"]
    assert {name: list(noise[name]) for name in ("theta", "alpha", "beta")} == {
        "theta": ["white noise"],
        "alpha": ["white noise"],
        "beta": ["white noise"],
    }
    assert noise[BROADBAND]["1/f noise"].shape == (n_times,)
    assert noise["alpha"]["white noise"].shape == (n_times - 2 * trim,)

    scales = make_scales(n_times, 8, 0.1, 40)
    slopes = local_slopes(scales, mfdfa(noise[BROADBAND]["1/f noise"], scales, [2.0]).Fq, 2)[0]
    seconds = scales / sfreq
    # Means over 20 realisations: 1.38 below 0.03 s (the 50 Hz low-pass smooths the series),
    # 0.997 +- 0.013 between 0.1 and 0.5 s, and 0.09 (0.12 at most) above 5 s, where the 0.5 Hz
    # high-pass has left nothing to accumulate.
    assert np.nanmean(slopes[seconds < 0.03]) > 1.25
    assert np.nanmean(slopes[(seconds > 0.1) & (seconds < 0.5)]) == pytest.approx(1.0, abs=0.06)
    assert np.nanmean(slopes[seconds > 5.0]) < 0.25


# --- the inspection, end to end ------------------------------------------------------------------


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> tuple[Config, pd.DataFrame]:
    """Five cleaned recordings of 1/f noise, and a configuration that fits their 60 s."""
    root = tmp_path_factory.mktemp("scaling")
    full = load_config(CONFIG_DIR)
    prep = full.preprocessing
    mfdfa_config = full.mfdfa
    config = dataclasses.replace(
        full,
        preprocessing=dataclasses.replace(
            prep, output=dataclasses.replace(prep.output, dir=root / "clean")
        ),
        mfdfa=dataclasses.replace(
            mfdfa_config,
            # A tenth of the trimmed envelope is 5.8 s.
            envelope=dataclasses.replace(mfdfa_config.envelope, fit_ranges={"main": (2.0, 5.5)}),
            inspection=dataclasses.replace(
                mfdfa_config.inspection,
                n_recordings=3,
                n_scales=15,
                null_realisations=2,
                n_jobs=1,
                report=root / "reports" / "scaling.md",
                figures_dir=root / "reports" / "figures",
            ),
        ),
    )
    rng = np.random.default_rng(0)
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    n_times = int(DURATION * SFREQ)
    for subject in SUBJECTS:
        data = np.array([power_law_noise(n_times, 1.0, rng) for _ in CHANNELS]) * 5e-6
        path = output_paths(config, subject, "s1")["raw"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "_clean.fif" is not one of MNE's suffixes
            mne.io.RawArray(data, info, verbose="error").save(path, verbose="error")
    recordings = pd.DataFrame({"subject": SUBJECTS, "session": "s1", "group": GROUP})
    return config, recordings


@pytest.fixture(scope="module")
def inspection(project):
    config, recordings = project
    return run_inspection(config, recordings)


def test_inspection_returns_curves_and_fits_for_every_series(project, inspection):
    config, _ = project
    n_scales = config.mfdfa.inspection.n_scales

    assert inspection.sfreq == SFREQ
    assert inspection.n_times == int(DURATION * SFREQ)
    assert list(inspection.channels) == CHANNELS
    assert list(inspection.curves) == [BROADBAND, "theta", "alpha", "beta"]
    for name, curves in inspection.curves.items():
        assert curves.fq.shape == (3, len(CHANNELS), 5, n_scales)
        assert np.isfinite(curves.fq).all()
        for values in curves.null.values():
            assert values.shape == (2, 5, n_scales)
        # The envelope is inspected up to a tenth of its own, trimmed, length.
        longest = 1536 if name == BROADBAND else 1484
        assert curves.scales[-1] == longest

    fits = inspection.fits
    eeg = fits[fits["source"] == EEG]
    # Two broadband ranges and one for each of three bands, for 3 recordings of 6 channels.
    assert len(eeg) == 3 * len(CHANNELS) * 5
    assert set(eeg["unit"]) == {"R1", "R2", "R3"}
    assert set(eeg["channel"]) == set(CHANNELS)
    # Two noises for broadband with two ranges each, one noise and one range for each band.
    assert len(fits) - len(eeg) == 2 * (2 * 2 + 3)
    assert np.isfinite(fits[["h2", "delta_alpha", "min_r2", "h_q_min", "h_q_max"]]).all().all()
    # The recordings are unfiltered 1/f noise, whose exponent is 1 on any range.
    long = eeg[(eeg["series"] == BROADBAND) & (eeg["fit_range"] == "long")]
    assert long["h2"].median() == pytest.approx(1.0, abs=0.1)


def test_inspection_is_reproducible_and_does_not_need_the_labels(project, inspection):
    config, recordings = project

    again = run_inspection(config, recordings[["subject", "session"]], n_jobs=1)

    for name, curves in inspection.curves.items():
        assert np.array_equal(curves.fq, again.curves[name].fq)
        for noise, values in curves.null.items():
            assert np.array_equal(values, again.curves[name].null[noise])
    pd.testing.assert_frame_equal(inspection.fits, again.fits)


def test_report_shows_the_curves_and_no_identity(project, inspection):
    config, recordings = project
    settings = config.mfdfa.inspection

    report = write_report(config, inspection, len(recordings))

    assert report == settings.report
    figures = ["local_slopes.png", *(f"fq_{name}.png" for name in inspection.curves)]
    text = report.read_text(encoding="utf-8")
    for figure in figures:
        assert (settings.figures_dir / figure).is_file()
        assert f"figures/{figure}" in text
    assert "3 of the 5 recordings" in text and "R1 to R3" in text
    for identity in (*SUBJECTS, GROUP):
        assert identity not in text
    for expected in ("long: 0.3 to 1 s", "main: 2 to 5.5 s", "1/f² noise", "white noise"):
        assert expected in text
