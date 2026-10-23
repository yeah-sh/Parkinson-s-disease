import numpy as np
import pytest

from pdeeg.features.curves import CURVE_COLUMNS, fit_curves, recording_curves
from pdeeg.features.extract import CLEAN, FULL, fit_features, range_scales
from pdeeg.features.mfdfa import make_qs
from pdeeg.features.scaling import BROADBAND, series_names
from test_extract import N_TIMES, SFREQ, noise, small_config


def test_curves_hold_the_h2_the_feature_tables_hold():
    config = small_config().mfdfa
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    x = noise(1, 1)[0]
    scales = range_scales(BROADBAND, x.size, SFREQ, config)["long"]

    curves = fit_curves(x, scales, qs, config)
    features = fit_features(x, scales, qs, config)

    assert curves["h"][qs == 2.0][0] == pytest.approx(features["h2"])
    assert curves["alpha"].max() - curves["alpha"].min() == pytest.approx(features["delta_alpha"])
    # tau(0) = -1, so f = q alpha - tau is 1 at q = 0 whatever was measured.
    assert curves["f_alpha"][qs == 0.0][0] == pytest.approx(1.0)


def test_curves_are_missing_where_the_features_are():
    config = small_config().mfdfa
    qs = make_qs(config.q_min, config.q_max, config.q_step)
    x = noise(2, 1)[0]
    scales = range_scales("alpha", x.size, SFREQ, config)["main"]
    exclude = np.zeros(x.size, dtype=bool)
    exclude[: int(0.9 * x.size)] = True  # too few windows left at most scales

    curves = fit_curves(x, scales, qs, config, exclude)

    assert np.isnan(fit_features(x, scales, qs, config, exclude)["h2"])
    assert all(np.isnan(values).all() for values in curves.values())


def test_recording_curves_cover_every_channel_series_range_and_segments():
    config = small_config().mfdfa
    channels = ["Fz", "Cz"]
    bad = np.zeros(N_TIMES, dtype=bool)
    bad[5000:5500] = True

    table = recording_curves(noise(3, len(channels)), bad, SFREQ, channels, config)

    assert list(table.columns) == list(CURVE_COLUMNS[3:])
    n_qs = len(make_qs(config.q_min, config.q_max, config.q_step))
    n_ranges = len(config.broadband.fit_ranges) + len(config.envelope.bands) * len(
        config.envelope.fit_ranges
    )
    assert len(table) == len(channels) * n_ranges * 2 * n_qs
    assert set(table["series"]) == set(series_names(config))
    assert set(table["segments"]) == {FULL, CLEAN}
    full = table[table["segments"] == FULL].reset_index(drop=True)
    clean = table[table["segments"] == CLEAN].reset_index(drop=True)
    # The bad stretch changes the clean curves and nothing else.
    assert not np.allclose(full["h"], clean["h"])
