import dataclasses
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from pdeeg.config import load_config
from pdeeg.features.mfdfa import make_qs
from pdeeg.stats.analysis import (
    ANALYSES,
    OFF_VS_ON,
    OFF_VS_ON_ALL,
    PD_VS_HC,
    PD_VS_HC_ADJUSTED,
    add_fdr,
    random_stream,
    result_paths,
    run_analysis,
    write_results,
)
from pdeeg.stats.features import (
    EXPLORATORY,
    PRIMARY,
    StatsError,
    between_recordings,
    catalogue,
    paired_subjects,
)
from pdeeg.stats.report import write_report

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CHANNELS = mne.channels.make_standard_montage("biosemi32").ch_names
PLANTED = ["Fz", "FC1", "FC2", "Cz"]
# Effects planted in the synthetic tables: patients higher than controls in both sessions, and
# lower on than off medication. The third primary feature has no effect anywhere.
DISEASE = "envelope_beta_main_h2"
MEDICATION = "broadband_long_delta_alpha"
NULL = "envelope_beta_main_delta_alpha"
# The control whose value of one channel is missing in one exploratory feature.
MISSING = ("hc7", "log_relative_power_alpha", "Oz")


def stats_config(root: Path):
    """The repository configuration with the band powers as the only exploratory features."""
    config = load_config(CONFIG_DIR)
    stats = config.stats
    return dataclasses.replace(
        config,
        stats=dataclasses.replace(
            stats,
            features=dataclasses.replace(stats.features, exploratory_tables=("psd",)),
            cluster=dataclasses.replace(stats.cluster, n_permutations=1000),
            bootstrap=dataclasses.replace(stats.bootstrap, n_resamples=200),
            report=root / "reports" / "stats_results.md",
            figures_dir=root / "reports" / "figures" / "stats",
            n_jobs=2,
        ),
        data=dataclasses.replace(
            config.data, paths=dataclasses.replace(config.data.paths, processed=root / "processed")
        ),
    )


def synthetic_recordings() -> pd.DataFrame:
    """16 controls and 15 patients with two sessions, among them pd6 and pd16."""
    rng = np.random.default_rng(1)
    rows = [
        {"subject": f"hc{i}", "session": "hc", "group": "HC", "condition": "HC"}
        for i in range(1, 17)
    ]
    patients = [3, 5, 6, 9, 11, 12, 13, 14, 16, 17, 19, 22, 23, 26, 28]
    for number in patients:
        for session, condition in (("off", "PD_OFF"), ("on", "PD_ON")):
            rows.append(
                {
                    "subject": f"pd{number}",
                    "session": session,
                    "group": "PD",
                    "condition": condition,
                }
            )
    table = pd.DataFrame(rows)
    people = table["subject"].unique()
    ages = dict(zip(people, rng.integers(50, 80, len(people)), strict=True))
    sexes = dict(zip(people, rng.choice(["m", "f"], len(people)), strict=True))
    return table.assign(
        participant_id="sub-" + table["subject"],
        age=table["subject"].map(ages),
        sex=table["subject"].map(sexes),
    )


def synthetic_tables(config, recordings: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long feature tables in the layout of the real ones, with the planted effects."""
    rng = np.random.default_rng(2)
    planted = np.isin(CHANNELS, PLANTED)
    rows = {"mfdfa": [], "psd": []}
    for feature in catalogue(config):
        base = {s: 0.8 * rng.standard_normal(len(CHANNELS)) for s in recordings["subject"]}
        for recording in recordings.to_dict("records"):
            values = base[recording["subject"]] + 0.5 * rng.standard_normal(len(CHANNELS))
            if feature.key == DISEASE and recording["group"] == "PD":
                values = values + 1.5 * planted
            if feature.key == MEDICATION and recording["session"] == "on":
                values = values - 1.0 * planted
            for segments in ("full", "clean"):
                names = CHANNELS + ["frontal"]
                site_values = np.append(values, values[:9].mean())
                for name, value in zip(names, site_values, strict=True):
                    if (recording["subject"], feature.key, name) == MISSING:
                        value = np.nan
                    rows[feature.table].append(
                        {
                            "subject": recording["subject"],
                            "session": recording["session"],
                            "group": recording["group"],
                            "level": "region" if name == "frontal" else "channel",
                            "name": name,
                            "segments": segments,
                            **dict(feature.selection),
                            "value": value,
                        }
                    )
    return {table: pd.DataFrame(table_rows) for table, table_rows in rows.items()}


def synthetic_curves(recordings: pd.DataFrame) -> pd.DataFrame:
    """h(q) falling with q, and the spectrum that goes with it, for every recording."""
    rng = np.random.default_rng(3)
    qs = make_qs(-5.0, 5.0, 0.5)
    parts = []
    for recording in recordings.to_dict("records"):
        for series, fit_range in (("beta", "main"), ("broadband", "long")):
            slope = 0.03 + 0.01 * rng.standard_normal()
            h = 0.8 - slope * qs
            tau = qs * h - 1
            alpha = np.gradient(tau, qs)
            parts.append(
                pd.DataFrame(
                    {
                        "subject": recording["subject"],
                        "session": recording["session"],
                        "group": recording["group"],
                        "channel": "Fz",
                        "series": series,
                        "fit_range": fit_range,
                        "segments": "clean",
                        "q": qs,
                        "h": h,
                        "alpha": alpha,
                        "f_alpha": qs * alpha - tau,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def analysed(tmp_path_factory):
    root = tmp_path_factory.mktemp("stats")
    config = stats_config(root)
    recordings = synthetic_recordings()
    results = run_analysis(config, recordings, synthetic_tables(config, recordings))
    return config, recordings, results


def row_of(results, analysis, feature):
    tests = results.tests
    return tests[(tests["analysis"] == analysis) & (tests["feature"] == feature)].iloc[0]


def best_cluster(results, analysis, feature):
    clusters = results.clusters
    chosen = clusters[(clusters["analysis"] == analysis) & (clusters["feature"] == feature)]
    return chosen.sort_values("p").iloc[0]


# --- what is tested, on whom ---------------------------------------------------------------------


def test_catalogue_puts_the_primary_features_first_and_labels_the_rest_exploratory():
    config = load_config(CONFIG_DIR)
    features = catalogue(config)
    keys = [feature.key for feature in features]

    assert keys[:3] == list(config.stats.features.primary)
    assert [f.tier for f in features[:3]] == [PRIMARY] * 3
    assert all(f.tier == EXPLORATORY for f in features[3:])
    assert len(set(keys)) == len(keys)
    # The other MFDFA features of the wide table and the five band powers.
    assert "broadband_short_h2" in keys and "envelope_theta_main_asymmetry" in keys
    assert "broadband_short_delta_alpha" not in keys  # not in the wide table
    assert [k for k in keys if k.startswith("log_relative_power_")] == [
        f"log_relative_power_{band}" for band in config.psd.bands
    ]
    assert len(keys) == 1 + 5 + 3 * 5 + 5


def test_an_unknown_primary_feature_is_an_error():
    config = load_config(CONFIG_DIR)
    stats = config.stats
    wrong = dataclasses.replace(
        config,
        stats=dataclasses.replace(
            stats, features=dataclasses.replace(stats.features, primary=("envelope_beta_h2",))
        ),
    )

    with pytest.raises(StatsError, match="envelope_beta_h2"):
        catalogue(wrong)


def test_one_recording_per_participant_and_the_session_is_configurable():
    config = load_config(CONFIG_DIR)
    recordings = synthetic_recordings()

    chosen = between_recordings(recordings, config)
    assert chosen["subject"].is_unique
    assert set(chosen.loc[chosen["side"] == "a", "session"]) == {"off"}
    assert (chosen["side"] == "a").sum() == 15 and (chosen["side"] == "b").sum() == 16

    on = dataclasses.replace(
        config,
        stats=dataclasses.replace(
            config.stats, between=dataclasses.replace(config.stats.between, pd_session="on")
        ),
    )
    assert set(between_recordings(recordings, on).query("side == 'a'")["session"]) == {"on"}


def test_pairs_leave_out_the_listed_patients_only_when_asked():
    config = load_config(CONFIG_DIR)
    recordings = synthetic_recordings()

    main = paired_subjects(recordings, config)
    every = paired_subjects(recordings, config, exclude=False)

    assert len(every) == 15 and len(main) == 13
    assert set(every) - set(main) == {"pd6", "pd16"}
    assert not any(subject.startswith("hc") for subject in every)


def test_each_test_draws_its_own_random_numbers():
    first = random_stream(7, PD_VS_HC, "a").random(3)

    np.testing.assert_array_equal(first, random_stream(7, PD_VS_HC, "a").random(3))
    assert not np.array_equal(first, random_stream(7, PD_VS_HC, "b").random(3))
    assert not np.array_equal(first, random_stream(8, PD_VS_HC, "a").random(3))


# --- the results ---------------------------------------------------------------------------------


def test_every_feature_goes_through_every_analysis(analysed):
    config, _, results = analysed
    keys = [feature.key for feature in catalogue(config)]

    assert len(results.tests) == len(keys) * len(ANALYSES)
    assert set(results.tests["analysis"]) == set(ANALYSES)
    assert len(results.channels) == len(results.tests) * len(CHANNELS)
    # Region rows of the tables are not channels.
    assert set(results.channels["channel"]) == set(CHANNELS)


def test_the_planted_disease_effect_is_found_where_it_was_put(analysed):
    _, _, results = analysed

    for analysis in (PD_VS_HC, PD_VS_HC_ADJUSTED):
        best = best_cluster(results, analysis, DISEASE)
        assert best["significant"] and best["sign"] == 1
        assert set(PLANTED) <= set(best["channels"].split(", "))
    # Both sessions of the patients carry it, so it is no medication effect.
    assert not row_of(results, OFF_VS_ON, DISEASE)["significant"]


def test_the_planted_medication_effect_is_found_within_patients(analysed):
    _, _, results = analysed

    for analysis in (OFF_VS_ON, OFF_VS_ON_ALL):
        best = best_cluster(results, analysis, MEDICATION)
        assert best["significant"] and best["sign"] == 1  # off above on
        assert set(PLANTED) <= set(best["channels"].split(", "))
    assert not row_of(results, PD_VS_HC, MEDICATION)["significant"]
    assert not row_of(results, PD_VS_HC, NULL)["significant"]
    assert not row_of(results, OFF_VS_ON, NULL)["significant"]


def test_sample_sizes_and_degrees_of_freedom(analysed):
    _, _, results = analysed

    main = row_of(results, PD_VS_HC, DISEASE)
    assert (main["n_a"], main["n_b"], main["df"]) == (15, 16, 29)
    # Age and sex use up two more degrees of freedom.
    assert row_of(results, PD_VS_HC_ADJUSTED, DISEASE)["df"] == 27
    assert row_of(results, OFF_VS_ON, MEDICATION)["n_a"] == 13
    # 1000 sign flips asked for, of the 4096 there are: not exact.
    assert not row_of(results, OFF_VS_ON, MEDICATION)["exact"]
    assert row_of(results, OFF_VS_ON, MEDICATION)["n_permutations"] == 1000
    assert row_of(results, OFF_VS_ON_ALL, MEDICATION)["n_a"] == 15


def test_a_missing_value_leaves_out_that_participant_from_that_feature_only(analysed):
    _, _, results = analysed
    subject, feature, _ = MISSING

    row = row_of(results, PD_VS_HC, feature)
    assert row["left_out"] == subject and row["n_b"] == 15
    assert row_of(results, PD_VS_HC, DISEASE)["left_out"] == ""


def test_fdr_runs_within_each_family_over_both_contrasts(analysed):
    _, _, results = analysed
    tests = results.tests

    for tier in (PRIMARY, EXPLORATORY):
        members = tests[(tests["tier"] == tier) & tests["analysis"].isin([PD_VS_HC, OFF_VS_ON])]
        expected = int(np.maximum(members["n_clusters"], 1).sum())
        assert (members["m"] == expected).all()
    assert tests["q"].between(0, 1).all()
    assert (results.clusters["q"] >= results.clusters["p"] - 1e-12).all()


def test_fdr_by_hand():
    tests = pd.DataFrame(
        {
            "analysis": [PD_VS_HC, OFF_VS_ON, PD_VS_HC_ADJUSTED, OFF_VS_ON_ALL],
            "feature": ["x"] * 4,
            "tier": [PRIMARY] * 4,
            "n_clusters": [2, 0, 1, 1],
        }
    )
    clusters = pd.DataFrame(
        {
            "analysis": [PD_VS_HC, PD_VS_HC, PD_VS_HC_ADJUSTED, OFF_VS_ON_ALL],
            "feature": ["x"] * 4,
            "cluster": [1, 2, 1, 1],
            "p": [0.01, 0.30, 0.02, 0.04],
        }
    )

    tests, clusters = add_fdr(tests, clusters, 0.05)

    # Main set: 0.01 and 0.30 from PD vs HC, and 1 for OFF vs ON, which found no cluster.
    assert clusters["q"].iloc[0] == pytest.approx(0.03)
    assert clusters["q"].iloc[1] == pytest.approx(0.45)
    assert tests["q"].iloc[1] == pytest.approx(1.0)
    # The adjusted test takes the place of PD vs HC: 0.02 and the stand-in 1.
    assert clusters["q"].iloc[2] == pytest.approx(0.04)
    # All pairs takes the place of OFF vs ON: 0.01, 0.30 and 0.04.
    assert clusters["q"].iloc[3] == pytest.approx(0.06)
    assert list(tests["significant"]) == [True, False, True, False]


def test_results_and_report_are_written(analysed):
    config, _, results = analysed

    paths = write_results(config, results)
    assert paths == result_paths(config)
    back = pd.read_parquet(paths["clusters"])
    pd.testing.assert_frame_equal(back, results.clusters.reset_index(drop=True), check_dtype=False)

    report = write_report(config, results, synthetic_curves(synthetic_recordings()))
    text = report.read_text(encoding="utf-8")
    for heading in (
        "Summary in plain language",
        "Age and sex",
        "Primary features",
        "Exploratory features",
    ):
        assert f"## {heading}" in text
    assert "h2, beta envelope: PD OFF > HC" in text
    assert "SHA-256" in text and "pd6, pd16" in text
    assert "not a diagnostic" in text or "diagnostic test" in text
    for name in (
        "primary_pd_vs_hc",
        "primary_off_vs_on",
        "exploratory_pd_vs_hc",
        "exploratory_off_vs_on",
        "paired_off_on",
        "mfdfa_curves",
    ):
        assert (config.stats.figures_dir / f"{name}.png").is_file()


def test_report_without_curves_says_how_to_make_them(analysed):
    config, _, results = analysed

    text = write_report(config, results, None).read_text(encoding="utf-8")

    assert "pdeeg mfdfa-curves" in text
