"""The pipeline, step by step: download, recordings, preprocess, features, stats, classify, report.

Every step is skipped when it is current. ``download``, ``recordings``, the band powers, the
h(q) curves, ``stats`` and ``classify`` are keyed on their settings and the contents of their
inputs (:mod:`pdeeg.stages`); preprocessing and the MFDFA runs use the per-recording caches of
their own modules, keyed on the settings and the times of the input files. The reports are
always rewritten from what is stored, which takes seconds and gives the same files for the same
inputs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd

from pdeeg.config import Config
from pdeeg.stages import Stage, without

Say = Callable[[str], None]


class StepFailed(Exception):
    """A step could not finish; the message says why."""


# --- helpers -------------------------------------------------------------------------------------


def read_recordings(config: Config) -> pd.DataFrame:
    path = config.data.paths.recordings
    if not path.is_file():
        raise StepFailed(f"{path} not found; run `pdeeg recordings` first")
    return pd.read_parquet(path)


def _cleaned_files(config: Config, recordings: pd.DataFrame) -> list:
    from pdeeg.preprocessing.pipeline import output_paths

    files = []
    for row in recordings.to_dict("records"):
        paths = output_paths(config, row["subject"], row["session"])
        files += [paths["raw"], paths["log"]]
    return files


def _tables(config: Config, form: str) -> dict[str, pd.DataFrame]:
    from pdeeg.features.tables import table_paths

    tables = {}
    for kind in ("mfdfa", "psd"):
        path = table_paths(config, kind)[form]
        if not path.is_file():
            raise StepFailed(f"{path} not found; run `pdeeg features` first")
        tables[kind] = pd.read_parquet(path)
    return tables


def _statuses(results: list[dict[str, Any]], what: str, say: Say) -> None:
    counts = Counter(result["status"] for result in results)
    say(f"{what}: " + ", ".join(f"{count} {status}" for status, count in counts.items()))
    failed = [r for r in results if r["status"] not in ("processed", "cached")]
    if failed:
        details = "\n".join(
            f"sub-{r['subject']} ses-{r['session']}: {r['status']}\n{r['detail']}" for r in failed
        )
        raise StepFailed(f"{what}: {len(failed)} recording(s) not done\n{details}")


def _show(say: Say) -> Callable[[dict[str, Any]], None]:
    def show(result: dict[str, Any]) -> None:
        detail = result["detail"].strip().splitlines()
        say(
            f"sub-{result['subject']} ses-{result['session']}: {result['status']} "
            f"({result['seconds']} s)" + (f" - {detail[0]}" if detail else "")
        )

    return show


# --- steps ---------------------------------------------------------------------------------------


def download_step(config: Config, *, force: bool = False, say: Say = print) -> None:
    """Fetch or verify the raw dataset; skipped while every file is as it was verified."""
    from pdeeg.data.download import download

    raw = config.data.paths.raw

    def stage() -> Stage:
        files = sorted(p for p in raw.rglob("*") if p.is_file()) if raw.is_dir() else []
        return Stage(config, "download", config.data.dataset, [], files)

    current = stage()
    if not force and current.outputs and current.is_current():
        say(f"download: current ({len(current.outputs)} files in {raw})")
        return
    dataset = config.data.dataset
    say(f"download: {dataset.id} {dataset.version} -> {raw}")
    download(config)
    stage().done()


def recordings_step(config: Config, *, force: bool = False, say: Say = print) -> pd.DataFrame:
    """The table of recordings, from participants.tsv and the recordings on disk."""
    from pdeeg.data.bids import list_recordings

    data = config.data
    raw = data.paths.raw
    extension = data.dataset.extension
    inputs = [raw / "participants.tsv", *sorted(raw.rglob(f"*_eeg{extension}"))]
    settings = {
        "dataset": data.dataset,
        "sessions": data.sessions,
        "participants_rename": data.participants_rename,
    }
    stage = Stage(config, "recordings", settings, inputs, [data.paths.recordings])
    if not force and stage.is_current():
        say(f"recordings: current ({data.paths.recordings})")
        return read_recordings(config)
    recordings = list_recordings(raw, data.sessions, rename_columns=data.participants_rename)
    data.paths.recordings.parent.mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(data.paths.recordings, index=False)
    stage.done()
    say(f"recordings: {len(recordings)} -> {data.paths.recordings}")
    return recordings


def preprocess_step(
    config: Config, *, n_jobs: int | None = None, force: bool = False, say: Say = print
) -> None:
    """Clean every recording (each one cached on its inputs and settings) and the QC report."""
    from pdeeg.preprocessing.pipeline import run
    from pdeeg.preprocessing.report import write_report

    recordings = read_recordings(config)
    results = run(config, recordings, n_jobs=n_jobs, force=force, progress=_show(say))
    _statuses(results, "preprocess", say)
    say(f"preprocessing QC report -> {write_report(config, recordings)}")


def psd_step(
    config: Config, *, n_jobs: int = 1, force: bool = False, say: Say = print
) -> pd.DataFrame:
    """The band-power tables; returns the long one."""
    from pdeeg.features.psd import build_psd_table
    from pdeeg.features.tables import psd_wide, table_paths, write_tables

    recordings = read_recordings(config)
    paths = table_paths(config, "psd")
    settings = {"psd": config.psd, "regions": config.data.regions}
    stage = Stage(
        config,
        "psd",
        settings,
        [config.data.paths.recordings, *_cleaned_files(config, recordings)],
        paths.values(),
    )
    if not force and stage.is_current():
        say(f"band power: current ({paths['long']})")
        return pd.read_parquet(paths["long"])
    long = build_psd_table(config, recordings, n_jobs=n_jobs)
    write_tables(config, "psd", long, psd_wide(long))
    stage.done()
    say(f"band power: {len(long)} values -> {paths['long']}")
    return long


def mfdfa_step(
    config: Config, *, n_jobs: int | None = None, force: bool = False, say: Say = print
) -> None:
    """MFDFA runs of every recording (each cached), the feature tables and their QC report."""
    from pdeeg.features.extract import run
    from pdeeg.features.report import write_report
    from pdeeg.features.tables import build_mfdfa_table, mfdfa_wide, table_paths, write_tables

    recordings = read_recordings(config)
    results = run(config, recordings, n_jobs=n_jobs, force=force, progress=_show(say))
    _statuses(results, "MFDFA", say)
    long = build_mfdfa_table(config, recordings)
    paths = write_tables(config, "mfdfa", long, mfdfa_wide(long, config.mfdfa))
    say(f"MFDFA: {len(long)} values -> {paths['long']}")
    psd_path = table_paths(config, "psd")["long"]
    psd = pd.read_parquet(psd_path) if psd_path.is_file() else None
    say(f"MFDFA QC report -> {write_report(config, recordings, long, psd)}")


def curves_step(
    config: Config, *, n_jobs: int | None = None, force: bool = False, say: Say = print
) -> None:
    """h(q), alpha and f(alpha) of every channel, series and fit range."""
    from pdeeg.features.curves import build_curves_table, curves_path

    recordings = read_recordings(config)
    path = curves_path(config)
    settings = without(config.mfdfa, "extraction", "inspection", "features", "poor_fit_r2")
    stage = Stage(
        config,
        "curves",
        settings,
        [config.data.paths.recordings, *_cleaned_files(config, recordings)],
        [path],
    )
    if not force and stage.is_current():
        say(f"h(q) curves: current ({path})")
        return
    jobs = config.mfdfa.extraction.n_jobs if n_jobs is None else n_jobs
    curves = build_curves_table(config, recordings, n_jobs=jobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_parquet(path, index=False)
    stage.done()
    say(f"h(q) curves: {len(curves)} rows -> {path}")


def features_step(
    config: Config, *, n_jobs: int | None = None, force: bool = False, say: Say = print
) -> None:
    """Band power, then MFDFA (whose QC report shows the band powers), then the curves."""
    psd_step(config, force=force, say=say)
    mfdfa_step(config, n_jobs=n_jobs, force=force, say=say)
    curves_step(config, n_jobs=n_jobs, force=force, say=say)


def _stats_stage(config: Config) -> Stage:
    from pdeeg.features.tables import table_paths
    from pdeeg.stats.analysis import result_paths

    data = config.data
    settings = {
        "stats": without(config.stats, "n_jobs", "report", "figures_dir"),
        "sessions": data.sessions,
        "montage": data.dataset.montage,
    }
    inputs = [data.paths.recordings, *(table_paths(config, k)["long"] for k in ("mfdfa", "psd"))]
    return Stage(config, "stats", settings, inputs, result_paths(config).values())


def stats_step(
    config: Config,
    *,
    n_jobs: int | None = None,
    force: bool = False,
    write: bool = True,
    say: Say = print,
) -> None:
    """The cluster tests, stored as tables, and (with ``write``) their report."""
    from pdeeg.stats.analysis import load_results, run_analysis, write_results

    recordings = read_recordings(config)
    tables = _tables(config, "long")
    stage = _stats_stage(config)
    if not force and stage.is_current():
        say("stats: current")
        results = load_results(config, recordings, tables)
    else:
        results = run_analysis(config, recordings, tables, n_jobs=n_jobs)
        paths = write_results(config, results)
        stage.done()
        say(f"stats: {len(results.tests)} tests -> {paths['tests'].parent}")
    if write:
        _write_stats_report(config, results, say)


def _write_stats_report(config: Config, results, say: Say) -> None:
    from pdeeg.features.curves import curves_path
    from pdeeg.stats.report import write_report

    curves = curves_path(config)
    table = pd.read_parquet(curves) if curves.is_file() else None
    if table is None:
        say(f"{curves} not found; the stats report leaves out the h(q) figure")
    say(f"stats report -> {write_report(config, results, table)}")


def summary_path(config: Config):
    dataset = config.data.dataset
    name = f"classification_summary_{dataset.id}-{dataset.version}.parquet"
    return config.data.paths.processed / name


def _classify_stage(config: Config) -> Stage:
    from pdeeg.features.tables import table_paths
    from pdeeg.models.classify import planned
    from pdeeg.models.store import stored_files

    data = config.data
    settings = {
        "model": without(config.model, "n_jobs", "report", "figures_dir", "tracking"),
        "sessions": data.sessions,
        "regions": data.regions,
        "dataset": data.dataset,
    }
    inputs = [data.paths.recordings, *(table_paths(config, k)["wide"] for k in ("mfdfa", "psd"))]
    outputs = [*stored_files(config, planned(config)), summary_path(config)]
    return Stage(config, "classify", settings, inputs, outputs)


def classify_step(
    config: Config,
    *,
    n_jobs: int | None = None,
    force: bool = False,
    write: bool = True,
    say: Say = print,
) -> None:
    """Every evaluation, each tracked in MLflow, stored, and (with ``write``) the report."""
    from pdeeg.features.tables import table_paths
    from pdeeg.models.classify import planned, run_classification, summary_table
    from pdeeg.models.store import load_evaluations, save_evaluations
    from pdeeg.models.tracking import Tracker

    recordings = read_recordings(config)
    tables = _tables(config, "wide")
    stage = _classify_stage(config)
    tracker = None
    if not force and stage.is_current():
        say("classify: current")
        results = load_evaluations(config, recordings, tables, planned(config))
    else:
        inputs = {kind: table_paths(config, kind)["wide"] for kind in ("psd", "mfdfa")}
        tracker = Tracker(config, inputs)

        def show(result) -> None:
            run_id = tracker.log(result)
            say(
                f"{result.task} / {result.feature_set} / {result.model}: balanced accuracy "
                f"{result.observed['balanced_accuracy']:.2f} "
                f"(p = {result.p_values['balanced_accuracy']:.3f}) - MLflow run {run_id}"
            )

        results = run_classification(config, recordings, tables, n_jobs=n_jobs, progress=show)
        save_evaluations(config, results)
        summary_table(results, config).to_parquet(summary_path(config), index=False)
        stage.done()
    if write:
        files = _write_classification_report(config, results, say)
        if tracker is not None:
            tracker.log_summary([*files, summary_path(config)])


def _write_classification_report(config: Config, results, say: Say) -> list:
    from pdeeg.models.classify import summary_table
    from pdeeg.models.report import write_report

    files = write_report(config, results, summary_table(results, config))
    say(f"classification report -> {files[0]}")
    return files


def report_step(config: Config, *, say: Say = print) -> None:
    """Rewrite every report from what the steps stored; nothing is computed again.

    Stored results that are older than their inputs or settings are not reported: the step
    that makes them has to run first.
    """
    from pdeeg.features.report import write_report as write_mfdfa_report
    from pdeeg.models.classify import planned
    from pdeeg.models.store import load_evaluations
    from pdeeg.preprocessing.report import write_report as write_preprocessing_report
    from pdeeg.stats.analysis import load_results

    stale = [
        command
        for command, stage in (("stats", _stats_stage), ("classify", _classify_stage))
        if not stage(config).is_current()
    ]
    if stale:
        raise StepFailed(
            "stored results do not match their inputs or settings; run "
            + " and ".join(f"`pdeeg {command}`" for command in stale)
            + " first"
        )
    recordings = read_recordings(config)
    say(f"preprocessing QC report -> {write_preprocessing_report(config, recordings)}")
    long = _tables(config, "long")
    path = write_mfdfa_report(config, recordings, long["mfdfa"], long["psd"])
    say(f"MFDFA QC report -> {path}")
    _write_stats_report(config, load_results(config, recordings, long), say)
    wide = _tables(config, "wide")
    results = load_evaluations(config, recordings, wide, planned(config))
    _write_classification_report(config, results, say)


def all_steps(config: Config, *, force: bool = False, say: Say = print) -> None:
    """The whole pipeline, each step with its configured parallelism.

    Every step writes its own report, so the reports are those `pdeeg report` would write.
    """
    download_step(config, force=force, say=say)
    recordings_step(config, force=force, say=say)
    preprocess_step(config, force=force, say=say)
    features_step(config, force=force, say=say)
    stats_step(config, force=force, say=say)
    classify_step(config, force=force, say=say)
