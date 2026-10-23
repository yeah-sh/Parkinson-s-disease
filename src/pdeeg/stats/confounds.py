"""Age and sex: are the groups different, and the covariates for the adjusted tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pdeeg.stats.features import StatsError

# How sex enters a regression; the recordings table codes it as in participants.tsv.
SEX_CODES = {"m": 1.0, "f": 0.0}


def covariate_matrix(rows: pd.DataFrame, covariates: Sequence[str]) -> np.ndarray:
    """One column per covariate for the rows of the recordings table ``rows``.

    ``sex`` is coded by ``SEX_CODES``; any other covariate must be numeric. A missing value is
    an error, because a participant without one cannot be adjusted.
    """
    columns = []
    for name in covariates:
        if name not in rows.columns:
            raise StatsError(f"covariate {name!r} is not a column of the recordings table")
        values = rows[name]
        if name == "sex":
            unknown = sorted(set(values.dropna()) - set(SEX_CODES))
            if unknown:
                raise StatsError(f"sex: no code for {unknown}; known are {list(SEX_CODES)}")
            values = values.map(SEX_CODES)
        values = pd.to_numeric(values, errors="raise").astype(float)
        if values.isna().any():
            raise StatsError(f"covariate {name!r} is missing for some participants")
        columns.append(values.to_numpy())
    return np.column_stack(columns) if columns else np.empty((len(rows), 0))


def compare_age(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Welch's t-test of the ages of two groups, with their means and standard deviations."""
    a, b = a.astype(float), b.astype(float)
    test = scipy_stats.ttest_ind(a, b, equal_var=False)
    return {
        "variable": "age",
        "a": f"{a.mean():.1f} ± {a.std(ddof=1):.1f}",
        "b": f"{b.mean():.1f} ± {b.std(ddof=1):.1f}",
        "test": "Welch's t-test",
        "statistic": f"t({test.df:.1f}) = {test.statistic:.2f}",
        "p": float(test.pvalue),
    }


def compare_sex(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Fisher's exact test of the sex counts of two groups."""
    # Rows are the sexes, columns the groups.
    counts = {code: (int((a == code).sum()), int((b == code).sum())) for code in SEX_CODES}
    test = scipy_stats.fisher_exact([counts[code] for code in SEX_CODES])
    return {
        "variable": "sex",
        "a": f"{counts['m'][0]} m / {counts['f'][0]} f",
        "b": f"{counts['m'][1]} m / {counts['f'][1]} f",
        "test": "Fisher's exact test",
        "statistic": f"odds ratio = {test.statistic:.2f}",
        "p": float(test.pvalue),
    }


def compare_groups(rows: pd.DataFrame) -> pd.DataFrame:
    """Age and sex of the two sides of the between-group comparison, and tests of the difference.

    ``rows`` is :func:`pdeeg.stats.features.between_recordings`: side ``a`` against ``b``.
    """
    a, b = rows[rows["side"] == "a"], rows[rows["side"] == "b"]
    return pd.DataFrame([compare_age(a["age"], b["age"]), compare_sex(a["sex"], b["sex"])])
