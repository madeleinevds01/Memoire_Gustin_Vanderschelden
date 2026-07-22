"""Shared statistical helpers used by both the IMU and questionnaire analyses.

Everything here favours robust estimators (median, MAD, trimmed mean, IQR)
over the plain mean/SD, and reports both. That way a handful of extreme
samples (a sensor glitch, a participant who answered every item with a 5)
cannot silently dominate the summary the way they would with mean/SD alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_series(x: pd.Series) -> dict:
    """Return classic + robust descriptive statistics for one numeric series.

    NaNs are dropped before computing anything; n / n_valid / n_missing let
    the caller see how much data actually went into the numbers.
    """
    x = pd.Series(x).astype(float)
    n = len(x)
    valid = x.dropna()
    n_valid = len(valid)
    out = {
        "n": n,
        "n_valid": n_valid,
        "n_missing": n - n_valid,
    }
    if n_valid == 0:
        out.update({k: np.nan for k in [
            "mean", "sd", "median", "mad", "trimmed_mean_10",
            "q1", "q3", "iqr", "min", "max", "skewness", "kurtosis",
        ]})
        return out

    q1, median, q3 = np.percentile(valid, [25, 50, 75])
    mad = stats.median_abs_deviation(valid, scale="normal") if n_valid > 1 else 0.0
    trimmed = stats.trim_mean(valid, 0.1) if n_valid >= 5 else float(valid.mean())

    out.update({
        "mean": float(valid.mean()),
        "sd": float(valid.std(ddof=1)) if n_valid > 1 else 0.0,
        "median": float(median),
        "mad": float(mad),
        "trimmed_mean_10": float(trimmed),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "skewness": float(stats.skew(valid)) if n_valid > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(valid)) if n_valid > 3 else np.nan,
    })
    return out


def bootstrap_ci(x: pd.Series, statistic=np.mean, n_resamples: int = 5000,
                  confidence: float = 0.95, random_state: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic (mean or median by default).

    Used instead of a normal-theory CI because the inputs here (session
    summaries, Likert scale scores) are small samples that are not
    guaranteed to be normally distributed.
    """
    valid = pd.Series(x).astype(float).dropna().to_numpy()
    if len(valid) < 3:
        return (np.nan, np.nan)
    res = stats.bootstrap(
        (valid,), statistic, n_resamples=n_resamples,
        confidence_level=confidence, method="BCa",
        random_state=random_state,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def robust_outlier_flags(x: pd.Series, threshold: float = 3.5) -> pd.Series:
    """Flag outliers using the median/MAD modified z-score (Iglewicz & Hoaglin).

    More robust than a mean/SD z-score because the median and MAD are
    themselves insensitive to the outliers being searched for.
    """
    x = pd.Series(x).astype(float)
    median = x.median()
    mad = stats.median_abs_deviation(x.dropna(), scale="normal")
    if mad == 0 or np.isnan(mad):
        return pd.Series(False, index=x.index)
    modified_z = 0.6745 * (x - median) / mad
    return modified_z.abs() > threshold


def cronbach_alpha(item_df: pd.DataFrame) -> dict:
    """Cronbach's alpha for a participants-by-items matrix.

    Uses listwise deletion (only participants who answered every item in
    `item_df` are used) because the covariance matrix needs to be computed
    on a common set of respondents. The number of participants actually used
    is returned alongside alpha so this is never silently approximate.
    """
    complete = item_df.dropna(axis=0, how="any")
    k = item_df.shape[1]
    n = complete.shape[0]
    if n < 2 or k < 2:
        return {"alpha": np.nan, "n_participants": n, "n_items": k}
    item_variances = complete.var(axis=0, ddof=1)
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return {"alpha": np.nan, "n_participants": n, "n_items": k}
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return {"alpha": float(alpha), "n_participants": n, "n_items": k}


def item_total_correlations(item_df: pd.DataFrame) -> pd.Series:
    """Corrected item-total correlation: each item vs the sum of the OTHER items.

    A low or negative value flags an item that does not covary with the rest
    of the scale (candidate for miscoding, e.g. an unnoticed reverse-worded
    item, or a genuinely different construct).
    """
    complete = item_df.dropna(axis=0, how="any")
    correlations = {}
    total = complete.sum(axis=1)
    for col in complete.columns:
        rest = total - complete[col]
        if complete[col].std(ddof=1) == 0 or rest.std(ddof=1) == 0:
            correlations[col] = np.nan
        else:
            correlations[col] = complete[col].corr(rest)
    return pd.Series(correlations, name="item_total_r")
