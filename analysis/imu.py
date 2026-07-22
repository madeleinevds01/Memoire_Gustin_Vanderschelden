"""Loading and statistics for the inertial (IMU) recordings of the seat.

Each file in data/moon_mars_walk_data/ is one walk session (Moon or Mars,
condition unknown from the file itself -- see the module docstring in
run_analysis.py). Columns, per the protocol: gFx/gFy/gFz (acceleration with
gravity), ax/ay/az (linear acceleration without gravity), wx/wy/wz (gyroscope
angular velocity), yaw/pitch/roll (orientation), all at a nominal 100 Hz.

The filenames are raw unix timestamps with no participant identifier, so
these sessions cannot currently be linked to a specific person or
questionnaire response -- that link lives in the experimenter's session log,
which is not part of this dataset (see the protocol, sec. protocol-order).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from stats_utils import describe_series, bootstrap_ci, robust_outlier_flags

EXPECTED_COLUMNS = ["time", "gFx", "gFy", "gFz", "ax", "ay", "az",
                     "wx", "wy", "wz", "yaw", "pitch", "roll"]

# A session is unusable for signal analysis if it is missing the
# accelerometer/orientation channels, or if it stopped after only a few
# seconds (aborted/test recording). This single rule, applied to the 36 raw
# files in data/moon_mars_walk_data, isolates exactly the 32 sessions the
# thesis protocol describes as "complete enough to be analysed" -- it is not
# an arbitrary choice, it is derived from and cross-checked against that count.
MIN_SAMPLES_FOR_USABLE = 500  # ~5 s at the nominal 100 Hz


def list_session_files(data_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def load_session(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    if {"ax", "ay", "az"}.issubset(df.columns):
        df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    if {"gFx", "gFy", "gFz"}.issubset(df.columns):
        df["gravity_mag"] = np.sqrt(df["gFx"] ** 2 + df["gFy"] ** 2 + df["gFz"] ** 2)
    if {"wx", "wy", "wz"}.issubset(df.columns):
        df["gyro_mag"] = np.sqrt(df["wx"] ** 2 + df["wy"] ** 2 + df["wz"] ** 2)
    return df


def session_quality(session_id: str, df: pd.DataFrame) -> dict:
    """Data-quality / usability metadata for one session.

    Reports both the wall-clock duration (last - first timestamp) and the
    nominal duration implied by the sample count at 100 Hz, because the two
    can differ a lot when the recording paused (app backgrounded, etc.) --
    a single large gap inflates wall-clock duration without adding any real
    walking data. Gaps above 1 s are counted and summed separately for the
    same reason.
    """
    n_samples = len(df)
    has_all_columns = set(EXPECTED_COLUMNS).issubset(df.columns)
    dt = df["time"].diff().dt.total_seconds().dropna()

    wallclock_duration_s = float((df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()) if n_samples > 1 else 0.0
    nominal_duration_s = n_samples / 100.0
    gaps = dt[dt > 1.0]

    usable = has_all_columns and n_samples >= MIN_SAMPLES_FOR_USABLE

    return {
        "session_id": session_id,
        "n_samples": n_samples,
        "has_all_columns": has_all_columns,
        "median_dt_s": float(dt.median()) if len(dt) else np.nan,
        "wallclock_duration_s": wallclock_duration_s,
        "nominal_duration_s": nominal_duration_s,
        "n_gaps_gt_1s": int(len(gaps)),
        "total_gap_s": float(gaps.sum()),
        "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "usable": bool(usable),
    }


def load_all_sessions(data_dir: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load every session file and its quality metadata.

    Returns (sessions dict keyed by session id, quality dataframe indexed by
    session id) so callers can filter to `usable` sessions before computing
    channel statistics.
    """
    sessions = {}
    quality_rows = []
    for path in list_session_files(data_dir):
        session_id = os.path.splitext(os.path.basename(path))[0]
        df = load_session(path)
        sessions[session_id] = df
        quality_rows.append(session_quality(session_id, df))
    quality = pd.DataFrame(quality_rows).set_index("session_id")
    return sessions, quality


CHANNELS = ["gFx", "gFy", "gFz", "ax", "ay", "az", "wx", "wy", "wz",
            "yaw", "pitch", "roll", "acc_mag", "gravity_mag", "gyro_mag"]


def per_session_channel_stats(sessions: dict[str, pd.DataFrame], quality: pd.DataFrame,
                               usable_only: bool = True) -> pd.DataFrame:
    """One row per (session, channel) with the robust descriptives of that signal."""
    rows = []
    for session_id, df in sessions.items():
        if usable_only and not quality.loc[session_id, "usable"]:
            continue
        for channel in CHANNELS:
            if channel not in df.columns:
                continue
            stats_dict = describe_series(df[channel])
            stats_dict["session_id"] = session_id
            stats_dict["channel"] = channel
            rows.append(stats_dict)
    return pd.DataFrame(rows).set_index(["session_id", "channel"])


def aggregate_across_sessions(session_channel_stats: pd.DataFrame) -> pd.DataFrame:
    """Treat each usable SESSION as one observation (n=32), not each sample.

    This is the statistically appropriate unit for "how does the average
    walk look", since sample-level pooling would let two nearly-identical
    150 s sessions outweigh one 130 s session for no meaningful reason and
    would understate the between-participant variability that actually
    matters here. For each channel we summarise the distribution, across
    sessions, of that channel's per-session mean -- with a bootstrap CI on
    the group mean since n=32 session-summaries is not large.
    """
    rows = []
    for channel in CHANNELS:
        if channel not in session_channel_stats.index.get_level_values("channel"):
            continue
        per_session_means = session_channel_stats.xs(channel, level="channel")["mean"]
        desc = describe_series(per_session_means)
        ci_low, ci_high = bootstrap_ci(per_session_means, statistic=np.mean)
        desc["mean_ci95_low"] = ci_low
        desc["mean_ci95_high"] = ci_high
        desc["channel"] = channel
        desc["n_sessions"] = per_session_means.dropna().shape[0]
        rows.append(desc)
    return pd.DataFrame(rows).set_index("channel")


def flag_outlier_sessions(session_channel_stats: pd.DataFrame, channel: str = "acc_mag",
                           metric: str = "mean") -> pd.DataFrame:
    """Sessions whose overall signal level is a robust outlier vs the other usable sessions."""
    values = session_channel_stats.xs(channel, level="channel")[metric]
    flags = robust_outlier_flags(values)
    out = values.to_frame(name=metric)
    out["is_outlier"] = flags
    return out.sort_values(metric)
