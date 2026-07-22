"""Run the full statistical analysis for the Moon/Mars spatial-walk data.

What this does
---------------
1. IMU (data/moon_mars_walk_data/*.csv): loads every session, flags which
   ones are usable (see imu.MIN_SAMPLES_FOR_USABLE), and computes robust
   per-session and across-session statistics for every recorded channel.
   Moon and Mars are POOLED together here, on purpose: the condition of each
   walk lives in the experimenter's session log (per the protocol) and is
   not recoverable from the filenames, which are plain unix timestamps.
2. Questionnaires (data/EuroSpaceCenter-Questionnaires.xlsx): parses the
   SPATIAL WALKING section only (the ROTOR section belongs to a different,
   independent experiment and is excluded), computes item-level statistics,
   per-participant scale scores, Cronbach's alpha, item-total correlations,
   and the correlation between the two scales' scores.
3. IMU <-> questionnaire linkage: NOT performed. There is no participant
   identifier in the IMU filenames, so a given walk session cannot currently
   be attributed to a specific respondent. This needs the session log
   mentioned in the protocol; once available, it is a straightforward join
   on participant id and can be added here.

Outputs
-------
All tables are written as CSV to output/, all figures as PNG to
output/figures/. A plain-text summary is also printed to stdout.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

import imu
import questionnaire as q

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
IMU_DIR = os.path.join(DATA_DIR, "moon_mars_walk_data")
QUESTIONNAIRE_PATH = os.path.join(DATA_DIR, "EuroSpaceCenter-Questionnaires.xlsx")
OUTPUT_DIR = os.path.join(REPO_ROOT, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Single accent colour for one-series magnitude charts; a neutral diverging
# pair (blue - white - red) for correlation heatmaps. Kept minimal and
# consistent across every figure rather than re-picked per chart.
ACCENT = "#3B6FA0"
DIVERGING_CMAP = "RdBu_r"


def _ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def run_imu_analysis() -> None:
    print("\n" + "=" * 70)
    print("IMU (seat inertial recordings) -- data/moon_mars_walk_data/")
    print("=" * 70)

    sessions, quality = imu.load_all_sessions(IMU_DIR)
    quality.to_csv(os.path.join(OUTPUT_DIR, "imu_session_quality.csv"))

    n_total = len(quality)
    n_usable = int(quality["usable"].sum())
    print(f"Sessions found: {n_total} | usable (>= {imu.MIN_SAMPLES_FOR_USABLE} samples, "
          f"all channels present): {n_usable} | excluded: {n_total - n_usable}")
    excluded = quality[~quality["usable"]]
    if len(excluded):
        print("Excluded sessions:")
        print(excluded[["n_samples", "has_all_columns", "nominal_duration_s"]].to_string())

    session_stats = imu.per_session_channel_stats(sessions, quality, usable_only=True)
    session_stats.to_csv(os.path.join(OUTPUT_DIR, "imu_per_session_channel_stats.csv"))

    pooled = imu.aggregate_across_sessions(session_stats)
    pooled.to_csv(os.path.join(OUTPUT_DIR, "imu_pooled_channel_stats.csv"))
    print("\nAcross-session statistics (n = usable sessions, one observation per "
          "session mean), key channels:")
    print(pooled.loc[["acc_mag", "gravity_mag", "gyro_mag"],
                      ["n_sessions", "mean", "median", "sd", "mad",
                       "mean_ci95_low", "mean_ci95_high"]].to_string())

    outliers = imu.flag_outlier_sessions(session_stats, channel="acc_mag", metric="mean")
    outliers.to_csv(os.path.join(OUTPUT_DIR, "imu_outlier_sessions_acc_mag.csv"))
    n_out = int(outliers["is_outlier"].sum())
    print(f"\nSessions with an outlying mean |acceleration| (robust modified z-score > 3.5): {n_out}")
    if n_out:
        print(outliers[outliers["is_outlier"]].to_string())

    _plot_imu(session_stats, quality)


def _plot_imu(session_stats: pd.DataFrame, quality: pd.DataFrame) -> None:
    usable = quality[quality["usable"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(usable["nominal_duration_s"], usable["wallclock_duration_s"],
               color=ACCENT, s=28, alpha=0.85, edgecolor="white", linewidth=0.5)
    lims = [0, max(usable["wallclock_duration_s"].max(), usable["nominal_duration_s"].max()) * 1.05]
    ax.plot(lims, lims, color="#999999", linewidth=1, linestyle="--", label="no gap (y = x)")
    ax.set_xlabel("Nominal duration = n samples / 100 Hz (s)")
    ax.set_ylabel("Wall-clock duration = last - first timestamp (s)")
    ax.set_title("Recording gaps: wall-clock vs. nominal session duration")
    ax.legend(frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "imu_session_duration_gaps.png"), dpi=150)
    plt.close(fig)

    channels = ["acc_mag", "gravity_mag", "gyro_mag"]
    data = [session_stats.xs(c, level="channel")["mean"].dropna().to_numpy() for c in channels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data, labels=channels, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.55)
    ax.set_ylabel("Per-session mean (m/s^2 or rad/s)")
    ax.set_title(f"Distribution of per-session mean signal magnitude (n={len(data[0])} usable sessions)")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "imu_channel_magnitude_boxplot.png"), dpi=150)
    plt.close(fig)


def run_questionnaire_analysis() -> tuple[pd.Series, pd.Series]:
    print("\n" + "=" * 70)
    print("Questionnaires -- data/EuroSpaceCenter-Questionnaires.xlsx (SPATIAL WALKING section)")
    print("=" * 70)

    emotional_df, presence_df = q.load_spatial_walking_questionnaires(QUESTIONNAIRE_PATH)
    print(f"Emotional/somatic scale: {emotional_df.shape[1]} items, "
          f"{emotional_df.notna().any(axis=1).sum()} respondents "
          f"(with >=1 item answered) out of {emotional_df.shape[0]} identifiers")
    print(f"Presence Questionnaire: {presence_df.shape[1]} items, "
          f"{presence_df.notna().any(axis=1).sum()} respondents "
          f"(with >=1 item answered) out of {presence_df.shape[0]} identifiers")

    emotional_items = q.item_stats_table(emotional_df)
    presence_items = q.item_stats_table(presence_df)
    emotional_items.to_csv(os.path.join(OUTPUT_DIR, "questionnaire_emotional_item_stats.csv"))
    presence_items.to_csv(os.path.join(OUTPUT_DIR, "questionnaire_presence_item_stats.csv"))

    emotional_scores = q.scale_scores(emotional_df)
    presence_scores = q.scale_scores(presence_df)
    emotional_scores.to_csv(os.path.join(OUTPUT_DIR, "questionnaire_emotional_scale_scores.csv"))
    presence_scores.to_csv(os.path.join(OUTPUT_DIR, "questionnaire_presence_scale_scores.csv"))

    emotional_rel = q.reliability_report(emotional_df)
    presence_rel = q.reliability_report(presence_df)
    print(f"\nCronbach's alpha, emotional scale (raw items, listwise n="
          f"{emotional_rel['alpha']['n_participants']}): {emotional_rel['alpha']['alpha']:.3f}")
    print(f"Cronbach's alpha, Presence Questionnaire (raw items, no reverse-scoring "
          f"applied -- see questionnaire.py docstring, listwise n="
          f"{presence_rel['alpha']['n_participants']}): {presence_rel['alpha']['alpha']:.3f}")

    emotional_rel["item_total_correlations"].to_frame("item_total_r").to_csv(
        os.path.join(OUTPUT_DIR, "questionnaire_emotional_item_total_r.csv"))
    presence_rel["item_total_correlations"].to_frame("item_total_r").to_csv(
        os.path.join(OUTPUT_DIR, "questionnaire_presence_item_total_r.csv"))

    weak_presence_items = presence_rel["item_total_correlations"][
        presence_rel["item_total_correlations"] < 0.2]
    if len(weak_presence_items):
        print(f"\n{len(weak_presence_items)} Presence Questionnaire item(s) with weak/negative "
              f"item-total correlation (< 0.2) -- candidates for a reverse-scoring check:")
        print(weak_presence_items.to_string())

    corr = q.scale_correlation(emotional_scores["score_mean"], presence_scores["score_mean"])
    print(f"\nEmotional scale score vs. Presence Questionnaire score, n={corr['n']} participants "
          f"with both scores:")
    print(f"  Pearson  r = {corr['pearson_r']:.3f} (95% CI [{corr['pearson_ci95_low']:.3f}, "
          f"{corr['pearson_ci95_high']:.3f}]), p = {corr['pearson_p']:.4f}")
    print(f"  Spearman r = {corr['spearman_r']:.3f}, p = {corr['spearman_p']:.4f}")
    pd.Series(corr).to_frame("value").to_csv(
        os.path.join(OUTPUT_DIR, "questionnaire_emotional_vs_presence_correlation.csv"))

    _plot_questionnaire(emotional_df, presence_df, emotional_items, presence_items,
                        emotional_scores, presence_scores)

    return emotional_scores["score_mean"], presence_scores["score_mean"]


def _truncate_label(label: str, max_len: int = 48) -> str:
    text = label.split(". ", 1)[-1]
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0]
    return cut + "..."


def _bar_with_ci(ax, item_stats: pd.DataFrame, title: str, xlabel: str):
    ordered = item_stats.sort_values("mean")
    y = np.arange(len(ordered))
    ax.barh(y, ordered["mean"], color=ACCENT, alpha=0.85)
    ax.errorbar(ordered["mean"], y, xerr=ordered["sd"], fmt="none",
                ecolor="#333333", elinewidth=1, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_truncate_label(lbl) for lbl in ordered.index], fontsize=7)
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _plot_questionnaire(emotional_df, presence_df, emotional_items, presence_items,
                         emotional_scores, presence_scores) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    _bar_with_ci(ax, emotional_items, "Emotional/somatic scale -- item means (+/- 1 SD)",
                 "Mean rating (0-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "questionnaire_emotional_item_means.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 10))
    _bar_with_ci(ax, presence_items, "Presence Questionnaire -- item means (+/- 1 SD)",
                 "Mean rating (1-7)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "questionnaire_presence_item_means.png"), dpi=150)
    plt.close(fig)

    corr_matrix = presence_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_matrix, cmap=DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_title("Presence Questionnaire -- inter-item correlation (Spearman)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="Spearman r", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "questionnaire_presence_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    paired = pd.concat([
        emotional_scores["score_mean"].rename("emotional"),
        presence_scores["score_mean"].rename("presence"),
    ], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(paired["emotional"], paired["presence"], color=ACCENT, s=32,
               alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Emotional/somatic scale score (mean item rating, 0-5)")
    ax.set_ylabel("Presence Questionnaire score (mean item rating, 1-7)")
    ax.set_title(f"Emotional score vs. Presence score (n={len(paired)} participants)")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "questionnaire_emotional_vs_presence_scatter.png"), dpi=150)
    plt.close(fig)


def main():
    _ensure_dirs()
    run_imu_analysis()
    run_questionnaire_analysis()
    print("\n" + "=" * 70)
    print(f"Tables written to {OUTPUT_DIR}")
    print(f"Figures written to {FIG_DIR}")
    print("=" * 70)
    print(
        "\nNOTE: IMU sessions and questionnaire respondents are NOT linked in this "
        "run -- the IMU filenames carry no participant id, and Moon/Mars condition "
        "for each session is unknown (it lives in the experimenter's session log, "
        "not in these files). All IMU statistics above pool both conditions and all "
        "usable sessions together, as requested. The Presence Questionnaire score "
        "uses raw (non reverse-scored) items -- see analysis/questionnaire.py."
    )


if __name__ == "__main__":
    main()
