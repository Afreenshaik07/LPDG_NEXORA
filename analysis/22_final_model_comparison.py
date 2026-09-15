from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.data import ChallengeData


# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 15

CHALLENGE_WEEKS = pd.to_datetime(
    [
        "2026-02-02",
        "2026-02-09",
        "2026-02-16",
        "2026-02-23",
        "2026-03-02",
        "2026-03-09",
        "2026-03-16",
        "2026-03-23",
    ]
)


# ============================================================
# RISK TRANSFORMATION
# ============================================================

def percentile_risk(
    series: pd.Series,
    higher_is_worse: bool,
) -> pd.Series:
    """
    Convert a feature into a relative 0-1 risk score.
    Higher returned values always mean higher risk.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    # Median imputation only for the purpose of ranking.
    median = values.median()

    if pd.isna(median):
        median = 0.0

    values = values.fillna(median)

    ranks = values.rank(
        pct=True,
        method="average",
    )

    if higher_is_worse:
        return ranks

    return 1.0 - ranks


# ============================================================
# MODEL A — CURRENT PRODUCTION SCORE
# ============================================================

def current_production_score(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    x = frame.copy()

    risk_success = percentile_risk(
        x["success_rate"],
        higher_is_worse=False,
    )

    risk_change = percentile_risk(
        x["success_change"],
        higher_is_worse=False,
    )

    risk_persistence = percentile_risk(
        x["low_success_streak"],
        higher_is_worse=True,
    )

    technical_parts = []

    for column in [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]:
        technical_parts.append(
            percentile_risk(
                x[column],
                higher_is_worse=True,
            )
        )

    technical = pd.concat(
        technical_parts,
        axis=1,
    ).mean(axis=1)

    x["score"] = (
        0.45 * risk_success
        + 0.35 * risk_change
        + 0.10 * risk_persistence
        + 0.10 * technical
    )

    return x.sort_values(
        [
            "score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


# ============================================================
# MODEL B — SUCCESS + DETERIORATION
# ============================================================

def success_deterioration_score(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    x = frame.copy()

    risk_success = percentile_risk(
        x["success_rate"],
        higher_is_worse=False,
    )

    risk_change = percentile_risk(
        x["success_change"],
        higher_is_worse=False,
    )

    x["score"] = (
        0.60 * risk_success
        + 0.40 * risk_change
    )

    return x.sort_values(
        [
            "score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 82)
print("NEXORA 2026 — FINAL MODEL COMPARISON")
print("=" * 82)
print()

data = ChallengeData(
    "data"
)

weekly = data.build_weekly_dataset()

weekly["week_start"] = pd.to_datetime(
    weekly["week_start"],
    errors="coerce",
)

print(
    f"Weekly dataset shape: {weekly.shape}"
)

print(
    f"Available data: "
    f"{weekly['week_start'].min().date()} "
    f"to "
    f"{weekly['week_start'].max().date()}"
)

print()


# ============================================================
# WEEK-BY-WEEK COMPARISON
# ============================================================

all_details = []
summary_rows = []


for decision_week in CHALLENGE_WEEKS:

    print(
        f"Processing {decision_week.date()}..."
    )

    # --------------------------------------------------------
    # STRICT CUTOFF
    #
    # Nothing from the decision Monday or later is used.
    # --------------------------------------------------------

    historical = weekly[
        weekly["week_start"]
        < decision_week
    ].copy()

    if historical.empty:
        print(
            "  No historical data available."
        )
        continue

    # --------------------------------------------------------
    # Latest known information per gateway.
    # --------------------------------------------------------

    latest = (
        historical
        .sort_values(
            [
                "gateway_norm",
                "week_start",
            ]
        )
        .groupby(
            "gateway_norm",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    # --------------------------------------------------------
    # Active gateways only.
    # --------------------------------------------------------

    if "decommissioned_on" in latest.columns:

        latest["decommissioned_on"] = pd.to_datetime(
            latest["decommissioned_on"],
            errors="coerce",
        )

        latest = latest[
            latest["decommissioned_on"].isna()
            |
            (
                latest["decommissioned_on"]
                >= decision_week
            )
        ].copy()

    if len(latest) < TOP_K:

        print(
            f"  Only {len(latest)} eligible gateways."
        )

        continue

    # --------------------------------------------------------
    # MODEL A
    # --------------------------------------------------------

    model_a = current_production_score(
        latest
    )

    model_a["rank_a"] = (
        model_a.index + 1
    )

    model_a_top = (
        model_a
        .head(TOP_K)
        .copy()
    )

    # --------------------------------------------------------
    # MODEL B
    # --------------------------------------------------------

    model_b = success_deterioration_score(
        latest
    )

    model_b["rank_b"] = (
        model_b.index + 1
    )

    model_b_top = (
        model_b
        .head(TOP_K)
        .copy()
    )

    # --------------------------------------------------------
    # TOP-15 GATEWAY SETS
    # --------------------------------------------------------

    ids_a = list(
        model_a_top[
            "gateway_norm"
        ]
    )

    ids_b = list(
        model_b_top[
            "gateway_norm"
        ]
    )

    set_a = set(ids_a)

    set_b = set(ids_b)

    overlap = (
        set_a
        & set_b
    )

    only_a = (
        set_a
        - set_b
    )

    only_b = (
        set_b
        - set_a
    )

    overlap_count = len(
        overlap
    )

    overlap_percentage = (
        overlap_count
        / TOP_K
    )

    # --------------------------------------------------------
    # Rank comparison for shared gateways
    # --------------------------------------------------------

    rank_a_map = dict(
        zip(
            model_a_top[
                "gateway_norm"
            ],
            model_a_top[
                "rank_a"
            ],
        )
    )

    rank_b_map = dict(
        zip(
            model_b_top[
                "gateway_norm"
            ],
            model_b_top[
                "rank_b"
            ],
        )
    )

    shared_rank_changes = []

    for gateway in sorted(
        overlap
    ):

        rank_change = (
            rank_b_map[gateway]
            - rank_a_map[gateway]
        )

        shared_rank_changes.append(
            rank_change
        )

    if shared_rank_changes:

        mean_absolute_rank_change = (
            pd.Series(
                shared_rank_changes
            )
            .abs()
            .mean()
        )

    else:

        mean_absolute_rank_change = 0.0

    # --------------------------------------------------------
    # Save detailed rows
    # --------------------------------------------------------

    for rank in range(
        1,
        TOP_K + 1,
    ):

        gateway_a = ids_a[
            rank - 1
        ]

        gateway_b = ids_b[
            rank - 1
        ]

        all_details.append(
            {
                "decision_week":
                    decision_week.date().isoformat(),

                "rank_position":
                    rank,

                "model_a_gateway":
                    gateway_a,

                "model_b_gateway":
                    gateway_b,

                "same_gateway":
                    gateway_a == gateway_b,
            }
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows.append(
        {
            "decision_week":
                decision_week.date().isoformat(),

            "eligible_gateways":
                len(latest),

            "top15_overlap_count":
                overlap_count,

            "top15_overlap_rate":
                overlap_percentage,

            "model_a_only":
                len(only_a),

            "model_b_only":
                len(only_b),

            "mean_absolute_rank_change_shared":
                mean_absolute_rank_change,
        }
    )

    print(
        f"  Eligible gateways: {len(latest)}"
    )

    print(
        f"  Top-15 overlap: "
        f"{overlap_count}/15 "
        f"({overlap_percentage:.1%})"
    )

    print(
        f"  Model A only: "
        f"{len(only_a)}"
    )

    print(
        f"  Model B only: "
        f"{len(only_b)}"
    )

    print()


# ============================================================
# BUILD OUTPUT TABLES
# ============================================================

details = pd.DataFrame(
    all_details
)

summary = pd.DataFrame(
    summary_rows
)


if summary.empty:
    raise RuntimeError(
        "No challenge-week comparison results were generated."
    )


# ============================================================
# OVERALL STATISTICS
# ============================================================

mean_overlap = (
    summary[
        "top15_overlap_rate"
    ].mean()
)

minimum_overlap = (
    summary[
        "top15_overlap_rate"
    ].min()
)

mean_rank_change = (
    summary[
        "mean_absolute_rank_change_shared"
    ].mean()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 82)
print("OVERALL MODEL COMPARISON")
print("=" * 82)

print()

print(
    f"Challenge weeks evaluated: "
    f"{len(summary)}"
)

print(
    f"Mean top-15 overlap: "
    f"{mean_overlap:.1%}"
)

print(
    f"Minimum top-15 overlap: "
    f"{minimum_overlap:.1%}"
)

print(
    f"Mean absolute rank change "
    f"among shared gateways: "
    f"{mean_rank_change:.2f}"
)

print()

print("-" * 82)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# RECOMMENDATION
# ============================================================

print()
print("-" * 82)
print("MODEL SELECTION INTERPRETATION")
print("-" * 82)

if mean_overlap >= 0.80:

    print(
        "The two models produce highly similar "
        "challenge-week recommendations."
    )

elif mean_overlap >= 0.60:

    print(
        "The two models have substantial but "
        "non-identical recommendation overlap."
    )

else:

    print(
        "The two models produce materially "
        "different recommendations."
    )

print()

print(
    "This analysis uses only information strictly "
    "before each challenge Monday."
)

print(
    "It is a recommendation-stability comparison, "
    "not a hidden-ground-truth score."
)


# ============================================================
# SAVE
# ============================================================

detail_output = (
    OUTPUT_DIR
    / "final_model_comparison_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "final_model_comparison_summary.csv"
)

details.to_csv(
    detail_output,
    index=False,
)

summary.to_csv(
    summary_output,
    index=False,
)


print()
print("=" * 82)
print("OUTPUTS CREATED")
print("=" * 82)

print(detail_output)
print(summary_output)

print()
print("DONE")