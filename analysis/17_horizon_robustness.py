from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import calculate_risk_scores


# ============================================================
# PATHS
# ============================================================

INPUT = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week_corrected.csv"
)

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
# PARAMETERS
# ============================================================

TOP_K = 15

FAILURE_THRESHOLD = 0.60

VISIT_COST = 380

MISSED_FAILURE_COST = 600

HORIZONS = [1, 2, 3]


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce",
)

numeric_columns = [
    "success_rate",
    "success_change",
    "low_success_streak",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

df = (
    df
    .dropna(
        subset=[
            "gateway_norm",
            "week_start",
        ]
    )
    .sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def rank_simple(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple reference:
    lowest current success_rate gets highest priority.
    """

    x = frame.copy()

    median_success = x["success_rate"].median()

    if pd.isna(median_success):
        median_success = 0.0

    x["ranking_score"] = (
        -x["success_rate"].fillna(
            median_success
        )
    )

    return (
        x
        .sort_values(
            [
                "ranking_score",
                "gateway_norm",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


def rank_combined(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Current Data Science score.
    """

    feature_columns = [
        "gateway_norm",
        "success_rate",
        "success_change",
        "low_success_streak",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]

    available = [
        c
        for c in feature_columns
        if c in frame.columns
    ]

    x = calculate_risk_scores(
        frame[available].copy()
    )

    return (
        x
        .sort_values(
            [
                "visit_priority_score",
                "gateway_norm",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


# ============================================================
# RUN HORIZON ANALYSIS
# ============================================================

all_results = []

for horizon in HORIZONS:

    # --------------------------------------------------------
    # Build future failure labels for this horizon.
    # --------------------------------------------------------

    target = df[
        [
            "gateway_norm",
            "week_start",
            "success_rate",
        ]
    ].copy()

    grouped = target.groupby(
        "gateway_norm",
        group_keys=False,
    )

    future_failure_columns = []

    for step in range(1, horizon + 1):

        col = f"future_failure_{step}"

        target_success = (
            grouped["success_rate"]
            .shift(-step)
        )

        target[col] = (
            target_success
            < FAILURE_THRESHOLD
        )

        future_failure_columns.append(
            col
        )

    # Need the full future horizon to exist.
    complete = target.copy()

    complete["future_complete"] = (
        complete[
            future_failure_columns
        ]
        .notna()
        .all(axis=1)
    )

    # --------------------------------------------------------
    # Only complete target rows are evaluation weeks.
    # --------------------------------------------------------

    target_eval = complete[
        complete["future_complete"]
    ].copy()

    decision_weeks = (
        target_eval["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    horizon_results = []

    for decision_week in decision_weeks:

        # ----------------------------------------------------
        # STRICT CUTOFF:
        # only data before decision Monday.
        # ----------------------------------------------------

        historical = df[
            df["week_start"] < decision_week
        ].copy()

        if historical.empty:
            continue

        # Latest known feature record for each gateway.
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

        # Future labels are kept separate from features.
        labels = target_eval[
            target_eval["week_start"] == decision_week
        ][
            [
                "gateway_norm"
            ]
            + future_failure_columns
        ].copy()

        latest = latest.merge(
            labels,
            on="gateway_norm",
            how="inner",
        )

        if len(latest) < TOP_K:
            continue

        # ----------------------------------------------------
        # SIMPLE STRATEGY
        # ----------------------------------------------------

        simple = rank_simple(
            latest
        )

        simple_selected = (
            simple
            .head(TOP_K)
            .copy()
        )

        simple_ids = set(
            simple_selected[
                "gateway_norm"
            ]
        )

        # ----------------------------------------------------
        # COMBINED STRATEGY
        # ----------------------------------------------------

        combined = rank_combined(
            latest
        )

        combined_selected = (
            combined
            .head(TOP_K)
            .copy()
        )

        combined_ids = set(
            combined_selected[
                "gateway_norm"
            ]
        )

        # ----------------------------------------------------
        # FUTURE FAILURE EVENTS
        # ----------------------------------------------------

        label_frame = latest[
            [
                "gateway_norm"
            ]
            + future_failure_columns
        ].copy()

        total_failure_events = int(
            label_frame[
                future_failure_columns
            ]
            .astype(int)
            .sum()
            .sum()
        )

        # ----------------------------------------------------
        # SIMPLE EVALUATION
        # ----------------------------------------------------

        label_frame["selected_simple"] = (
            label_frame[
                "gateway_norm"
            ]
            .isin(simple_ids)
        )

        simple_missed = 0

        for col in future_failure_columns:

            simple_missed += int(
                (
                    (~label_frame["selected_simple"])
                    &
                    label_frame[col]
                ).sum()
            )

        simple_captured = (
            total_failure_events
            - simple_missed
        )

        simple_recall = (
            simple_captured
            / total_failure_events
            if total_failure_events > 0
            else np.nan
        )

        simple_precision = (
            simple_captured
            / (
                TOP_K
                * horizon
            )
            if TOP_K * horizon > 0
            else np.nan
        )

        simple_visit_cost = (
            TOP_K
            * VISIT_COST
        )

        simple_missed_cost = (
            simple_missed
            * MISSED_FAILURE_COST
        )

        simple_total_cost = (
            simple_visit_cost
            + simple_missed_cost
        )

        # ----------------------------------------------------
        # COMBINED EVALUATION
        # ----------------------------------------------------

        label_frame["selected_combined"] = (
            label_frame[
                "gateway_norm"
            ]
            .isin(combined_ids)
        )

        combined_missed = 0

        for col in future_failure_columns:

            combined_missed += int(
                (
                    (~label_frame["selected_combined"])
                    &
                    label_frame[col]
                ).sum()
            )

        combined_captured = (
            total_failure_events
            - combined_missed
        )

        combined_recall = (
            combined_captured
            / total_failure_events
            if total_failure_events > 0
            else np.nan
        )

        combined_precision = (
            combined_captured
            / (
                TOP_K
                * horizon
            )
            if TOP_K * horizon > 0
            else np.nan
        )

        combined_visit_cost = (
            TOP_K
            * VISIT_COST
        )

        combined_missed_cost = (
            combined_missed
            * MISSED_FAILURE_COST
        )

        combined_total_cost = (
            combined_visit_cost
            + combined_missed_cost
        )

        horizon_results.append(
            {
                "decision_week": decision_week,

                "future_failure_events":
                    total_failure_events,

                "simple_recall":
                    simple_recall,

                "simple_precision":
                    simple_precision,

                "simple_missed_failure_weeks":
                    simple_missed,

                "simple_total_cost":
                    simple_total_cost,

                "combined_recall":
                    combined_recall,

                "combined_precision":
                    combined_precision,

                "combined_missed_failure_weeks":
                    combined_missed,

                "combined_total_cost":
                    combined_total_cost,

                "cost_difference":
                    (
                        combined_total_cost
                        - simple_total_cost
                    ),
            }
        )

    horizon_df = pd.DataFrame(
        horizon_results
    )

    if horizon_df.empty:
        continue

    all_results.append(
        horizon_df.assign(
            horizon_weeks=horizon
        )
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

if not all_results:
    raise RuntimeError(
        "No valid horizon results were produced."
    )

result = pd.concat(
    all_results,
    ignore_index=True,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print(
    "NEXORA 2026 — HORIZON ROBUSTNESS ANALYSIS"
)
print("=" * 78)

for horizon in HORIZONS:

    subset = result[
        result["horizon_weeks"]
        == horizon
    ]

    if subset.empty:
        continue

    print()
    print("-" * 78)
    print(
        f"HORIZON: {horizon} WEEK"
        + ("S" if horizon > 1 else "")
    )

    print()

    print(
        "Simple Recall@15: "
        f"{subset['simple_recall'].mean():.1%}"
    )

    print(
        "Combined Recall@15: "
        f"{subset['combined_recall'].mean():.1%}"
    )

    print()

    print(
        "Simple Precision@15: "
        f"{subset['simple_precision'].mean():.1%}"
    )

    print(
        "Combined Precision@15: "
        f"{subset['combined_precision'].mean():.1%}"
    )

    print()

    print(
        "Simple total historical cost: "
        f"€{subset['simple_total_cost'].sum():,.0f}"
    )

    print(
        "Combined total historical cost: "
        f"€{subset['combined_total_cost'].sum():,.0f}"
    )

    cost_difference = (
        subset["combined_total_cost"].sum()
        -
        subset["simple_total_cost"].sum()
    )

    if cost_difference < 0:

        print(
            "Combined score savings: "
            f"€{abs(cost_difference):,.0f}"
        )

    elif cost_difference > 0:

        print(
            "Combined score extra cost: "
            f"€{cost_difference:,.0f}"
        )

    else:

        print(
            "No cost difference."
        )


# ============================================================
# OVERALL SUMMARY TABLE
# ============================================================

summary_rows = []

for horizon in HORIZONS:

    subset = result[
        result["horizon_weeks"]
        == horizon
    ]

    if subset.empty:
        continue

    simple_total = (
        subset[
            "simple_total_cost"
        ].sum()
    )

    combined_total = (
        subset[
            "combined_total_cost"
        ].sum()
    )

    summary_rows.append(
        {
            "horizon_weeks": horizon,

            "weeks_evaluated":
                len(subset),

            "simple_recall":
                subset[
                    "simple_recall"
                ].mean(),

            "combined_recall":
                subset[
                    "combined_recall"
                ].mean(),

            "simple_precision":
                subset[
                    "simple_precision"
                ].mean(),

            "combined_precision":
                subset[
                    "combined_precision"
                ].mean(),

            "simple_total_cost":
                simple_total,

            "combined_total_cost":
                combined_total,

            "combined_minus_simple":
                (
                    combined_total
                    - simple_total
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)

print()
print("=" * 78)
print("HORIZON SUMMARY")
print("=" * 78)

display_summary = summary.copy()

for column in [
    "simple_recall",
    "combined_recall",
    "simple_precision",
    "combined_precision",
]:
    display_summary[column] = (
        display_summary[column]
        .round(3)
    )

print(
    display_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

detail_output = (
    OUTPUT_DIR
    / "horizon_robustness_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "horizon_robustness_summary.csv"
)

result.to_csv(
    detail_output,
    index=False,
)

summary.to_csv(
    summary_output,
    index=False,
)


print()
print("=" * 78)
print("OUTPUTS CREATED")
print("=" * 78)

print(detail_output)
print(summary_output)

print()
print("DONE")