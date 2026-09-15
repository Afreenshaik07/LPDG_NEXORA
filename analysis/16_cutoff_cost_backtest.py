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
# BUSINESS PARAMETERS
# ============================================================

TOP_K = 15

VISIT_COST = 380

MISSED_FAILURE_COST_PER_WEEK = 600

FAILURE_THRESHOLD = 0.60

FUTURE_HORIZON_WEEKS = 2


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Required input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required_columns = [
    "gateway_norm",
    "week_start",
    "success_rate",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce",
)

if df["week_start"].isna().any():
    raise ValueError(
        "Invalid week_start values found."
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
    .sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# CREATE FUTURE TARGETS
#
# IMPORTANT:
# These targets are used ONLY for evaluation.
# They are NEVER passed into the scoring function.
# ============================================================

target_df = df[
    [
        "gateway_norm",
        "week_start",
        "success_rate",
    ]
].copy()

target_group = target_df.groupby(
    "gateway_norm",
    group_keys=False,
)

target_df["future_week_1_success"] = (
    target_group["success_rate"].shift(-1)
)

target_df["future_week_2_success"] = (
    target_group["success_rate"].shift(-2)
)

target_df["future_week_1_failure"] = (
    target_df["future_week_1_success"]
    < FAILURE_THRESHOLD
)

target_df["future_week_2_failure"] = (
    target_df["future_week_2_success"]
    < FAILURE_THRESHOLD
)

target_df["future_complete"] = (
    target_df["future_week_1_success"].notna()
    &
    target_df["future_week_2_success"].notna()
)


target_evaluation = target_df[
    target_df["future_complete"]
].copy()


# ============================================================
# DECISION WEEKS
# ============================================================

decision_weeks = (
    target_evaluation["week_start"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


# ============================================================
# SIMPLE CURRENT-SUCCESS RANKER
# ============================================================

def rank_by_current_success(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simple reference ranker.

    Lower meter-read success means higher priority.
    """

    x = frame.copy()

    median_success = x["success_rate"].median()

    if pd.isna(median_success):
        median_success = 0.0

    x["rank_score"] = (
        -x["success_rate"].fillna(
            median_success
        )
    )

    x = x.sort_values(
        [
            "rank_score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return x


# ============================================================
# COMBINED DATA SCIENCE RANKER
# ============================================================

def rank_by_combined_score(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Current project Data Science ranking.

    The scoring function receives features only.
    Future target columns are deliberately excluded.
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
        col
        for col in feature_columns
        if col in frame.columns
    ]

    features = frame[
        available
    ].copy()

    x = calculate_risk_scores(
        features
    )

    x = x.sort_values(
        [
            "visit_priority_score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    )

    x["rank_score"] = (
        x["visit_priority_score"]
    )

    return x


# ============================================================
# BACKTEST
# ============================================================

results = []


for decision_week in decision_weeks:

    # --------------------------------------------------------
    # STRICT CUTOFF
    #
    # Only information strictly BEFORE the decision Monday
    # is allowed to influence the ranking.
    # --------------------------------------------------------

    historical = df[
        df["week_start"] < decision_week
    ].copy()

    if historical.empty:
        continue

    # --------------------------------------------------------
    # Latest known feature record per gateway before the
    # decision week.
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
    # Target information is kept separate.
    # It is used ONLY after ranking.
    # --------------------------------------------------------

    target = target_evaluation[
        target_evaluation["week_start"] == decision_week
    ][
        [
            "gateway_norm",
            "future_week_1_failure",
            "future_week_2_failure",
        ]
    ].copy()

    latest = latest.merge(
        target,
        on="gateway_norm",
        how="inner",
    )

    if len(latest) < TOP_K:
        continue

    # ========================================================
    # STRATEGY A — SIMPLE CURRENT SUCCESS
    # ========================================================

    simple_ranked = rank_by_current_success(
        latest
    )

    simple_selected = (
        simple_ranked
        .head(TOP_K)
        .copy()
    )

    # ========================================================
    # STRATEGY B — COMBINED DATA SCIENCE SCORE
    # ========================================================

    combined_ranked = rank_by_combined_score(
        latest
    )

    combined_selected = (
        combined_ranked
        .head(TOP_K)
        .copy()
    )

    # ========================================================
    # TRUE FUTURE FAILURE EVENTS
    # ========================================================

    future_events = latest[
        [
            "gateway_norm",
            "future_week_1_failure",
            "future_week_2_failure",
        ]
    ].copy()

    total_future_failure_events = int(
        future_events[
            [
                "future_week_1_failure",
                "future_week_2_failure",
            ]
        ]
        .astype(int)
        .sum()
        .sum()
    )

    # ========================================================
    # SIMPLE STRATEGY EVALUATION
    # ========================================================

    simple_ids = set(
        simple_selected[
            "gateway_norm"
        ]
    )

    future_events["selected_simple"] = (
        future_events[
            "gateway_norm"
        ]
        .isin(simple_ids)
    )

    simple_missed_week_1 = (
        (
            ~future_events["selected_simple"]
        )
        &
        future_events["future_week_1_failure"]
    ).astype(int)

    simple_missed_week_2 = (
        (
            ~future_events["selected_simple"]
        )
        &
        future_events["future_week_2_failure"]
    ).astype(int)

    simple_missed_failure_weeks = int(
        simple_missed_week_1.sum()
        +
        simple_missed_week_2.sum()
    )

    simple_captured_events = (
        total_future_failure_events
        - simple_missed_failure_weeks
    )

    simple_recall = (
        simple_captured_events
        / total_future_failure_events
        if total_future_failure_events > 0
        else np.nan
    )

    simple_precision = (
        simple_captured_events
        / (
            TOP_K
            * FUTURE_HORIZON_WEEKS
        )
    )

    simple_visit_cost = (
        TOP_K
        * VISIT_COST
    )

    simple_missed_cost = (
        simple_missed_failure_weeks
        * MISSED_FAILURE_COST_PER_WEEK
    )

    simple_total_cost = (
        simple_visit_cost
        + simple_missed_cost
    )

    # ========================================================
    # COMBINED STRATEGY EVALUATION
    # ========================================================

    combined_ids = set(
        combined_selected[
            "gateway_norm"
        ]
    )

    future_events["selected_combined"] = (
        future_events[
            "gateway_norm"
        ]
        .isin(combined_ids)
    )

    combined_missed_week_1 = (
        (
            ~future_events["selected_combined"]
        )
        &
        future_events["future_week_1_failure"]
    ).astype(int)

    combined_missed_week_2 = (
        (
            ~future_events["selected_combined"]
        )
        &
        future_events["future_week_2_failure"]
    ).astype(int)

    combined_missed_failure_weeks = int(
        combined_missed_week_1.sum()
        +
        combined_missed_week_2.sum()
    )

    combined_captured_events = (
        total_future_failure_events
        - combined_missed_failure_weeks
    )

    combined_recall = (
        combined_captured_events
        / total_future_failure_events
        if total_future_failure_events > 0
        else np.nan
    )

    combined_precision = (
        combined_captured_events
        / (
            TOP_K
            * FUTURE_HORIZON_WEEKS
        )
    )

    combined_visit_cost = (
        TOP_K
        * VISIT_COST
    )

    combined_missed_cost = (
        combined_missed_failure_weeks
        * MISSED_FAILURE_COST_PER_WEEK
    )

    combined_total_cost = (
        combined_visit_cost
        + combined_missed_cost
    )

    # ========================================================
    # SAVE WEEK RESULT
    # ========================================================

    results.append(
        {
            "decision_week": decision_week,

            "future_failure_events":
                total_future_failure_events,

            "simple_recall":
                simple_recall,

            "simple_precision":
                simple_precision,

            "simple_missed_failure_weeks":
                simple_missed_failure_weeks,

            "simple_visit_cost":
                simple_visit_cost,

            "simple_missed_cost":
                simple_missed_cost,

            "simple_total_cost":
                simple_total_cost,

            "combined_recall":
                combined_recall,

            "combined_precision":
                combined_precision,

            "combined_missed_failure_weeks":
                combined_missed_failure_weeks,

            "combined_visit_cost":
                combined_visit_cost,

            "combined_missed_cost":
                combined_missed_cost,

            "combined_total_cost":
                combined_total_cost,

            "cost_difference_combined_minus_simple":
                (
                    combined_total_cost
                    - simple_total_cost
                ),
        }
    )


result = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)
print("NEXORA 2026 — LEAKAGE-SAFE COST BACKTEST")
print("=" * 78)

print()

if result.empty:

    print(
        "No valid historical decision weeks were found."
    )

else:

    print(
        f"Decision weeks evaluated: "
        f"{len(result)}"
    )

    print(
        f"Future horizon: "
        f"{FUTURE_HORIZON_WEEKS} weeks"
    )

    print(
        f"Failure definition: "
        f"success_rate < "
        f"{FAILURE_THRESHOLD:.0%}"
    )

    print()
    print("-" * 78)

    print(
        "STRATEGY A — SIMPLE CURRENT-SUCCESS RANKING"
    )

    print(
        f"Mean Recall@15: "
        f"{result['simple_recall'].mean():.1%}"
    )

    print(
        f"Mean Precision: "
        f"{result['simple_precision'].mean():.1%}"
    )

    print(
        f"Total historical cost: "
        f"€{result['simple_total_cost'].sum():,.0f}"
    )

    print(
        f"Mean weekly cost: "
        f"€{result['simple_total_cost'].mean():,.0f}"
    )

    print()
    print("-" * 78)

    print(
        "STRATEGY B — COMBINED DATA SCIENCE SCORE"
    )

    print(
        f"Mean Recall@15: "
        f"{result['combined_recall'].mean():.1%}"
    )

    print(
        f"Mean Precision: "
        f"{result['combined_precision'].mean():.1%}"
    )

    print(
        f"Total historical cost: "
        f"€{result['combined_total_cost'].sum():,.0f}"
    )

    print(
        f"Mean weekly cost: "
        f"€{result['combined_total_cost'].mean():,.0f}"
    )

    print()
    print("-" * 78)

    total_cost_difference = (
        result["combined_total_cost"].sum()
        - result["simple_total_cost"].sum()
    )

    mean_cost_difference = (
        result[
            "cost_difference_combined_minus_simple"
        ].mean()
    )

    print(
        "COMPARISON"
    )

    if total_cost_difference < 0:

        print(
            f"Combined score saves: "
            f"€{abs(total_cost_difference):,.0f}"
        )

    elif total_cost_difference > 0:

        print(
            f"Combined score costs: "
            f"€{total_cost_difference:,.0f} more"
        )

    else:

        print(
            "Combined score and simple ranking "
            "have equal total historical cost."
        )

    print(
        f"Mean weekly cost difference: "
        f"€{mean_cost_difference:,.0f}"
    )

    print()

    display_columns = [
        "decision_week",
        "simple_recall",
        "combined_recall",
        "simple_total_cost",
        "combined_total_cost",
        "cost_difference_combined_minus_simple",
    ]

    display_df = result[
        display_columns
    ].copy()

    display_df["simple_recall"] = (
        display_df["simple_recall"].round(3)
    )

    display_df["combined_recall"] = (
        display_df["combined_recall"].round(3)
    )

    print(
        display_df
        .tail(15)
        .to_string(index=False)
    )


# ============================================================
# SAVE
# ============================================================

output = (
    OUTPUT_DIR
    / "cutoff_cost_backtest.csv"
)

result.to_csv(
    output,
    index=False,
)


print()
print("=" * 78)
print("OUTPUT CREATED")
print("=" * 78)

print(output)

print()
print("DONE")