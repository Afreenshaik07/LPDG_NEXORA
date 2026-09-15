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

VISIT_CAPACITY = 15

VISIT_COST = 380

MISSED_FAILURE_COST = 600

FAILURE_THRESHOLD = 0.60

HORIZON_WEEKS = 2

THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required = [
    "gateway_norm",
    "week_start",
    "success_rate",
]

missing = [
    c for c in required
    if c not in df.columns
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
# CREATE FUTURE FAILURE TARGET
#
# IMPORTANT:
# Target is evaluation-only.
# It is never supplied to the ranking function.
# ============================================================

target = df[
    [
        "gateway_norm",
        "week_start",
        "success_rate",
    ]
].copy()

grouped_target = target.groupby(
    "gateway_norm",
    group_keys=False,
)

target["future_1_success"] = (
    grouped_target["success_rate"].shift(-1)
)

target["future_2_success"] = (
    grouped_target["success_rate"].shift(-2)
)

target["future_1_failure"] = (
    target["future_1_success"]
    < FAILURE_THRESHOLD
)

target["future_2_failure"] = (
    target["future_2_success"]
    < FAILURE_THRESHOLD
)

target["future_complete"] = (
    target["future_1_success"].notna()
    &
    target["future_2_success"].notna()
)


target_eval = target[
    target["future_complete"]
].copy()


# ============================================================
# DECISION WEEKS
# ============================================================

decision_weeks = (
    target_eval["week_start"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

results = []


for decision_week in decision_weeks:

    # --------------------------------------------------------
    # STRICT PRE-MONDAY CUTOFF
    # --------------------------------------------------------

    historical = df[
        df["week_start"] < decision_week
    ].copy()

    if historical.empty:
        continue

    # Latest known information per gateway.
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

    # Future labels only for evaluation.
    labels = target_eval[
        target_eval["week_start"] == decision_week
    ][
        [
            "gateway_norm",
            "future_1_failure",
            "future_2_failure",
        ]
    ].copy()

    latest = latest.merge(
        labels,
        on="gateway_norm",
        how="inner",
    )

    if len(latest) < VISIT_CAPACITY:
        continue

    # --------------------------------------------------------
    # DATA SCIENCE RANKING
    # --------------------------------------------------------

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

    available_features = [
        c for c in feature_columns
        if c in latest.columns
    ]

    scoring_input = latest[
        available_features
    ].copy()

    scored = calculate_risk_scores(
        scoring_input
    )

    scored = scored.sort_values(
        [
            "visit_priority_score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    # Attach labels back for evaluation.
    scored = scored.merge(
        labels,
        on="gateway_norm",
        how="left",
    )

    # --------------------------------------------------------
    # TRUE FAILURE EVENTS IN NEXT TWO WEEKS
    # --------------------------------------------------------

    total_future_failures = int(
        scored[
            [
                "future_1_failure",
                "future_2_failure",
            ]
        ]
        .astype(int)
        .sum()
        .sum()
    )

    # --------------------------------------------------------
    # EACH THRESHOLD
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        qualified = scored[
            scored["visit_priority_score"]
            >= threshold
        ].copy()

        qualified_count = len(
            qualified
        )

        # ----------------------------------------------------
        # Capacity-constrained operational list
        # ----------------------------------------------------

        dispatch = qualified.head(
            VISIT_CAPACITY
        )

        dispatch_ids = set(
            dispatch["gateway_norm"]
        )

        # ----------------------------------------------------
        # Capture / misses
        # ----------------------------------------------------

        evaluation = scored[
            [
                "gateway_norm",
                "future_1_failure",
                "future_2_failure",
            ]
        ].copy()

        evaluation["selected"] = (
            evaluation[
                "gateway_norm"
            ]
            .isin(dispatch_ids)
        )

        missed_week_1 = int(
            (
                (~evaluation["selected"])
                &
                evaluation[
                    "future_1_failure"
                ]
            ).sum()
        )

        missed_week_2 = int(
            (
                (~evaluation["selected"])
                &
                evaluation[
                    "future_2_failure"
                ]
            ).sum()
        )

        missed_failure_weeks = (
            missed_week_1
            + missed_week_2
        )

        captured = (
            total_future_failures
            - missed_failure_weeks
        )

        recall = (
            captured
            / total_future_failures
            if total_future_failures > 0
            else np.nan
        )

        # Precision among dispatched gateways.
        dispatched_failure_events = 0

        for selected_gateway in dispatch_ids:

            selected_rows = evaluation[
                evaluation[
                    "gateway_norm"
                ]
                == selected_gateway
            ]

            dispatched_failure_events += int(
                selected_rows[
                    [
                        "future_1_failure",
                        "future_2_failure",
                    ]
                ]
                .astype(int)
                .sum()
                .sum()
            )

        precision = (
            dispatched_failure_events
            / (
                VISIT_CAPACITY
                * HORIZON_WEEKS
            )
        )

        # ----------------------------------------------------
        # Costs
        # ----------------------------------------------------

        fixed_visit_cost = (
            VISIT_CAPACITY
            * VISIT_COST
        )

        missed_failure_cost = (
            missed_failure_weeks
            * MISSED_FAILURE_COST
        )

        total_cost = (
            fixed_visit_cost
            + missed_failure_cost
        )

        # ----------------------------------------------------
        # Threshold-only operational signal
        # ----------------------------------------------------

        # How many gateways would be above the threshold
        # without applying the 15-gateway submission cap?
        #
        # This is useful for the manager discussion:
        # "How much demand does this threshold create?"
        #

        results.append(
            {
                "decision_week": decision_week,
                "threshold": threshold,

                "qualified_gateways":
                    qualified_count,

                "dispatched_gateways":
                    len(dispatch),

                "future_failure_events":
                    total_future_failures,

                "captured_failure_events":
                    captured,

                "missed_failure_weeks":
                    missed_failure_weeks,

                "recall_at_15":
                    recall,

                "precision_at_15":
                    precision,

                "visit_cost":
                    fixed_visit_cost,

                "missed_failure_cost":
                    missed_failure_cost,

                "total_cost":
                    total_cost,
            }
        )


result = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY BY THRESHOLD
# ============================================================

if result.empty:
    raise RuntimeError(
        "No threshold results were generated."
    )


summary = (
    result
    .groupby("threshold")
    .agg(
        weeks_evaluated=(
            "decision_week",
            "nunique",
        ),
        avg_qualified_gateways=(
            "qualified_gateways",
            "mean",
        ),
        mean_recall_at_15=(
            "recall_at_15",
            "mean",
        ),
        median_recall_at_15=(
            "recall_at_15",
            "median",
        ),
        mean_precision_at_15=(
            "precision_at_15",
            "mean",
        ),
        mean_missed_failure_weeks=(
            "missed_failure_weeks",
            "mean",
        ),
        total_cost=(
            "total_cost",
            "sum",
        ),
        mean_weekly_cost=(
            "total_cost",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# BEST THRESHOLD BY HISTORICAL COST
# ============================================================

best_row = summary.loc[
    summary["total_cost"].idxmin()
]


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("NEXORA 2026 — THRESHOLD COST SENSITIVITY")
print("=" * 80)

print()
print(
    f"Failure definition: "
    f"success_rate < {FAILURE_THRESHOLD:.0%}"
)

print(
    f"Future horizon: "
    f"{HORIZON_WEEKS} weeks"
)

print(
    f"Visit capacity: "
    f"{VISIT_CAPACITY}"
)

print(
    f"Visit cost: "
    f"€{VISIT_COST}"
)

print(
    f"Missed failure cost: "
    f"€{MISSED_FAILURE_COST} per gateway-week"
)

print()
print("-" * 80)

display_columns = [
    "threshold",
    "avg_qualified_gateways",
    "mean_recall_at_15",
    "mean_precision_at_15",
    "mean_missed_failure_weeks",
    "total_cost",
    "mean_weekly_cost",
]

display = summary[
    display_columns
].copy()

display["avg_qualified_gateways"] = (
    display[
        "avg_qualified_gateways"
    ].round(1)
)

display["mean_recall_at_15"] = (
    display[
        "mean_recall_at_15"
    ].round(3)
)

display["mean_precision_at_15"] = (
    display[
        "mean_precision_at_15"
    ].round(3)
)

display["mean_missed_failure_weeks"] = (
    display[
        "mean_missed_failure_weeks"
    ].round(1)
)

print(
    display.to_string(
        index=False
    )
)

print()
print("-" * 80)

print(
    "LOWEST HISTORICAL TOTAL COST"
)

print(
    f"Threshold: "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Total cost: "
    f"€{best_row['total_cost']:,.0f}"
)

print(
    f"Mean weekly cost: "
    f"€{best_row['mean_weekly_cost']:,.0f}"
)

print(
    f"Mean Recall@15: "
    f"{best_row['mean_recall_at_15']:.1%}"
)

print(
    f"Mean Precision@15: "
    f"{best_row['mean_precision_at_15']:.1%}"
)

print()
print(
    "NOTE: The threshold is an operational decision "
    "parameter. The submitted file still requires "
    "exactly 15 gateways per week."
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

detail_output = (
    OUTPUT_DIR
    / "threshold_cost_sensitivity_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "threshold_cost_sensitivity_summary.csv"
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
print("=" * 80)
print("OUTPUTS CREATED")
print("=" * 80)

print(detail_output)
print(summary_output)

print()
print("DONE")