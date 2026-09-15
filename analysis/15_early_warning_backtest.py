from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "analysis"
    / "outputs"
    / "early_warning_target.csv"
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
# CONFIG
# ============================================================

TOP_K = 15

MIN_HISTORY_WEEKS = 4


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

df["week_start"] = pd.to_datetime(
    df["week_start"]
)

numeric_columns = [
    "success_rate",
    "success_change",
    "low_success_streak",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "future_failure",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


df = (
    df
    .sort_values(
        [
            "week_start",
            "gateway_norm",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# RELATIVE RISK FUNCTION
# ============================================================

def percentile_risk(
    series: pd.Series,
    higher_is_worse: bool,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

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
# SCORE ONE DECISION WEEK
# ============================================================

def score_week(
    history: pd.DataFrame,
) -> pd.DataFrame:

    x = history.copy()

    x["risk_success"] = percentile_risk(
        x["success_rate"],
        higher_is_worse=False,
    )

    x["risk_success_change"] = percentile_risk(
        x["success_change"],
        higher_is_worse=False,
    )

    x["risk_persistence"] = percentile_risk(
        x["low_success_streak"],
        higher_is_worse=True,
    )

    technical_parts = []

    for col in [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]:

        technical_parts.append(
            percentile_risk(
                x[col],
                higher_is_worse=True,
            )
        )

    x["technical_confirmation"] = (
        pd.concat(
            technical_parts,
            axis=1,
        )
        .mean(axis=1)
    )

    x["visit_priority_score"] = (
        0.45 * x["risk_success"]
        + 0.35 * x["risk_success_change"]
        + 0.10 * x["risk_persistence"]
        + 0.10 * x["technical_confirmation"]
    )

    return x


# ============================================================
# HISTORICAL FORWARD BACKTEST
# ============================================================

weeks = (
    df["week_start"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

results = []

for decision_week in weeks:

    # We need a current decision record for this week,
    # and future target information already created in
    # early_warning_target.csv.
    current = df[
        df["week_start"] == decision_week
    ].copy()

    if current.empty:
        continue

    # Require enough prior history for a meaningful
    # decision.
    previous_weeks = [
        w for w in weeks
        if w < decision_week
    ]

    if len(previous_weeks) < MIN_HISTORY_WEEKS:
        continue

    scored = score_week(
        current
    )

    scored = scored.dropna(
        subset=[
            "visit_priority_score",
            "future_failure",
        ]
    )

    if scored.empty:
        continue

    scored = scored.sort_values(
        [
            "visit_priority_score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    )

    selected = scored.head(
        TOP_K
    )

    future_failures = int(
        scored["future_failure"].sum()
    )

    captured = int(
        selected["future_failure"].sum()
    )

    recall = (
        captured / future_failures
        if future_failures > 0
        else np.nan
    )

    precision = (
        captured / len(selected)
        if len(selected) > 0
        else np.nan
    )

    results.append(
        {
            "decision_week": decision_week,
            "selected": len(selected),
            "future_failures_in_week": future_failures,
            "future_failures_captured": captured,
            "recall_at_15": recall,
            "precision_at_15": precision,
            "mean_selected_score": selected[
                "visit_priority_score"
            ].mean(),
        }
    )


result_df = pd.DataFrame(
    results
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 72)
print("NEXORA 2026 — EARLY WARNING TOP-15 BACKTEST")
print("=" * 72)

print()
print(
    f"Decision weeks evaluated: "
    f"{len(result_df)}"
)

print(
    f"Top-K per week: "
    f"{TOP_K}"
)

if not result_df.empty:

    print()
    print(
        "Mean Recall@15: "
        f"{result_df['recall_at_15'].mean():.1%}"
    )

    print(
        "Median Recall@15: "
        f"{result_df['recall_at_15'].median():.1%}"
    )

    print(
        "Mean Precision@15: "
        f"{result_df['precision_at_15'].mean():.1%}"
    )

    print(
        "Median Precision@15: "
        f"{result_df['precision_at_15'].median():.1%}"
    )

    print()
    print(
        result_df
        .tail(10)
        .round(3)
        .to_string(index=False)
    )


# ============================================================
# COMPARE WITH SIMPLE <60% RULE
# ============================================================

simple_rows = []

for decision_week in weeks:

    current = df[
        df["week_start"] == decision_week
    ].copy()

    current = current.dropna(
        subset=[
            "future_failure",
            "success_rate",
        ]
    )

    if len(current) < TOP_K:
        continue

    flagged = current[
        current["success_rate"] < 0.60
    ]

    # Among the top-15-capacity setup, select the
    # worst current success values.
    simple_top15 = (
        current
        .sort_values(
            [
                "success_rate",
                "gateway_norm",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .head(TOP_K)
    )

    future_failures = int(
        current["future_failure"].sum()
    )

    captured = int(
        simple_top15["future_failure"].sum()
    )

    recall = (
        captured / future_failures
        if future_failures > 0
        else np.nan
    )

    precision = (
        captured / len(simple_top15)
        if len(simple_top15) > 0
        else np.nan
    )

    simple_rows.append(
        {
            "decision_week": decision_week,
            "recall_at_15": recall,
            "precision_at_15": precision,
            "threshold_flagged": len(flagged),
        }
    )


simple_df = pd.DataFrame(
    simple_rows
)


if not simple_df.empty:

    print()
    print("=" * 72)
    print("SIMPLE CURRENT-SUCCESS TOP-15 BASELINE")
    print("=" * 72)

    print()
    print(
        "Mean Recall@15: "
        f"{simple_df['recall_at_15'].mean():.1%}"
    )

    print(
        "Median Recall@15: "
        f"{simple_df['recall_at_15'].median():.1%}"
    )

    print(
        "Mean Precision@15: "
        f"{simple_df['precision_at_15'].mean():.1%}"
    )

    print(
        "Median Precision@15: "
        f"{simple_df['precision_at_15'].median():.1%}"
    )


# ============================================================
# SAVE
# ============================================================

output_path = (
    OUTPUT_DIR
    / "early_warning_backtest.csv"
)

result_df.to_csv(
    output_path,
    index=False,
)


simple_output = (
    OUTPUT_DIR
    / "early_warning_simple_baseline.csv"
)

simple_df.to_csv(
    simple_output,
    index=False,
)


print()
print("=" * 72)
print("OUTPUTS CREATED")
print("=" * 72)

print(output_path)
print(simple_output)

print()
print("DONE")