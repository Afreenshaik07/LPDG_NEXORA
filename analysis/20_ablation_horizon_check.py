from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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

required = [
    "gateway_norm",
    "week_start",
    "success_rate",
    "success_change",
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

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
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
# RELATIVE RISK
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
# THREE CANDIDATE DESIGNS
# ============================================================

def score_design(
    frame: pd.DataFrame,
    design: str,
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

    # Current production design components.
    risk_persistence = percentile_risk(
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

    technical = pd.concat(
        technical_parts,
        axis=1,
    ).mean(axis=1)

    if design == "A_current_success":

        x["score"] = (
            risk_success
        )

    elif design == "B_success_deterioration":

        x["score"] = (
            0.60 * risk_success
            +
            0.40 * risk_change
        )

    elif design == "C_current_production":

        x["score"] = (
            0.45 * risk_success
            +
            0.35 * risk_change
            +
            0.10 * risk_persistence
            +
            0.10 * technical
        )

    else:
        raise ValueError(
            f"Unknown design: {design}"
        )

    return (
        x
        .sort_values(
            [
                "score",
                "gateway_norm",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


DESIGNS = [
    "A_current_success",
    "B_success_deterioration",
    "C_current_production",
]

LABELS = {
    "A_current_success":
        "A. Current success only",

    "B_success_deterioration":
        "B. Success + deterioration",

    "C_current_production":
        "C. Current production score",
}


# ============================================================
# ANALYSIS
# ============================================================

all_results = []


for horizon in HORIZONS:

    # --------------------------------------------------------
    # Future labels for this horizon.
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

    future_failure_cols = []

    for step in range(
        1,
        horizon + 1,
    ):

        future_success = (
            grouped["success_rate"]
            .shift(-step)
        )

        col = f"future_failure_{step}"

        target[col] = (
            future_success
            < FAILURE_THRESHOLD
        )

        future_failure_cols.append(
            col
        )

    target["future_complete"] = (
        target[
            [
                c
                for c in future_failure_cols
            ]
        ]
        .notna()
        .all(axis=1)
    )

    target_eval = target[
        target["future_complete"]
    ].copy()

    decision_weeks = (
        target_eval["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for decision_week in decision_weeks:

        # ----------------------------------------------------
        # STRICT PRE-MONDAY CUTOFF
        # ----------------------------------------------------

        historical = df[
            df["week_start"] < decision_week
        ].copy()

        if historical.empty:
            continue

        # Latest known feature record per gateway.
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

        # Future labels ONLY for evaluation.
        labels = target_eval[
            target_eval["week_start"] == decision_week
        ][
            [
                "gateway_norm"
            ]
            + future_failure_cols
        ].copy()

        latest = latest.merge(
            labels,
            on="gateway_norm",
            how="inner",
        )

        if len(latest) < TOP_K:
            continue

        total_future_events = int(
            latest[
                future_failure_cols
            ]
            .astype(int)
            .sum()
            .sum()
        )

        if total_future_events == 0:
            continue

        # ----------------------------------------------------
        # TEST THREE DESIGNS
        # ----------------------------------------------------

        for design in DESIGNS:

            ranked = score_design(
                latest,
                design,
            )

            selected = ranked.head(
                TOP_K
            )

            selected_ids = set(
                selected[
                    "gateway_norm"
                ]
            )

            evaluation = latest[
                [
                    "gateway_norm"
                ]
                + future_failure_cols
            ].copy()

            evaluation["selected"] = (
                evaluation[
                    "gateway_norm"
                ]
                .isin(selected_ids)
            )

            missed = 0

            for col in future_failure_cols:

                missed += int(
                    (
                        (~evaluation["selected"])
                        &
                        evaluation[col]
                    ).sum()
                )

            captured = (
                total_future_events
                - missed
            )

            recall = (
                captured
                / total_future_events
            )

            precision = (
                captured
                / (
                    TOP_K
                    * horizon
                )
            )

            visit_cost = (
                TOP_K
                * VISIT_COST
            )

            missed_cost = (
                missed
                * MISSED_FAILURE_COST
            )

            total_cost = (
                visit_cost
                + missed_cost
            )

            all_results.append(
                {
                    "horizon_weeks":
                        horizon,

                    "decision_week":
                        decision_week,

                    "design":
                        design,

                    "design_label":
                        LABELS[design],

                    "future_failure_events":
                        total_future_events,

                    "captured":
                        captured,

                    "missed_failure_weeks":
                        missed,

                    "recall_at_15":
                        recall,

                    "precision_at_15":
                        precision,

                    "total_cost":
                        total_cost,
                }
            )


# ============================================================
# RESULTS
# ============================================================

result = pd.DataFrame(
    all_results
)

if result.empty:
    raise RuntimeError(
        "No results generated."
    )


summary = (
    result
    .groupby(
        [
            "horizon_weeks",
            "design",
            "design_label",
        ]
    )
    .agg(
        weeks_evaluated=(
            "decision_week",
            "nunique",
        ),
        mean_recall_at_15=(
            "recall_at_15",
            "mean",
        ),
        mean_precision_at_15=(
            "precision_at_15",
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
# PRINT SUMMARY
# ============================================================

print()
print("=" * 84)
print("NEXORA 2026 — ABLATION HORIZON CHECK")
print("=" * 84)

for horizon in HORIZONS:

    subset = summary[
        summary["horizon_weeks"]
        == horizon
    ].copy()

    if subset.empty:
        continue

    subset = subset.sort_values(
        "total_cost"
    )

    print()
    print("-" * 84)
    print(
        f"HORIZON: {horizon} WEEK"
        + ("S" if horizon > 1 else "")
    )

    display = subset[
        [
            "design_label",
            "mean_recall_at_15",
            "mean_precision_at_15",
            "total_cost",
            "mean_weekly_cost",
        ]
    ].copy()

    display[
        "mean_recall_at_15"
    ] = display[
        "mean_recall_at_15"
    ].round(3)

    display[
        "mean_precision_at_15"
    ] = display[
        "mean_precision_at_15"
    ].round(3)

    display[
        "mean_weekly_cost"
    ] = display[
        "mean_weekly_cost"
    ].round(0)

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# BEST DESIGN PER HORIZON
# ============================================================

print()
print("=" * 84)
print("BEST DESIGN BY HORIZON")
print("=" * 84)

for horizon in HORIZONS:

    subset = summary[
        summary["horizon_weeks"]
        == horizon
    ]

    if subset.empty:
        continue

    best = subset.loc[
        subset["total_cost"].idxmin()
    ]

    print()
    print(
        f"{horizon}-week horizon:"
    )

    print(
        f"  {best['design_label']}"
    )

    print(
        f"  Cost: €{best['total_cost']:,.0f}"
    )

    print(
        f"  Recall@15: "
        f"{best['mean_recall_at_15']:.1%}"
    )

    print(
        f"  Precision@15: "
        f"{best['mean_precision_at_15']:.1%}"
    )


# ============================================================
# SAVE
# ============================================================

detail_output = (
    OUTPUT_DIR
    / "ablation_horizon_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "ablation_horizon_summary.csv"
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
print("=" * 84)
print("OUTPUTS CREATED")
print("=" * 84)

print(detail_output)
print(summary_output)

print()
print("DONE")