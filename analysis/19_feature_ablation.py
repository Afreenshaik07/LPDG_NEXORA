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
# BUSINESS / EVALUATION PARAMETERS
# ============================================================

TOP_K = 15

FAILURE_THRESHOLD = 0.60

HORIZON_WEEKS = 2

VISIT_COST = 380

MISSED_FAILURE_COST = 600


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
        "Missing columns: "
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
# FUTURE TARGET
# ============================================================

target = df[
    [
        "gateway_norm",
        "week_start",
        "success_rate",
    ]
].copy()

target_group = target.groupby(
    "gateway_norm",
    group_keys=False,
)

target["future_success_1"] = (
    target_group["success_rate"].shift(-1)
)

target["future_success_2"] = (
    target_group["success_rate"].shift(-2)
)

target["future_failure_1"] = (
    target["future_success_1"]
    < FAILURE_THRESHOLD
)

target["future_failure_2"] = (
    target["future_success_2"]
    < FAILURE_THRESHOLD
)

target["future_complete"] = (
    target["future_success_1"].notna()
    &
    target["future_success_2"].notna()
)

target_eval = target[
    target["future_complete"]
].copy()


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
# SCORE DESIGN
# ============================================================

def score_design(
    frame: pd.DataFrame,
    design: str,
) -> pd.DataFrame:

    x = frame.copy()

    # --------------------------------------------------------
    # A. Current success
    # --------------------------------------------------------

    risk_success = percentile_risk(
        x["success_rate"],
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # B. Deterioration
    # --------------------------------------------------------

    risk_change = percentile_risk(
        x["success_change"],
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # C. Persistence
    # --------------------------------------------------------

    risk_persistence = percentile_risk(
        x["low_success_streak"],
        higher_is_worse=True,
    )

    # --------------------------------------------------------
    # D. Technical confirmation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Design-specific score
    # --------------------------------------------------------

    if design == "A_current_success":

        x["ranking_score"] = (
            risk_success
        )

    elif design == "B_success_deterioration":

        x["ranking_score"] = (
            0.60 * risk_success
            +
            0.40 * risk_change
        )

    elif design == "C_success_persistence":

        x["ranking_score"] = (
            0.70 * risk_success
            +
            0.30 * risk_persistence
        )

    elif design == "D_business_only":

        x["ranking_score"] = (
            0.50 * risk_success
            +
            0.40 * risk_change
            +
            0.10 * risk_persistence
        )

    elif design == "E_final_combined":

        x["ranking_score"] = (
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
                "ranking_score",
                "gateway_norm",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# DESIGNS
# ============================================================

DESIGNS = [
    "A_current_success",
    "B_success_deterioration",
    "C_success_persistence",
    "D_business_only",
    "E_final_combined",
]


DESIGN_LABELS = {
    "A_current_success":
        "A. Current success only",

    "B_success_deterioration":
        "B. Success + deterioration",

    "C_success_persistence":
        "C. Success + persistence",

    "D_business_only":
        "D. Business-only combined",

    "E_final_combined":
        "E. Final + technical confirmation",
}


# ============================================================
# BACKTEST
# ============================================================

all_results = []


for decision_week in (
    target_eval["week_start"]
    .drop_duplicates()
    .sort_values()
    .tolist()
):

    # --------------------------------------------------------
    # STRICT CUTOFF
    # --------------------------------------------------------

    historical = df[
        df["week_start"] < decision_week
    ].copy()

    if historical.empty:
        continue

    # Latest known information before decision Monday.
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
    # Future labels are evaluation-only.
    # --------------------------------------------------------

    labels = target_eval[
        target_eval["week_start"] == decision_week
    ][
        [
            "gateway_norm",
            "future_failure_1",
            "future_failure_2",
        ]
    ].copy()

    latest = latest.merge(
        labels,
        on="gateway_norm",
        how="inner",
    )

    if len(latest) < TOP_K:
        continue

    # Total future failure events.
    total_future_events = int(
        latest[
            [
                "future_failure_1",
                "future_failure_2",
            ]
        ]
        .astype(int)
        .sum()
        .sum()
    )

    if total_future_events == 0:
        continue

    # --------------------------------------------------------
    # TEST EVERY DESIGN
    # --------------------------------------------------------

    for design in DESIGNS:

        ranked = score_design(
            latest,
            design,
        )

        selected = ranked.head(
            TOP_K
        )

        selected_ids = set(
            selected["gateway_norm"]
        )

        eval_frame = latest[
            [
                "gateway_norm",
                "future_failure_1",
                "future_failure_2",
            ]
        ].copy()

        eval_frame["selected"] = (
            eval_frame[
                "gateway_norm"
            ].isin(selected_ids)
        )

        missed_1 = int(
            (
                (~eval_frame["selected"])
                &
                eval_frame["future_failure_1"]
            ).sum()
        )

        missed_2 = int(
            (
                (~eval_frame["selected"])
                &
                eval_frame["future_failure_2"]
            ).sum()
        )

        missed = (
            missed_1
            + missed_2
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
                * HORIZON_WEEKS
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
                "decision_week":
                    decision_week,

                "design":
                    design,

                "design_label":
                    DESIGN_LABELS[design],

                "future_failure_events":
                    total_future_events,

                "captured_failure_events":
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
# RESULT TABLE
# ============================================================

result = pd.DataFrame(
    all_results
)

if result.empty:
    raise RuntimeError(
        "No valid backtest results were produced."
    )


summary = (
    result
    .groupby(
        [
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
# RANK DESIGNS BY COST
# ============================================================

summary = (
    summary
    .sort_values(
        "total_cost"
    )
    .reset_index(drop=True)
)

summary["cost_rank"] = (
    summary.index + 1
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 82)
print("NEXORA 2026 — FEATURE ABLATION STUDY")
print("=" * 82)

print()
print(
    "Strict cutoff: information must exist before "
    "the decision Monday."
)

print(
    f"Future failure definition: "
    f"success_rate < {FAILURE_THRESHOLD:.0%}"
)

print(
    f"Evaluation horizon: "
    f"{HORIZON_WEEKS} weeks"
)

print(
    f"Field capacity: "
    f"{TOP_K} gateways"
)

print()
print("-" * 82)

display = summary[
    [
        "cost_rank",
        "design_label",
        "weeks_evaluated",
        "mean_recall_at_15",
        "mean_precision_at_15",
        "total_cost",
        "mean_weekly_cost",
    ]
].copy()

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

display["mean_weekly_cost"] = (
    display[
        "mean_weekly_cost"
    ].round(0)
)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# BEST DESIGN
# ============================================================

best = summary.iloc[0]

print()
print("-" * 82)

print(
    "LOWEST-COST DESIGN"
)

print(
    f"{best['design_label']}"
)

print(
    f"Total historical proxy cost: "
    f"€{best['total_cost']:,.0f}"
)

print(
    f"Mean Recall@15: "
    f"{best['mean_recall_at_15']:.1%}"
)

print(
    f"Mean Precision@15: "
    f"{best['mean_precision_at_15']:.1%}"
)


# ============================================================
# SAVE
# ============================================================

detail_output = (
    OUTPUT_DIR
    / "feature_ablation_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "feature_ablation_summary.csv"
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
print("=" * 82)
print("OUTPUTS CREATED")
print("=" * 82)

print(detail_output)
print(summary_output)

print()
print("DONE")