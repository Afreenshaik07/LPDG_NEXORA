from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. HISTORICAL REPAIR VS NO-ERROR
# ============================================================

visits = pd.read_csv(
    ROOT / "data" / "field_visits.csv",
    encoding="cp1252",
)

visits["outcome_clean"] = (
    visits["outcome"]
    .astype(str)
    .str.strip()
)

# Use only the two outcomes relevant to the business comparison.
visits = visits[
    visits["outcome_clean"].isin(
        [
            "Fehler behoben",
            "Kein Fehler gefunden",
        ]
    )
].copy()


# The historical analysis used pre-visit meter success.
# Build weekly meter success from the supplied file.
meter = pd.read_csv(
    ROOT / "data" / "meter_read_success.csv",
    encoding="cp1252",
)

meter["week_start"] = pd.to_datetime(
    meter["week_start"],
    errors="coerce",
)

meter["success_rate"] = (
    meter["meters_read"] / meter["meters_expected"]
)

meter["gateway_norm"] = (
    meter["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.strip()
    .str.upper()
)

visits["visited_on"] = pd.to_datetime(
    visits["visited_on"],
    errors="coerce",
)

visits["gateway_norm"] = (
    visits["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.strip()
    .str.upper()
)

visits["visit_week"] = (
    visits["visited_on"]
    - pd.to_timedelta(
        visits["visited_on"].dt.weekday,
        unit="D",
    )
).dt.normalize()


records = []

for _, visit in visits.iterrows():

    gateway = visit["gateway_norm"]
    visit_week = visit["visit_week"]

    history = meter[
        (meter["gateway_norm"] == gateway)
        & (meter["week_start"] < visit_week)
    ].sort_values("week_start")

    if history.empty:
        continue

    recent = history.tail(4)

    records.append(
        {
            "outcome": (
                "Repair"
                if visit["outcome_clean"] == "Fehler behoben"
                else "No error found"
            ),
            "success_rate": recent["success_rate"].mean(),
        }
    )


comparison = pd.DataFrame(records)

if not comparison.empty:
    summary = (
        comparison
        .groupby("outcome")["success_rate"]
        .mean()
        .reindex(
            ["Repair", "No error found"]
        )
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        summary.index,
        summary.values,
    )
    plt.ylabel("Average 4-week meter-read success")
    plt.xlabel("Historical visit outcome")
    plt.title(
        "Historical Performance Before Field Visits"
    )
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        OUT / "01_historical_repair_vs_no_error.png",
        dpi=200,
    )
    plt.close()


# ============================================================
# 2. RISK SCORE VS HISTORICAL REPAIR OUTCOME
# ============================================================

# Use the existing historical cost-analysis output when available.
cost_files = [
    OUT / "risk_score_cost_analysis.csv",
    OUT / "cost_analysis.csv",
    OUT / "10_cost_analysis.csv",
]

cost_path = next(
    (p for p in cost_files if p.exists()),
    None,
)

if cost_path is not None:

    risk = pd.read_csv(cost_path)

    # Try to identify the score and repair outcome columns.
    score_col = next(
        (
            c for c in risk.columns
            if c in [
                "combined_score",
                "visit_priority_score",
                "score",
            ]
        ),
        None,
    )

    outcome_col = next(
        (
            c for c in risk.columns
            if c in [
                "outcome",
                "repair",
                "is_repair",
            ]
        ),
        None,
    )

    if score_col and outcome_col:

        plot_df = risk[
            [score_col, outcome_col]
        ].copy()

        plot_df = plot_df.dropna()

        # Convert common repair labels to readable categories.
        if plot_df[outcome_col].dtype == object:
            plot_df["outcome_label"] = (
                plot_df[outcome_col]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "Fehler behoben": "Repair",
                        "Kein Fehler gefunden": "No error found",
                    }
                )
            )
        else:
            plot_df["outcome_label"] = (
                plot_df[outcome_col]
                .map(
                    {
                        1: "Repair",
                        0: "No error found",
                    }
                )
            )

        groups = []

        for label in ["Repair", "No error found"]:
            values = plot_df.loc[
                plot_df["outcome_label"] == label,
                score_col,
            ]

            if not values.empty:
                groups.append(values)

        if groups:
            plt.figure(figsize=(8, 5))

            positions = range(
                1,
                len(groups) + 1,
            )

            plt.boxplot(
                groups,
                tick_labels=[
                    "Repair",
                    "No error found",
                ][:len(groups)],
            )

            plt.ylabel(
                "Relative visit-priority score"
            )
            plt.title(
                "Priority Score by Historical Outcome"
            )
            plt.grid(
                axis="y",
                alpha=0.25,
            )
            plt.tight_layout()

            plt.savefig(
                OUT / "02_score_vs_historical_outcome.png",
                dpi=200,
            )
            plt.close()


# ============================================================
# 3. TOP 15 RECOMMENDATIONS
# ============================================================

predictions = pd.read_csv(
    ROOT / "predictions.csv"
)

first_week = predictions[
    predictions["week_start"] == "2026-02-02"
].copy()

first_week = first_week.sort_values(
    "rank"
)

if not first_week.empty:

    labels = first_week["gateway_id"].astype(str)

    scores = first_week["score"]

    plt.figure(figsize=(10, 6))

    plt.barh(
        labels[::-1],
        scores[::-1],
    )

    plt.xlabel(
        "Visit-priority score"
    )
    plt.ylabel(
        "Gateway ID"
    )
    plt.title(
        "Top 15 Gateways — Week Starting 2026-02-02"
    )
    plt.xlim(0, 1)
    plt.grid(
        axis="x",
        alpha=0.25,
    )
    plt.tight_layout()

    plt.savefig(
        OUT / "03_top15_gateway_priorities.png",
        dpi=200,
    )
    plt.close()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("MANAGER CHARTS CREATED")
print("=" * 60)

for file in sorted(
    OUT.glob("*.png")
):
    print(file.name)

print()
print(f"Saved in: {OUT}")