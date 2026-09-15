from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VISITS_PER_WEEK = 15
VISIT_COST_EUR = 380
FAULTY_WEEK_COST_EUR = 600

# Interpretation bands for the ordinal score.
CRITICAL_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.40


def priority_tier(rank: int) -> str:
    """Interpret the required top-15 list by relative weekly priority.

    The score is ordinal, so fixed global thresholds can misleadingly label
    every selected gateway as "Critical". Relative tiers communicate the
    ranking decision without implying calibrated probability.
    """
    if rank <= 5:
        return "Critical"
    if rank <= 10:
        return "High"
    return "Medium"


def build_weekly_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create a compact operations-manager summary from predictions.csv."""
    required = {"week_start", "rank", "gateway_id", "score", "reason"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["priority_tier"] = frame["rank"].astype(int).map(priority_tier)

    rows: list[dict[str, object]] = []

    for week, part in frame.groupby("week_start", sort=True):
        top = part.sort_values("rank").iloc[0]
        rows.append(
            {
                "week_start": week,
                "selected_gateways": len(part),
                "avg_score": round(float(part["score"].mean()), 4),
                "min_score": round(float(part["score"].min()), 4),
                "max_score": round(float(part["score"].max()), 4),
                "critical_count": int((part["priority_tier"] == "Critical").sum()),
                "high_count": int((part["priority_tier"] == "High").sum()),
                "medium_count": int((part["priority_tier"] == "Medium").sum()),
                "weekly_visit_cost_eur": VISITS_PER_WEEK * VISIT_COST_EUR,
                "top_gateway": str(top["gateway_id"]),
                "top_score": round(float(top["score"]), 4),
                "top_priority": str(priority_tier(int(top["rank"]))),
                "top_reason": str(top["reason"]),
            }
        )

    return pd.DataFrame(rows)


def build_cost_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise the fixed visit cost represented by the submitted top-15 plan."""
    weeks = predictions["week_start"].nunique()
    selected_visits = len(predictions)

    return pd.DataFrame(
        [
            {
                "decision_weeks": int(weeks),
                "selected_visits": int(selected_visits),
                "visit_capacity_per_week": VISITS_PER_WEEK,
                "visit_cost_eur": VISIT_COST_EUR,
                "fixed_dispatch_cost_eur": int(selected_visits * VISIT_COST_EUR),
                "faulty_gateway_week_cost_eur": FAULTY_WEEK_COST_EUR,
                "note": (
                    "The €600 figure is the challenge's recurring missed-fault "
                    "cost and is not estimated from predictions.csv."
                ),
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an operational summary from predictions.csv."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("predictions.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/outputs"),
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)

    if predictions.empty:
        raise ValueError("predictions.csv is empty.")

    weekly = build_weekly_summary(predictions)
    cost = build_cost_summary(predictions)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    weekly_path = args.out_dir / "final_weekly_summary.csv"
    cost_path = args.out_dir / "final_cost_summary.csv"

    weekly.to_csv(weekly_path, index=False)
    cost.to_csv(cost_path, index=False)

    print("=" * 72)
    print("NEXORA 2026 - OPERATIONAL SUMMARY")
    print("=" * 72)
    print()
    print(weekly.to_string(index=False))
    print()
    print(cost.to_string(index=False))
    print()
    print("Relative priority tiers:")
    print("  Ranks 1-5   -> Critical")
    print("  Ranks 6-10  -> High")
    print("  Ranks 11-15 -> Medium")
    print("Note: tiers describe relative priority within the required top-15 list;")
    print("the ordinal score is not a calibrated probability.")
    print()
    print(f"Saved: {weekly_path}")
    print(f"Saved: {cost_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
