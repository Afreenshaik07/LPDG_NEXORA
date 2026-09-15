from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "historical_backtest.csv"
)

VISIT_COST = 380
MISSED_FAILURE_WEEKLY_COST = 600


def main():

    print("=" * 70)
    print("STEP 10 — COST AND RISK-BAND ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # Risk bands
    # --------------------------------------------------------

    bins = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]

    labels = [
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
    ]

    df["risk_band"] = pd.cut(
        df["combined_score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    # --------------------------------------------------------
    # Empirical probability of genuine repair
    # --------------------------------------------------------

    summary = (
        df.groupby(
            "risk_band",
            observed=False,
        )
        .agg(
            visits=("outcome", "count"),
            genuine_repairs=(
                "outcome",
                lambda x: (
                    x == "Fehler behoben"
                ).sum(),
            ),
            no_problem=(
                "outcome",
                lambda x: (
                    x
                    == "Kein Fehler gefunden"
                ).sum(),
            ),
            mean_score=(
                "combined_score",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["repair_rate"] = (
        summary["genuine_repairs"]
        / summary["visits"]
    )

    summary["no_problem_rate"] = (
        summary["no_problem"]
        / summary["visits"]
    )

    # Expected loss from NOT visiting,
    # using empirical repair rate.
    summary["expected_missed_cost"] = (
        summary["repair_rate"]
        * MISSED_FAILURE_WEEKLY_COST
    )

    summary["visit_cost"] = VISIT_COST

    summary["visit_economically_preferred"] = (
        summary["expected_missed_cost"]
        > VISIT_COST
    )

    print("\n")
    print("=" * 70)
    print("RISK BANDS")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("THRESHOLD ECONOMIC ANALYSIS")
    print("=" * 70)

    threshold_rows = []

    for threshold in [
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        selected = (
            df["combined_score"]
            >= threshold
        )

        selected_data = df[selected]

        if selected_data.empty:
            continue

        repair_rate = (
            (
                selected_data["outcome"]
                == "Fehler behoben"
            ).mean()
        )

        no_problem_rate = (
            (
                selected_data["outcome"]
                == "Kein Fehler gefunden"
            ).mean()
        )

        expected_missed_cost = (
            repair_rate
            * MISSED_FAILURE_WEEKLY_COST
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "selected_records": len(
                    selected_data
                ),
                "repair_rate": repair_rate,
                "no_problem_rate": (
                    no_problem_rate
                ),
                "expected_missed_cost": (
                    expected_missed_cost
                ),
                "visit_cost": VISIT_COST,
                "missed_cost_gt_visit": (
                    expected_missed_cost
                    > VISIT_COST
                ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "analysis"
        / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        output_dir
        / "risk_bands.csv"
    )

    threshold_file = (
        output_dir
        / "cost_threshold_analysis.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    threshold_df.to_csv(
        threshold_file,
        index=False,
    )

    print("\nSaved:")
    print(summary_file)
    print(threshold_file)

    print("\n")
    print("=" * 70)
    print("STEP 10 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()