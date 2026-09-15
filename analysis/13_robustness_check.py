from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week_corrected.csv"
)

BACKTEST_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "score_design_backtest.csv"
)

TOP_K = 15


def main():

    print("=" * 70)
    print("STEP 13 — ROBUSTNESS CHECK")
    print("=" * 70)

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    backtest = pd.read_csv(
        BACKTEST_FILE,
        parse_dates=["decision_week", "previous_week"],
    )

    print(
        "\nFeature rows:",
        len(features),
    )

    # --------------------------------------------------------
    # 1. Missing feature check
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MISSING VALUES IN CORE FEATURES")
    print("=" * 70)

    core_features = [
        "success_rate",
        "success_change",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
        "low_success_streak",
        "n_meters_installed",
    ]

    print(
        features[core_features]
        .isna()
        .sum()
        .to_string()
    )

    # --------------------------------------------------------
    # 2. Extreme-value check
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXTREME VALUES")
    print("=" * 70)

    for feature in [
        "success_rate",
        "success_change",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]:

        series = features[feature].dropna()

        print(
            f"{feature}: "
            f"min={series.min():.3f}, "
            f"median={series.median():.3f}, "
            f"max={series.max():.3f}"
        )

    # --------------------------------------------------------
    # 3. Early vs late backtest performance
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EARLY VS LATE PERFORMANCE")
    print("=" * 70)

    d = backtest[
        backtest["model"]
        == "D_technical_confirm"
    ].copy()

    unique_weeks = sorted(
        d["decision_week"].unique()
    )

    midpoint = len(unique_weeks) // 2

    early_weeks = set(
        unique_weeks[:midpoint]
    )

    late_weeks = set(
        unique_weeks[midpoint:]
    )

    early = d[
        d["decision_week"].isin(
            early_weeks
        )
    ]

    late = d[
        d["decision_week"].isin(
            late_weeks
        )
    ]

    print(
        "\nEarly period:"
    )

    print(
        "Mean capture:",
        f"{early['capture_rate'].mean():.2%}"
    )

    print(
        "Median capture:",
        f"{early['capture_rate'].median():.2%}"
    )

    print(
        "\nLate period:"
    )

    print(
        "Mean capture:",
        f"{late['capture_rate'].mean():.2%}"
    )

    print(
        "Median capture:",
        f"{late['capture_rate'].median():.2%}"
    )

    # --------------------------------------------------------
    # 4. Weeks with weakest performance
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEAKEST WEEKS FOR DESIGN D")
    print("=" * 70)

    print(
        d.sort_values(
            "capture_rate"
        )
        [
            [
                "decision_week",
                "historical_repairs",
                "captured_top15",
                "capture_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Strongest weeks
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STRONGEST WEEKS FOR DESIGN D")
    print("=" * 70)

    print(
        d.sort_values(
            "capture_rate",
            ascending=False,
        )
        [
            [
                "decision_week",
                "historical_repairs",
                "captured_top15",
                "capture_rate",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 6. Repeated high-risk gateways
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PERSISTENTLY LOW-SUCCESS GATEWAYS")
    print("=" * 70)

    persistent = (
        features[
            features["low_success_streak"] >= 3
        ]
        .groupby("gateway_norm")
        .agg(
            weeks_low_success=(
                "low_success_streak",
                "max",
            ),
            min_success=(
                "success_rate",
                "min",
            ),
            avg_success=(
                "success_rate",
                "mean",
            ),
        )
        .sort_values(
            "weeks_low_success",
            ascending=False,
        )
        .head(20)
    )

    print(
        persistent.to_string()
    )

    # --------------------------------------------------------
    # 7. Final observation
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        """
Design D is currently our leading candidate because it
combines business degradation, trend, persistence, and a
small amount of technical confirmation.

This step checks whether the result is stable and whether
there are obvious data-quality or extreme-value problems.

Do not tune the model based only on individual weeks.
"""
    )

    print("\n")
    print("=" * 70)
    print("STEP 13 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()