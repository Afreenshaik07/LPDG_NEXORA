from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week.csv"
)


def load_data():

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    return df


def percentile_score(series, higher_is_worse=True):

    score = series.rank(
        pct=True,
        method="average",
    )

    if higher_is_worse:
        return score

    return 1 - score


def build_risk_score(df):

    df = df.copy()

    # --------------------------------------------------------
    # Candidate risk components
    # --------------------------------------------------------

    # Lower meter success = worse
    df["risk_success"] = percentile_score(
        df["success_rate"],
        higher_is_worse=False,
    )

    # Higher offline duration = worse
    df["risk_offline"] = percentile_score(
        df["offline_duration_sec"],
        higher_is_worse=True,
    )

    # Higher disconnections = worse
    df["risk_disconnect"] = percentile_score(
        df["disconnection_cnt"],
        higher_is_worse=True,
    )

    # Higher reboots = worse
    df["risk_reboot"] = percentile_score(
        df["reboot_cnt"],
        higher_is_worse=True,
    )

    # --------------------------------------------------------
    # Equal-weight transparent score
    # --------------------------------------------------------

    df["risk_score"] = (
        df["risk_success"]
        + df["risk_offline"]
        + df["risk_disconnect"]
        + df["risk_reboot"]
    ) / 4

    return df


def main():

    print("=" * 70)
    print("EDA STEP 7 — FIRST RISK SCORE EXPERIMENT")
    print("=" * 70)

    df = load_data()

    print(
        "\nInput shape:",
        df.shape,
    )

    df = build_risk_score(df)

    print("\nRisk score summary:")
    print(
        df["risk_score"].describe()
    )

    # --------------------------------------------------------
    # Highest-risk gateway weeks
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP 20 HIGHEST-RISK GATEWAY WEEKS")
    print("=" * 70)

    columns = [
        "week_start",
        "gateway_norm",
        "success_rate",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "risk_score",
    ]

    print(
        df.sort_values(
            "risk_score",
            ascending=False,
        )[columns]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "analysis"
        / "outputs"
    )

    output_file = (
        output_dir
        / "risk_score_experiment.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("EDA STEP 7 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()