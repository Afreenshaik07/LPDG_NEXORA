from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FEATURE_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week_corrected.csv"
)


def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


def load_data():
    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="cp1252",
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce",
    )

    features["gateway_norm"] = normalize_id(
        features["gateway_norm"]
    )

    visits["gateway_norm"] = normalize_id(
        visits["gateway_id"]
    )

    visits["visit_week"] = (
        visits["requested_on"]
        - pd.to_timedelta(
            visits["requested_on"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    return features, visits


def percentile(series):
    return series.rank(
        pct=True,
        method="average",
    )


def create_scores(df):

    df = df.copy()

    # Lower success = worse
    df["risk_success"] = (
        1 - percentile(df["success_rate"])
    )

    # Larger negative success change = worse
    df["risk_success_change"] = (
        1 - percentile(
            df["success_change"].fillna(0)
        )
    )

    # Larger offline time = worse
    df["risk_offline"] = percentile(
        df["offline_duration_sec"].fillna(0)
    )

    # More disconnections = worse
    df["risk_disconnect"] = percentile(
        df["disconnection_cnt"].fillna(0)
    )

    # More reboots = worse
    df["risk_reboot"] = percentile(
        df["reboot_cnt"].fillna(0)
    )

    # More reboot duration = worse
    df["risk_reboot_duration"] = percentile(
        df["reboot_duration_sec"].fillna(0)
    )

    # Longer low-success streak = worse
    df["risk_persistence"] = percentile(
        df["low_success_streak"].fillna(0)
    )

    # --------------------------------------------------------
    # Candidate scores
    # --------------------------------------------------------

    df["business_score"] = (
        df["risk_success"]
        + df["risk_success_change"]
    ) / 2

    df["technical_score"] = (
        df["risk_offline"]
        + df["risk_disconnect"]
        + df["risk_reboot"]
        + df["risk_reboot_duration"]
    ) / 4

    df["combined_score"] = (
        0.5 * df["business_score"]
        + 0.4 * df["technical_score"]
        + 0.1 * df["risk_persistence"]
    )

    return df


def build_previsit_records(features, visits):

    selected = visits[
        visits["outcome"].isin(
            [
                "Fehler behoben",
                "Kein Fehler gefunden",
            ]
        )
    ].copy()

    rows = []

    for _, visit in selected.iterrows():

        gateway = visit["gateway_norm"]
        visit_week = visit["visit_week"]

        history = features[
            (features["gateway_norm"] == gateway)
            & (features["week_start"] < visit_week)
        ].sort_values("week_start")

        if history.empty:
            continue

        # Most recent COMPLETE week before the visit week
        previous = history.iloc[-1].copy()

        row = previous.to_dict()

        row["outcome"] = visit["outcome"]
        row["requested_on"] = visit["requested_on"]

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_score(
    df,
    score_column,
    threshold,
):

    fixed = df[
        df["outcome"] == "Fehler behoben"
    ][score_column]

    no_error = df[
        df["outcome"] == "Kein Fehler gefunden"
    ][score_column]

    fixed_rate = (
        (fixed >= threshold).mean()
        if len(fixed)
        else np.nan
    )

    false_alarm_rate = (
        (no_error >= threshold).mean()
        if len(no_error)
        else np.nan
    )

    return fixed_rate, false_alarm_rate


def main():

    print("=" * 70)
    print("STEP 9 — HISTORICAL BACKTEST")
    print("=" * 70)

    features, visits = load_data()

    print(
        "\nFeature rows:",
        len(features),
    )

    historical = build_previsit_records(
        features,
        visits,
    )

    print(
        "Matched historical visits:",
        len(historical),
    )

    historical = create_scores(
        historical
    )

    print("\n")
    print("=" * 70)
    print("SCORE COMPARISON")
    print("=" * 70)

    score_columns = [
        "business_score",
        "technical_score",
        "combined_score",
    ]

    for score in score_columns:

        print(f"\n{score}")

        print(
            historical
            .groupby("outcome")[score]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                ]
            )
        )

    # --------------------------------------------------------
    # Threshold test
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("COMBINED SCORE THRESHOLDS")
    print("=" * 70)

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        fixed_rate, false_alarm_rate = (
            evaluate_score(
                historical,
                "combined_score",
                threshold,
            )
        )

        print(
            f"\nThreshold >= {threshold:.2f}"
        )

        print(
            f"Problem fixed captured: "
            f"{fixed_rate:.1%}"
        )

        print(
            f"No-problem visits flagged: "
            f"{false_alarm_rate:.1%}"
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

    output_file = (
        output_dir
        / "historical_backtest.csv"
    )

    historical.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("STEP 9 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()