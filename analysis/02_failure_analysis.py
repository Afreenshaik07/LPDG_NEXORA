from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_data():
    meter = pd.read_csv(
        DATA / "meter_read_success.csv",
        encoding="cp1252",
    )

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="cp1252",
    )

    return meter, visits


def prepare_data(meter, visits):
    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce",
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce",
    )

    meter["success_rate"] = (
        meter["meters_read"] / meter["meters_expected"]
    )

    # Normalize gateway IDs to one format.
    meter["gateway_norm"] = (
        meter["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    visits["gateway_norm"] = (
        visits["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    return meter, visits


def calculate_previsit_metrics(meter, visits):

    results = []

    selected_visits = visits[
        visits["outcome"].isin(
            [
                "Fehler behoben",
                "Kein Fehler gefunden",
            ]
        )
    ].copy()

    for _, visit in selected_visits.iterrows():

        gateway_id = visit["gateway_norm"]
        visit_date = visit["requested_on"]

        gateway_data = meter[
            meter["gateway_norm"] == gateway_id
        ].copy()

        if gateway_data.empty:
            continue

        # Monday of the week in which the visit was requested.
        visit_week_start = (
            visit_date
            - pd.Timedelta(days=visit_date.weekday())
        ).normalize()

        # IMPORTANT:
        # Use only COMPLETE weeks strictly before the visit week.
        previous = gateway_data[
            gateway_data["week_start"] < visit_week_start
        ].sort_values("week_start")

        if previous.empty:
            continue

        previous_4 = previous.tail(4)

        results.append(
            {
                "gateway_id": gateway_id,
                "requested_on": visit_date,
                "outcome": visit["outcome"],
                "avg_success_4w": previous_4["success_rate"].mean(),
                "last_success": previous_4.iloc[-1]["success_rate"],
                "min_success_4w": previous_4["success_rate"].min(),
                "max_success_4w": previous_4["success_rate"].max(),
            }
        )

    return pd.DataFrame(results)


def main():

    print("=" * 70)
    print("EDA STEP 2 — HISTORICAL FAILURE ANALYSIS")
    print("=" * 70)

    meter, visits = load_data()

    meter, visits = prepare_data(
        meter,
        visits,
    )

    results = calculate_previsit_metrics(
        meter,
        visits,
    )

    print("\nMatched historical visits:", len(results))

    if results.empty:
        print("No historical visits could be matched.")
        return

    print("\n")
    print("=" * 70)
    print("PRE-VISIT METER-READ PERFORMANCE")
    print("=" * 70)

    summary = (
        results
        .groupby("outcome")[
            [
                "avg_success_4w",
                "last_success",
                "min_success_4w",
                "max_success_4w",
            ]
        ]
        .agg(["count", "mean", "median"])
    )

    print(summary)

    print("\n")
    print("=" * 70)
    print("AVERAGE PRE-VISIT PERFORMANCE")
    print("=" * 70)

    simple_summary = (
        results
        .groupby("outcome")[
            [
                "avg_success_4w",
                "last_success",
                "min_success_4w",
            ]
        ]
        .mean()
        .sort_values("avg_success_4w")
    )

    print(simple_summary)

    print("\n")
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:

        flagged = (
            results["avg_success_4w"] < threshold
        )

        rates = (
            results.assign(below_threshold=flagged)
            .groupby("outcome")["below_threshold"]
            .mean()
            .mul(100)
        )

        print(f"\nThreshold < {threshold:.0%}")

        for outcome, rate in rates.items():
            print(f"  {outcome}: {rate:.1f}%")

    results["change_4w"] = (
        results["last_success"]
        - results["avg_success_4w"]
    )

    print("\n")
    print("=" * 70)
    print("LARGEST NEGATIVE CHANGES")
    print("=" * 70)

    print(
        results
        .sort_values("change_4w")
        [
            [
                "gateway_id",
                "requested_on",
                "outcome",
                "avg_success_4w",
                "last_success",
                "change_4w",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("EDA STEP 2 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()