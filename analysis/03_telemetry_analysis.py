from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TELEMETRY = DATA / "telemetry"


# ============================================================
# TELEMETRY COLUMNS
# ============================================================

TELEMETRY_COLUMNS = [
    "gateway_id",
    "ts_utc",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "rssi_bad",
    "rscp_rsrp_bad",
    "ecio_rsrq_bad",
    "network_unknown",
]


# ============================================================
# LOAD TELEMETRY
# ============================================================

def load_telemetry():

    frames = []

    files = sorted(
        TELEMETRY.glob("month=*/part-*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No telemetry parquet files found in {TELEMETRY}"
        )

    print(f"Found {len(files)} telemetry files.")

    for file in files:

        print(f"Reading: {file.parent.name}")

        frame = pd.read_parquet(
            file,
            columns=TELEMETRY_COLUMNS,
        )

        frames.append(frame)

    telemetry = pd.concat(
        frames,
        ignore_index=True,
    )

    telemetry["ts"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True,
        errors="coerce",
    )

    telemetry["gateway_norm"] = (
        telemetry["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    return telemetry


# ============================================================
# CREATE WEEK
# ============================================================

def add_week(telemetry):

    telemetry["week_start"] = (
        telemetry["ts"]
        .dt.tz_convert(None)
        .dt.to_period("W-MON")
        .apply(lambda x: x.start_time)
    )

    return telemetry


# ============================================================
# WEEKLY AGGREGATION
# ============================================================

def aggregate_weekly(telemetry):

    metrics = [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
        "rssi_bad",
        "rscp_rsrp_bad",
        "ecio_rsrq_bad",
        "network_unknown",
    ]

    weekly = (
        telemetry
        .groupby(
            [
                "gateway_norm",
                "week_start",
            ]
        )[metrics]
        .sum()
        .reset_index()
    )

    return weekly


# ============================================================
# NORMALIZE VISITS
# ============================================================

def load_visits():

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="cp1252",
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce",
    )

    visits["gateway_norm"] = (
        visits["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    visits["visit_week"] = (
        visits["requested_on"]
        - pd.to_timedelta(
            visits["requested_on"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    return visits


# ============================================================
# PRE-VISIT TELEMETRY
# ============================================================

def build_previsit_dataset(
    weekly,
    visits,
):

    selected = visits[
        visits["outcome"].isin(
            [
                "Fehler behoben",
                "Kein Fehler gefunden",
            ]
        )
    ].copy()

    results = []

    for _, visit in selected.iterrows():

        gateway = visit["gateway_norm"]
        visit_week = visit["visit_week"]

        history = weekly[
            (weekly["gateway_norm"] == gateway)
            & (weekly["week_start"] < visit_week)
        ].sort_values("week_start")

        if len(history) < 4:
            continue

        last_4 = history.tail(4)

        row = {
            "gateway_norm": gateway,
            "requested_on": visit["requested_on"],
            "outcome": visit["outcome"],
        }

        metrics = [
            "offline_duration_sec",
            "disconnection_cnt",
            "reboot_cnt",
            "reboot_duration_sec",
            "rssi_bad",
            "rscp_rsrp_bad",
            "ecio_rsrq_bad",
            "network_unknown",
        ]

        for metric in metrics:

            row[f"{metric}_mean4w"] = (
                last_4[metric].mean()
            )

            row[f"{metric}_last"] = (
                last_4.iloc[-1][metric]
            )

            row[f"{metric}_max4w"] = (
                last_4[metric].max()
            )

            row[f"{metric}_change"] = (
                last_4.iloc[-1][metric]
                - last_4.iloc[0][metric]
            )

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("EDA STEP 3 — TELEMETRY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    telemetry = load_telemetry()

    print("\nTotal telemetry rows:", len(telemetry))

    print(
        "Unique gateways:",
        telemetry["gateway_norm"].nunique(),
    )

    print(
        "Telemetry time range:",
        telemetry["ts"].min(),
        "to",
        telemetry["ts"].max(),
    )

    # --------------------------------------------------------
    # Weekly aggregation
    # --------------------------------------------------------

    telemetry = add_week(telemetry)

    weekly = aggregate_weekly(telemetry)

    print(
        "\nWeekly gateway records:",
        len(weekly),
    )

    # --------------------------------------------------------
    # Load historical visits
    # --------------------------------------------------------

    visits = load_visits()

    # --------------------------------------------------------
    # Create pre-visit dataset
    # --------------------------------------------------------

    results = build_previsit_dataset(
        weekly,
        visits,
    )

    print(
        "Matched visits with 4 weeks telemetry:",
        len(results),
    )

    if results.empty:
        print("No matching visits found.")
        return

    # --------------------------------------------------------
    # Compare outcomes
    # --------------------------------------------------------

    metrics = [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
        "rssi_bad",
        "rscp_rsrp_bad",
        "ecio_rsrq_bad",
        "network_unknown",
    ]

    print("\n")
    print("=" * 70)
    print("AVERAGE PRE-VISIT TELEMETRY")
    print("=" * 70)

    for metric in metrics:

        mean4 = (
            results
            .groupby("outcome")
            [f"{metric}_mean4w"]
            .mean()
        )

        last = (
            results
            .groupby("outcome")
            [f"{metric}_last"]
            .mean()
        )

        change = (
            results
            .groupby("outcome")
            [f"{metric}_change"]
            .mean()
        )

        print(f"\n{metric}")

        print("  4-week average:")
        print(mean4)

        print("  Last week:")
        print(last)

        print("  Change:")
        print(change)

    # --------------------------------------------------------
    # Compact comparison table
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("COMPACT COMPARISON")
    print("=" * 70)

    summary_rows = []

    for metric in metrics:

        fixed_mean = results.loc[
            results["outcome"] == "Fehler behoben",
            f"{metric}_mean4w",
        ].mean()

        no_error_mean = results.loc[
            results["outcome"] == "Kein Fehler gefunden",
            f"{metric}_mean4w",
        ].mean()

        summary_rows.append(
            {
                "metric": metric,
                "fixed_problem_mean": fixed_mean,
                "no_problem_mean": no_error_mean,
                "difference": fixed_mean - no_error_mean,
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(
        summary
        .sort_values(
            "difference",
            ascending=False,
        )
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = ROOT / "analysis" / "outputs"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_dir / "previsit_telemetry.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "telemetry_comparison.csv",
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES CREATED")
    print("=" * 70)

    print(
        output_dir / "previsit_telemetry.csv"
    )

    print(
        output_dir / "telemetry_comparison.csv"
    )

    print("\n")
    print("=" * 70)
    print("EDA STEP 3 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()