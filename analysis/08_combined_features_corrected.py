from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TELEMETRY = DATA / "telemetry"


TELEMETRY_COLUMNS = [
    "gateway_id",
    "ts_utc",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
]


def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


def load_meter():

    meter = pd.read_csv(
        DATA / "meter_read_success.csv",
        encoding="cp1252",
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce",
    )

    meter["gateway_norm"] = normalize_id(
        meter["gateway_id"]
    )

    meter["success_rate"] = (
        meter["meters_read"]
        / meter["meters_expected"]
    )

    return meter


def load_master():

    master = pd.read_csv(
        DATA / "gateway_master.csv",
        encoding="cp1252",
    )

    master["gateway_norm"] = normalize_id(
        master["gateway_id"]
    )

    return master[
        [
            "gateway_norm",
            "n_meters_installed",
            "site_type",
            "region",
            "hw_model",
        ]
    ]


def load_telemetry_weekly():

    frames = []

    files = sorted(
        TELEMETRY.glob("month=*/part-*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No telemetry files found in {TELEMETRY}"
        )

    for file in files:

        print(f"Reading {file.parent.name}...")

        frame = pd.read_parquet(
            file,
            columns=TELEMETRY_COLUMNS,
        )

        frame["ts"] = pd.to_datetime(
            frame["ts_utc"],
            utc=True,
            errors="coerce",
        )

        frame["gateway_norm"] = normalize_id(
            frame["gateway_id"]
        )

        frames.append(frame)

    telemetry = pd.concat(
        frames,
        ignore_index=True,
    )

    ts = telemetry["ts"].dt.tz_convert(None)

    telemetry["week_start"] = (
        ts
        - pd.to_timedelta(
            ts.dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    # IMPORTANT:
    # offline_duration_sec is treated as a counter/state.
    # We use the final hourly reading of the week,
    # rather than summing repeated counter readings.
    #
    # Counts/durations that represent hourly activity are summed.

    weekly = (
        telemetry
        .sort_values("ts")
        .groupby(
            [
                "gateway_norm",
                "week_start",
            ]
        )
        .agg(
            offline_duration_sec=(
                "offline_duration_sec",
                "last",
            ),
            disconnection_cnt=(
                "disconnection_cnt",
                "sum",
            ),
            reboot_cnt=(
                "reboot_cnt",
                "sum",
            ),
            reboot_duration_sec=(
                "reboot_duration_sec",
                "sum",
            ),
        )
        .reset_index()
    )

    return weekly


def add_temporal_features(df):

    df = df.sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    ).copy()

    grouped = df.groupby(
        "gateway_norm",
        group_keys=False,
    )

    df["previous_success_rate"] = (
        grouped["success_rate"].shift(1)
    )

    df["previous_offline"] = (
        grouped["offline_duration_sec"].shift(1)
    )

    df["previous_disconnections"] = (
        grouped["disconnection_cnt"].shift(1)
    )

    df["previous_reboots"] = (
        grouped["reboot_cnt"].shift(1)
    )

    df["success_change"] = (
        df["success_rate"]
        - df["previous_success_rate"]
    )

    df["offline_change"] = (
        df["offline_duration_sec"]
        - df["previous_offline"]
    )

    df["disconnection_change"] = (
        df["disconnection_cnt"]
        - df["previous_disconnections"]
    )

    df["reboot_change"] = (
        df["reboot_cnt"]
        - df["previous_reboots"]
    )

    return df


def add_persistence_features(df):

    df = df.sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    ).copy()

    low_success = (
        df["success_rate"] < 0.80
    )

    success_group = (
        (~low_success)
        .groupby(df["gateway_norm"])
        .cumsum()
    )

    df["low_success_streak"] = (
        low_success.astype(int)
        .groupby(
            [
                df["gateway_norm"],
                success_group,
            ]
        )
        .cumsum()
    )

    return df


def main():

    print("=" * 70)
    print("STEP 8 — CORRECTED COMBINED FEATURES")
    print("=" * 70)

    print("\nLoading meter data...")
    meter = load_meter()

    print("Loading gateway master...")
    master = load_master()

    print("Loading telemetry...")
    telemetry = load_telemetry_weekly()

    print("\nCombining...")

    combined = meter.merge(
        telemetry,
        on=["gateway_norm", "week_start"],
        how="left",
    )

    combined = combined.merge(
        master,
        on="gateway_norm",
        how="left",
    )

    combined = add_temporal_features(
        combined
    )

    combined = add_persistence_features(
        combined
    )

    print("\nShape:", combined.shape)

    print("\nKey feature statistics:")

    print(
        combined[
            [
                "success_rate",
                "offline_duration_sec",
                "disconnection_cnt",
                "reboot_cnt",
                "reboot_duration_sec",
                "n_meters_installed",
                "low_success_streak",
            ]
        ].describe()
    )

    print("\nMaximum offline_duration_sec:")

    print(
        combined["offline_duration_sec"].max()
    )

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
        / "combined_gateway_week_corrected.csv"
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("STEP 8 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()