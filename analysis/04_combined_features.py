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
]


# ============================================================
# NORMALIZE GATEWAY ID
# ============================================================

def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


# ============================================================
# LOAD METER DATA
# ============================================================

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


# ============================================================
# LOAD GATEWAY MASTER
# ============================================================

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


# ============================================================
# LOAD TELEMETRY WEEKLY
# ============================================================

def load_telemetry_weekly():

    frames = []

    files = sorted(
        TELEMETRY.glob(
            "month=*/part-*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {TELEMETRY}"
        )

    for file in files:

        print(
            f"Reading {file.parent.name}..."
        )

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

    # Convert each timestamp to the Monday of its week.
    # Use UTC consistently with the baseline.
    ts = telemetry["ts"].dt.tz_convert(None)

    telemetry["week_start"] = (
        ts
        - pd.to_timedelta(
            ts.dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    weekly = (
        telemetry
        .groupby(
            [
                "gateway_norm",
                "week_start",
            ],
            as_index=False,
        )[
            [
                "offline_duration_sec",
                "disconnection_cnt",
                "reboot_cnt",
                "reboot_duration_sec",
            ]
        ]
        .sum()
    )

    return weekly


# ============================================================
# CREATE FEATURES
# ============================================================

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

    # Previous week's values
    df["previous_success_rate"] = (
        grouped["success_rate"]
        .shift(1)
    )

    df["previous_offline"] = (
        grouped["offline_duration_sec"]
        .shift(1)
    )

    df["previous_disconnections"] = (
        grouped["disconnection_cnt"]
        .shift(1)
    )

    df["previous_reboots"] = (
        grouped["reboot_cnt"]
        .shift(1)
    )

    # Changes
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


# ============================================================
# CREATE PERSISTENCE FEATURES
# ============================================================

def add_persistence_features(df):

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

    # Count consecutive weeks where success is below 80%.
    low_success = (
        df["success_rate"] < 0.80
    )

    groups = (
        (~low_success)
        .groupby(df["gateway_norm"])
        .cumsum()
    )

    df["low_success_streak"] = (
        low_success
        .astype(int)
        .groupby(
            [
                df["gateway_norm"],
                groups,
            ]
        )
        .cumsum()
    )

    # Count weeks with elevated technical activity.
    high_technical = (
        (df["offline_duration_sec"] > 400000)
        | (df["disconnection_cnt"] > 200)
        | (df["reboot_cnt"] > 15)
    )

    tech_groups = (
        (~high_technical)
        .groupby(df["gateway_norm"])
        .cumsum()
    )

    df["technical_streak"] = (
        high_technical
        .astype(int)
        .groupby(
            [
                df["gateway_norm"],
                tech_groups,
            ]
        )
        .cumsum()
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EDA STEP 4 — COMBINED FEATURE TABLE")
    print("=" * 70)

    print("\nLoading meter data...")
    meter = load_meter()

    print("Loading gateway master...")
    master = load_master()

    print("Loading telemetry...")
    telemetry = load_telemetry_weekly()

    print("\nCombining datasets...")

    combined = meter.merge(
        telemetry,
        on=[
            "gateway_norm",
            "week_start",
        ],
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

    # --------------------------------------------------------
    # Keep relevant weeks
    # --------------------------------------------------------

    combined = combined.sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )

    # --------------------------------------------------------
    # Show basic statistics
    # --------------------------------------------------------

    print("\nCombined table shape:")
    print(combined.shape)

    print("\nMissing values:")
    print(
        combined[
            [
                "success_rate",
                "offline_duration_sec",
                "disconnection_cnt",
                "reboot_cnt",
                "n_meters_installed",
            ]
        ].isna().sum()
    )

    print("\nFeature summary:")
    print(
        combined[
            [
                "success_rate",
                "success_change",
                "offline_duration_sec",
                "offline_change",
                "disconnection_cnt",
                "disconnection_change",
                "reboot_cnt",
                "reboot_change",
                "n_meters_installed",
                "low_success_streak",
                "technical_streak",
            ]
        ].describe()
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
        / "combined_gateway_week.csv"
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("EDA STEP 4 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()