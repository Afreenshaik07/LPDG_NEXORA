from pathlib import Path

import pandas as pd

# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the three main datasets needed for the first EDA step.

    cp1252 is used because the challenge data contains German
    characters such as ß, ä, ö, and ü.
    """

    gateway = pd.read_csv(
        DATA / "gateway_master.csv",
        encoding="cp1252"
    )

    meter = pd.read_csv(
        DATA / "meter_read_success.csv",
        encoding="cp1252"
    )

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="cp1252"
    )

    return gateway, meter, visits


# ============================================================
# MAIN EDA
# ============================================================

def main():

    gateway, meter, visits = load_data()

    # --------------------------------------------------------
    # 1. GATEWAY MASTER
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GATEWAY MASTER")
    print("=" * 60)

    print("Shape:", gateway.shape)

    print("\nColumns:")
    print(gateway.columns.tolist())

    print("\nMissing values:")
    print(gateway.isna().sum())

    print("\nFirst 5 rows:")
    print(gateway.head())

    # --------------------------------------------------------
    # 2. METER READ SUCCESS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("METER READ SUCCESS")
    print("=" * 60)

    print("Shape:", meter.shape)

    print("\nColumns:")
    print(meter.columns.tolist())

    print("\nMissing values:")
    print(meter.isna().sum())

    print("\nFirst 5 rows:")
    print(meter.head())

    # --------------------------------------------------------
    # 3. CALCULATE METER SUCCESS RATE
    # --------------------------------------------------------

    meter["success_rate"] = (
        meter["meters_read"] / meter["meters_expected"]
    )

    print("\n" + "=" * 60)
    print("METER SUCCESS RATE")
    print("=" * 60)

    print(meter["success_rate"].describe())

    # Check for invalid values
    invalid_rate = (
        (meter["success_rate"] < 0)
        | (meter["success_rate"] > 1)
    ).sum()

    print("\nInvalid success-rate rows:", invalid_rate)

    # --------------------------------------------------------
    # 4. FIELD VISITS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FIELD VISITS")
    print("=" * 60)

    print("Shape:", visits.shape)

    print("\nColumns:")
    print(visits.columns.tolist())

    print("\nMissing values:")
    print(visits.isna().sum())

    print("\nFirst 5 rows:")
    print(visits.head())

    # --------------------------------------------------------
    # 5. VISIT OUTCOMES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VISIT OUTCOMES")
    print("=" * 60)

    print(
        visits["outcome"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # 6. UNIQUE GATEWAYS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("UNIQUE GATEWAY COUNTS")
    print("=" * 60)

    print(
        "Gateway master:",
        gateway["gateway_id"].nunique()
    )

    print(
        "Meter-read dataset:",
        meter["gateway_id"].nunique()
    )

    print(
        "Field-visit dataset:",
        visits["gateway_id"].nunique()
    )

    # --------------------------------------------------------
    # 7. DATE RANGES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATE RANGES")
    print("=" * 60)

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce"
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce"
    )

    visits["visited_on"] = pd.to_datetime(
        visits["visited_on"],
        errors="coerce"
    )

    print(
        "Meter-read:",
        meter["week_start"].min(),
        "to",
        meter["week_start"].max()
    )

    print(
        "Visit requested:",
        visits["requested_on"].min(),
        "to",
        visits["requested_on"].max()
    )

    print(
        "Visit completed:",
        visits["visited_on"].min(),
        "to",
        visits["visited_on"].max()
    )

    # --------------------------------------------------------
    # 8. BASIC GATEWAY IMPACT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GATEWAY METER IMPACT")
    print("=" * 60)

    print(
        gateway["n_meters_installed"].describe()
    )

    # --------------------------------------------------------
    # 9. TOP 10 WORST GATEWAY-WEEKS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("10 WORST GATEWAY-WEEKS BY METER SUCCESS")
    print("=" * 60)

    worst = (
        meter[
            [
                "week_start",
                "gateway_id",
                "meters_expected",
                "meters_read",
                "success_rate",
            ]
        ]
        .sort_values("success_rate")
        .head(10)
    )

    print(worst.to_string(index=False))

    # --------------------------------------------------------
    # 10. BASIC FIELD-VISIT PARTS REPLACED ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PARTS REPLACED")
    print("=" * 60)

    parts_available = visits["parts_replaced"].notna().sum()

    print(
        "Visits with a recorded part replacement:",
        parts_available
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EDA STEP 1 COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()