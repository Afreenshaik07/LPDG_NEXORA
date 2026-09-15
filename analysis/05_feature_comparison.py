from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATURE_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


# ============================================================
# LOAD DATA
# ============================================================

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

    return features, visits


# ============================================================
# PREPARE VISITS
# ============================================================

def prepare_visits(visits):

    visits = visits[
        visits["outcome"].isin(
            [
                "Fehler behoben",
                "Kein Fehler gefunden",
            ]
        )
    ].copy()

    # Monday of the requested week
    visits["visit_week"] = (
        visits["requested_on"]
        - pd.to_timedelta(
            visits["requested_on"].dt.weekday,
            unit="D",
        )
    ).dt.normalize()

    return visits


# ============================================================
# MATCH VISITS TO PREVIOUS COMPLETE WEEK
# ============================================================

def match_previous_week(features, visits):

    rows = []

    for _, visit in visits.iterrows():

        gateway = visit["gateway_norm"]
        visit_week = visit["visit_week"]

        history = features[
            (features["gateway_norm"] == gateway)
            & (features["week_start"] < visit_week)
        ].sort_values("week_start")

        if history.empty:
            continue

        # Most recent complete week before the visit week
        previous = history.iloc[-1]

        row = previous.to_dict()

        row["outcome"] = visit["outcome"]
        row["requested_on"] = visit["requested_on"]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "success_rate",
    "success_change",
    "offline_duration_sec",
    "offline_change",
    "disconnection_cnt",
    "disconnection_change",
    "reboot_cnt",
    "reboot_change",
    "reboot_duration_sec",
    "n_meters_installed",
    "low_success_streak",
    "technical_streak",
]


# ============================================================
# ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("EDA STEP 5 — FEATURE COMPARISON")
    print("=" * 70)

    features, visits = load_data()

    visits = prepare_visits(visits)

    matched = match_previous_week(
        features,
        visits,
    )

    print(
        "\nMatched historical visits:",
        len(matched),
    )

    if matched.empty:
        print("No visits matched.")
        return

    # --------------------------------------------------------
    # Group counts
    # --------------------------------------------------------

    print("\nOutcome counts:")

    print(
        matched["outcome"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Compare features
    # --------------------------------------------------------

    rows = []

    for feature in FEATURES:

        fixed = matched.loc[
            matched["outcome"] == "Fehler behoben",
            feature,
        ].dropna()

        no_error = matched.loc[
            matched["outcome"] == "Kein Fehler gefunden",
            feature,
        ].dropna()

        if fixed.empty or no_error.empty:
            continue

        fixed_mean = fixed.mean()
        no_error_mean = no_error.mean()

        fixed_median = fixed.median()
        no_error_median = no_error.median()

        pooled = pd.concat(
            [
                fixed,
                no_error,
            ]
        )

        overall_std = pooled.std()

        if overall_std == 0 or pd.isna(overall_std):
            effect = 0.0
        else:
            effect = (
                fixed_mean
                - no_error_mean
            ) / overall_std

        rows.append(
            {
                "feature": feature,
                "fixed_mean": fixed_mean,
                "no_error_mean": no_error_mean,
                "mean_difference": (
                    fixed_mean
                    - no_error_mean
                ),
                "fixed_median": fixed_median,
                "no_error_median": no_error_median,
                "standardized_difference": effect,
            }
        )

    comparison = pd.DataFrame(rows)

    comparison["abs_effect"] = (
        comparison["standardized_difference"]
        .abs()
    )

    comparison = comparison.sort_values(
        "abs_effect",
        ascending=False,
    )

    print("\n")
    print("=" * 70)
    print("FEATURE COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Correlation-style direction
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)

    print(
        """
Larger positive standardized difference:
    Feature tends to be higher before genuine repairs.

Negative standardized difference:
    Feature tends to be lower before genuine repairs.

Large absolute difference:
    Potentially useful feature, but not automatically causal.

Use this table to guide feature selection.
Do not choose weights yet.
"""
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
        / "feature_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("EDA STEP 5 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()