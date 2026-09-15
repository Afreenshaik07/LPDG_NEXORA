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

REVIEW_FILE = (
    DATA
    / "engineer_review_2026-02.xlsx"
)


# ============================================================
# HELPER
# ============================================================

def normalize_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
        .str.strip()
    )


# ============================================================
# LOAD
# ============================================================

def load_data():

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    review = pd.read_excel(
        REVIEW_FILE
    )

    features["gateway_norm"] = normalize_id(
        features["gateway_norm"]
    )

    review["gateway_norm"] = normalize_id(
        review["gateway_id"]
    )

    review["reviewed_on"] = pd.to_datetime(
        review["reviewed_on"],
        errors="coerce",
    )

    return features, review


# ============================================================
# GET PRE-REVIEW WEEK
# ============================================================

def match_review(features, review):

    rows = []

    for _, person in review.iterrows():

        gateway = person["gateway_norm"]
        review_date = person["reviewed_on"]

        # Monday of the review week
        review_week = (
            review_date
            - pd.to_timedelta(
                review_date.weekday(),
                unit="D",
            )
        ).normalize()

        history = features[
            (features["gateway_norm"] == gateway)
            & (features["week_start"] < review_week)
        ].sort_values("week_start")

        if history.empty:
            continue

        # Last complete week before review
        previous = history.iloc[-1]

        row = previous.to_dict()

        row["Kategorie"] = person["Kategorie"]
        row["reviewed_on"] = review_date
        row["Bemerkung"] = person["Bemerkung"]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EDA STEP 6 — ENGINEER REVIEW VALIDATION")
    print("=" * 70)

    features, review = load_data()

    matched = match_review(
        features,
        review,
    )

    print(
        "\nMatched engineer reviews:",
        len(matched),
    )

    print("\nReview outcomes:")
    print(
        matched["Kategorie"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Features to compare
    # --------------------------------------------------------

    candidate_features = [
        "success_rate",
        "success_change",
        "offline_duration_sec",
        "offline_change",
        "disconnection_cnt",
        "disconnection_change",
        "reboot_cnt",
        "reboot_change",
        "reboot_duration_sec",
        "low_success_streak",
        "n_meters_installed",
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    rows = []

    for feature in candidate_features:

        normal = matched.loc[
            matched["Kategorie"] == "Normal",
            feature,
        ].dropna()

        bad = matched.loc[
            matched["Kategorie"] == "Schlecht",
            feature,
        ].dropna()

        if normal.empty or bad.empty:
            continue

        normal_mean = normal.mean()
        bad_mean = bad.mean()

        combined = pd.concat(
            [
                normal,
                bad,
            ]
        )

        std = combined.std()

        if std == 0 or pd.isna(std):
            effect = 0.0
        else:
            effect = (
                bad_mean - normal_mean
            ) / std

        rows.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "schlecht_mean": bad_mean,
                "difference": (
                    bad_mean - normal_mean
                ),
                "standardized_difference": effect,
                "abs_effect": abs(effect),
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "abs_effect",
        ascending=False,
    )

    print("\n")
    print("=" * 70)
    print("ENGINEER REVIEW FEATURE COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
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

    output_file = (
        output_dir
        / "engineer_review_comparison.csv"
    )

    comparison.to_csv(
        output_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)

    print("\n")
    print("=" * 70)
    print("EDA STEP 6 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()