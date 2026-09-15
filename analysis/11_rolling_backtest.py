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
    / "combined_gateway_week_corrected.csv"
)


TOP_K = 15


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

def load_features():

    df = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    df["gateway_norm"] = normalize_id(
        df["gateway_norm"]
    )

    return df


def load_visits():

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="cp1252",
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        errors="coerce",
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

    return visits


# ============================================================
# PERCENTILE RANK
# ============================================================

def percentile_rank(series, higher_is_worse=True):

    result = series.rank(
        pct=True,
        method="average",
    )

    if higher_is_worse:
        return result

    return 1 - result


# ============================================================
# CREATE SCORES FOR ONE DECISION WEEK
# ============================================================

def score_week(history):

    df = history.copy()

    # --------------------------------------------------------
    # Business risk
    # --------------------------------------------------------

    # Lower success = worse
    df["risk_success"] = percentile_rank(
        df["success_rate"],
        higher_is_worse=False,
    )

    # More negative change = worse
    df["risk_success_change"] = percentile_rank(
        df["success_change"].fillna(0),
        higher_is_worse=False,
    )

    df["business_score"] = (
        df["risk_success"]
        + df["risk_success_change"]
    ) / 2

    # --------------------------------------------------------
    # Technical risk
    # --------------------------------------------------------

    df["risk_offline"] = percentile_rank(
        df["offline_duration_sec"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_disconnect"] = percentile_rank(
        df["disconnection_cnt"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_reboot"] = percentile_rank(
        df["reboot_cnt"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_reboot_duration"] = percentile_rank(
        df["reboot_duration_sec"].fillna(0),
        higher_is_worse=True,
    )

    df["technical_score"] = (
        df["risk_offline"]
        + df["risk_disconnect"]
        + df["risk_reboot"]
        + df["risk_reboot_duration"]
    ) / 4

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    df["risk_persistence"] = percentile_rank(
        df["low_success_streak"].fillna(0),
        higher_is_worse=True,
    )

    # --------------------------------------------------------
    # Four candidate scoring designs
    # --------------------------------------------------------

    # A: Business only
    df["score_business"] = (
        df["business_score"]
    )

    # B: Technical only
    df["score_technical"] = (
        df["technical_score"]
    )

    # C: Business + Technical
    df["score_balanced"] = (
        0.5 * df["business_score"]
        + 0.5 * df["technical_score"]
    )

    # D: Business + Technical + Persistence
    df["score_full"] = (
        0.45 * df["business_score"]
        + 0.45 * df["technical_score"]
        + 0.10 * df["risk_persistence"]
    )

    return df


# ============================================================
# HISTORICAL LABELS
# ============================================================

def create_repair_labels(visits):

    repairs = visits[
        visits["outcome"] == "Fehler behoben"
    ].copy()

    # One positive label per gateway-week
    repair_labels = (
        repairs[
            [
                "visit_week",
                "gateway_norm",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "visit_week": "decision_week"
            }
        )
    )

    return repair_labels


# ============================================================
# RUN ROLLING BACKTEST
# ============================================================

def run_backtest(features, visits):

    repair_labels = create_repair_labels(
        visits
    )

    # --------------------------------------------------------
    # Decision weeks
    #
    # We need a previous complete week.
    # --------------------------------------------------------

    available_weeks = sorted(
        features["week_start"]
        .dropna()
        .unique()
    )

    decision_weeks = available_weeks[1:]

    records = []

    for decision_week in decision_weeks:

        previous_week = (
            decision_week
            - pd.Timedelta(days=7)
        )

        history = features[
            features["week_start"]
            == previous_week
        ].copy()

        if history.empty:
            continue

        # Require core features
        required = [
            "gateway_norm",
            "success_rate",
            "offline_duration_sec",
            "disconnection_cnt",
            "reboot_cnt",
            "reboot_duration_sec",
            "low_success_streak",
        ]

        history = history.dropna(
            subset=required
        )

        if len(history) < TOP_K:
            continue

        scored = score_week(history)

        # ----------------------------------------------------
        # Positive historical label
        # ----------------------------------------------------

        week_repairs = repair_labels[
            repair_labels["decision_week"]
            == decision_week
        ]

        repair_set = set(
            week_repairs["gateway_norm"]
        )

        # ----------------------------------------------------
        # Evaluate each scoring design
        # ----------------------------------------------------

        score_columns = {
            "business": "score_business",
            "technical": "score_technical",
            "balanced": "score_balanced",
            "full": "score_full",
        }

        for model_name, score_column in score_columns.items():

            ranked = (
                scored
                .sort_values(
                    [
                        score_column,
                        "gateway_norm",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .reset_index(drop=True)
            )

            ranked["rank"] = (
                ranked.index + 1
            )

            top15 = ranked.head(
                TOP_K
            )

            top15_ids = set(
                top15["gateway_norm"]
            )

            # Genuine repair gateways found in top 15
            captured_repairs = len(
                top15_ids
                & repair_set
            )

            total_repairs = len(
                repair_set
            )

            if total_repairs > 0:
                repair_capture = (
                    captured_repairs
                    / total_repairs
                )
            else:
                repair_capture = None

            # Rank of each known repair gateway
            repair_ranks = ranked.loc[
                ranked["gateway_norm"]
                .isin(repair_set),
                "rank",
            ]

            mean_repair_rank = (
                repair_ranks.mean()
                if not repair_ranks.empty
                else None
            )

            median_repair_rank = (
                repair_ranks.median()
                if not repair_ranks.empty
                else None
            )

            records.append(
                {
                    "decision_week": decision_week,
                    "previous_week": previous_week,
                    "model": model_name,
                    "gateways_available": len(ranked),
                    "historical_repairs": total_repairs,
                    "repairs_captured_top15": (
                        captured_repairs
                    ),
                    "repair_capture": (
                        repair_capture
                    ),
                    "mean_repair_rank": (
                        mean_repair_rank
                    ),
                    "median_repair_rank": (
                        median_repair_rank
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================
# SUMMARIZE
# ============================================================

def summarize(backtest):

    summary = (
        backtest
        .groupby("model")
        .agg(
            weeks_tested=(
                "decision_week",
                "nunique",
            ),
            avg_historical_repairs=(
                "historical_repairs",
                "mean",
            ),
            avg_repairs_captured=(
                "repairs_captured_top15",
                "mean",
            ),
            mean_repair_capture=(
                "repair_capture",
                "mean",
            ),
            median_repair_capture=(
                "repair_capture",
                "median",
            ),
            mean_repair_rank=(
                "mean_repair_rank",
                "mean",
            ),
            median_repair_rank=(
                "median_repair_rank",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_repair_capture",
            ascending=False,
        )
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 11 — ROLLING HISTORICAL BACKTEST")
    print("=" * 70)

    print("\nLoading corrected feature table...")
    features = load_features()

    print("Loading historical visits...")
    visits = load_visits()

    print(
        "\nFeature rows:",
        len(features),
    )

    print(
        "Feature weeks:",
        features["week_start"].min(),
        "to",
        features["week_start"].max(),
    )

    print(
        "Historical visits:",
        len(visits),
    )

    print(
        "Historical genuine repair rows:",
        (visits["outcome"] == "Fehler behoben").sum(),
    )

    print("\nRunning rolling backtest...")

    backtest = run_backtest(
        features,
        visits,
    )

    if backtest.empty:
        raise SystemExit(
            "Backtest produced no results."
        )

    print(
        "\nBacktest rows:",
        len(backtest),
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize(
        backtest
    )

    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Week-by-week comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEEK-BY-WEEK REPAIR CAPTURE")
    print("=" * 70)

    pivot = (
        backtest
        .pivot_table(
            index="decision_week",
            columns="model",
            values="repair_capture",
        )
        .reset_index()
    )

    print(
        pivot.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save outputs
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

    backtest_file = (
        output_dir
        / "rolling_backtest_detail.csv"
    )

    summary_file = (
        output_dir
        / "rolling_backtest_summary.csv"
    )

    backtest.to_csv(
        backtest_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(backtest_file)
    print(summary_file)

    print("\n")
    print("=" * 70)
    print("STEP 11 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()