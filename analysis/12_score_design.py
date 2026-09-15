from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week_corrected.csv"
)

DATA = ROOT / "data"

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


def percentile_risk(series, higher_is_worse=True):
    """
    Convert a feature into a 0-1 percentile risk score.
    Higher score always means worse.
    """

    rank = series.rank(
        pct=True,
        method="average",
    )

    if higher_is_worse:
        return rank

    return 1 - rank


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():

    features = pd.read_csv(
        FEATURE_FILE,
        parse_dates=["week_start"],
    )

    features["gateway_norm"] = normalize_id(
        features["gateway_norm"]
    )

    return features


# ============================================================
# LOAD HISTORICAL VISITS
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
# CREATE HISTORICAL REPAIR LABELS
# ============================================================

def create_repair_labels(visits):

    repairs = visits[
        visits["outcome"] == "Fehler behoben"
    ].copy()

    labels = (
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

    return labels


# ============================================================
# SCORE ONE DECISION WEEK
# ============================================================

def score_week(history):

    df = history.copy()

    # --------------------------------------------------------
    # BUSINESS: current success
    # --------------------------------------------------------

    df["risk_success"] = percentile_risk(
        df["success_rate"],
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # BUSINESS: recent deterioration
    # Lower change = worse
    # --------------------------------------------------------

    df["risk_success_change"] = percentile_risk(
        df["success_change"].fillna(0),
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    df["risk_persistence"] = percentile_risk(
        df["low_success_streak"].fillna(0),
        higher_is_worse=True,
    )

    # --------------------------------------------------------
    # TECHNICAL CONFIRMATION
    # --------------------------------------------------------

    df["risk_offline"] = percentile_risk(
        df["offline_duration_sec"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_disconnect"] = percentile_risk(
        df["disconnection_cnt"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_reboot"] = percentile_risk(
        df["reboot_cnt"].fillna(0),
        higher_is_worse=True,
    )

    df["risk_reboot_duration"] = percentile_risk(
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
    # FOUR CANDIDATE DESIGNS
    # --------------------------------------------------------

    # A. Current business condition
    df["score_A_business_level"] = (
        df["risk_success"]
    )

    # B. Business level + deterioration
    df["score_B_business_trend"] = (
        0.50 * df["risk_success"]
        + 0.50 * df["risk_success_change"]
    )

    # C. Business + deterioration + persistence
    df["score_C_persistence"] = (
        0.45 * df["risk_success"]
        + 0.40 * df["risk_success_change"]
        + 0.15 * df["risk_persistence"]
    )

    # D. Business + deterioration + persistence
    #    + small technical confirmation
    df["score_D_technical_confirm"] = (
        0.45 * df["risk_success"]
        + 0.35 * df["risk_success_change"]
        + 0.10 * df["risk_persistence"]
        + 0.10 * df["technical_score"]
    )

    return df


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(features, repair_labels):

    # Only use weeks where we have a previous complete week.
    available_weeks = sorted(
        features["week_start"]
        .dropna()
        .unique()
    )

    decision_weeks = available_weeks[1:]

    models = {
        "A_business_level": "score_A_business_level",
        "B_business_trend": "score_B_business_trend",
        "C_persistence": "score_C_persistence",
        "D_technical_confirm": "score_D_technical_confirm",
    }

    rows = []

    for decision_week in decision_weeks:

        previous_week = (
            decision_week
            - pd.Timedelta(days=7)
        )

        history = features[
            features["week_start"] == previous_week
        ].copy()

        # Need at least 15 gateways
        if len(history) < TOP_K:
            continue

        scored = score_week(history)

        repairs = repair_labels[
            repair_labels["decision_week"]
            == decision_week
        ]

        repair_set = set(
            repairs["gateway_norm"]
        )

        for model_name, score_column in models.items():

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

            captured = len(
                top15_ids & repair_set
            )

            total_repairs = len(
                repair_set
            )

            if total_repairs > 0:
                capture_rate = (
                    captured
                    / total_repairs
                )
            else:
                capture_rate = None

            rows.append(
                {
                    "decision_week": decision_week,
                    "previous_week": previous_week,
                    "model": model_name,
                    "historical_repairs": total_repairs,
                    "captured_top15": captured,
                    "capture_rate": capture_rate,
                }
            )

    return pd.DataFrame(rows)


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
            avg_captured_top15=(
                "captured_top15",
                "mean",
            ),
            mean_capture_rate=(
                "capture_rate",
                "mean",
            ),
            median_capture_rate=(
                "capture_rate",
                "median",
            ),
        )
        .reset_index()
        .sort_values(
            "mean_capture_rate",
            ascending=False,
        )
    )

    return summary


# ============================================================
# WEEKLY STABILITY
# ============================================================

def calculate_stability(backtest):

    stability = (
        backtest
        .groupby("model")["capture_rate"]
        .agg(
            mean="mean",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    stability["coefficient_of_variation"] = (
        stability["std"]
        / stability["mean"]
    )

    return stability


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 12 — SCORE DESIGN COMPARISON")
    print("=" * 70)

    features = load_features()
    visits = load_visits()

    repairs = create_repair_labels(
        visits
    )

    print(
        "\nFeature rows:",
        len(features),
    )

    print(
        "Historical repair records:",
        len(repairs),
    )

    print("\nRunning score-design backtest...")

    backtest = run_backtest(
        features,
        repairs,
    )

    if backtest.empty:
        raise SystemExit(
            "No backtest results were generated."
        )

    print(
        "\nBacktest rows:",
        len(backtest),
    )

    # --------------------------------------------------------
    # Main comparison
    # --------------------------------------------------------

    summary = summarize(
        backtest
    )

    print("\n")
    print("=" * 70)
    print("SCORE DESIGN COMPARISON")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Stability
    # --------------------------------------------------------

    stability = calculate_stability(
        backtest
    )

    print("\n")
    print("=" * 70)
    print("SCORE STABILITY")
    print("=" * 70)

    print(
        stability.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Week-by-week
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("WEEK-BY-WEEK CAPTURE")
    print("=" * 70)

    pivot = (
        backtest
        .pivot_table(
            index="decision_week",
            columns="model",
            values="capture_rate",
        )
        .reset_index()
    )

    print(
        pivot.to_string(
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

    backtest_file = (
        output_dir
        / "score_design_backtest.csv"
    )

    summary_file = (
        output_dir
        / "score_design_summary.csv"
    )

    stability_file = (
        output_dir
        / "score_design_stability.csv"
    )

    backtest.to_csv(
        backtest_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    stability.to_csv(
        stability_file,
        index=False,
    )

    print("\n")
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(backtest_file)
    print(summary_file)
    print(stability_file)

    print("\n")
    print("=" * 70)
    print("STEP 12 COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()