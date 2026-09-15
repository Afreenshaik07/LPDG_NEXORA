from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import ChallengeData
from src.scoring import calculate_risk_scores, create_reason


# ============================================================
# CHALLENGE SETTINGS
# ============================================================

DEFAULT_START_WEEK = pd.Timestamp("2026-02-02")
DEFAULT_NUM_WEEKS = 8

VISITS_PER_WEEK = 15


# ============================================================
# BUILD SCORING WEEKS
# ============================================================

def build_scored_weeks(
    start_week: pd.Timestamp,
    num_weeks: int,
) -> list[pd.Timestamp]:
    """
    Build consecutive Monday decision weeks.

    Keeping the start week and number of weeks configurable
    allows the same code to be exercised on a later unseen
    period during the live session.
    """

    start_week = pd.Timestamp(start_week)

    if start_week.weekday() != 0:
        raise ValueError(
            f"start_week must be a Monday, got {start_week.date()}."
        )

    if num_weeks < 1:
        raise ValueError(
            "num_weeks must be at least 1."
        )

    return [
        start_week + pd.Timedelta(weeks=i)
        for i in range(num_weeks)
    ]


# ============================================================
# NORMALIZE GATEWAY ID
# ============================================================

def normalize_id(series: pd.Series) -> pd.Series:
    """Convert gateway IDs to the accepted normalized format."""

    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.strip()
        .str.upper()
    )


# ============================================================
# PREPARE WEEKLY DATA
# ============================================================

def prepare_weekly_data(
    data: ChallengeData,
) -> pd.DataFrame:
    """
    Load and prepare the complete gateway-week dataset.
    """

    weekly = data.build_weekly_dataset()

    weekly["gateway_norm"] = normalize_id(
        weekly["gateway_norm"]
    )

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"],
        errors="coerce",
    )

    if weekly["week_start"].isna().any():
        raise ValueError(
            "weekly dataset contains invalid week_start values."
        )

    return weekly


# ============================================================
# DETERMINE ACTIVE GATEWAYS
# ============================================================

def active_gateways(
    weekly: pd.DataFrame,
    decision_week: pd.Timestamp,
) -> pd.DataFrame:
    """
    Keep gateways that were active at the decision date.

    A blank decommissioned_on means the gateway is still active.
    """

    candidates = weekly.copy()

    if "decommissioned_on" in candidates.columns:

        decommissioned = pd.to_datetime(
            candidates["decommissioned_on"],
            errors="coerce",
        )

        candidates = candidates[
            decommissioned.isna()
            | (decommissioned >= decision_week)
        ].copy()

    return candidates


# ============================================================
# BUILD DECISION DATA
# ============================================================

def build_decision_frame(
    weekly: pd.DataFrame,
    decision_week: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the gateway snapshot available for one decision Monday.

    Only information strictly before decision_week is used.

    For each gateway, the most recent complete available
    gateway-week is selected.
    """

    historical = weekly[
        weekly["week_start"] < decision_week
    ].copy()

    if historical.empty:
        raise ValueError(
            f"No historical data exists before "
            f"{decision_week.date()}."
        )

    historical = active_gateways(
        historical,
        decision_week,
    )

    if historical.empty:
        raise ValueError(
            f"No active gateway data exists before "
            f"{decision_week.date()}."
        )

    historical = historical.sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )

    latest = (
        historical
        .groupby(
            "gateway_norm",
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    if len(latest) < VISITS_PER_WEEK:
        raise ValueError(
            f"Only {len(latest)} eligible gateways are "
            f"available before {decision_week.date()}."
        )

    return latest.reset_index(drop=True)


# ============================================================
# GENERATE ONE WEEK
# ============================================================

def generate_week_predictions(
    weekly: pd.DataFrame,
    decision_week: pd.Timestamp,
) -> pd.DataFrame:
    """
    Generate the top 15 gateway recommendations for one week.
    """

    candidates = build_decision_frame(
        weekly,
        decision_week,
    )

    # --------------------------------------------------------
    # Calculate risk scores
    # --------------------------------------------------------

    scored = calculate_risk_scores(
        candidates
    )

    if "visit_priority_score" not in scored.columns:
        raise ValueError(
            "Scoring function did not create "
            "'visit_priority_score'."
        )

    # --------------------------------------------------------
    # Rank highest priority first
    # --------------------------------------------------------

    scored = scored.sort_values(
        [
            "visit_priority_score",
            "gateway_norm",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Select exactly 15 gateways
    # --------------------------------------------------------

    selected = scored.head(
        VISITS_PER_WEEK
    ).copy()

    selected["rank"] = (
        selected.index + 1
    )

    # --------------------------------------------------------
    # Create reason
    # --------------------------------------------------------

    selected["reason"] = selected.apply(
        create_reason,
        axis=1,
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    selected["score"] = (
        selected["visit_priority_score"]
        .astype(float)
    )

    selected["week_start"] = (
        decision_week.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # Final submission columns
    # --------------------------------------------------------

    output = selected[
        [
            "week_start",
            "rank",
            "gateway_norm",
            "score",
            "reason",
        ]
    ].rename(
        columns={
            "gateway_norm": "gateway_id",
        }
    )

    return output


# ============================================================
# GENERATE ALL WEEKS
# ============================================================

def generate_all_predictions(
    weekly: pd.DataFrame,
    scored_weeks: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Generate predictions for all requested decision weeks.
    """

    all_predictions = []

    print("\nGenerating predictions...")

    for decision_week in scored_weeks:

        print(
            f"Processing week "
            f"{decision_week.date()}..."
        )

        prediction = generate_week_predictions(
            weekly,
            decision_week,
        )

        all_predictions.append(
            prediction
        )

        print(
            f"  Selected {len(prediction)} gateways."
        )

    if not all_predictions:
        raise ValueError(
            "No predictions were generated."
        )

    result = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    return result


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def validate_predictions(
    predictions: pd.DataFrame,
    scored_weeks: list[pd.Timestamp],
) -> None:
    """
    Validate the generated output before writing it.
    """

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    ]

    if list(predictions.columns) != expected_columns:
        raise ValueError(
            "Incorrect output columns.\n"
            f"Expected: {expected_columns}\n"
            f"Got:      {list(predictions.columns)}"
        )

    expected_rows = (
        len(scored_weeks)
        * VISITS_PER_WEEK
    )

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(predictions)}."
        )

    # --------------------------------------------------------
    # Validate each week
    # --------------------------------------------------------

    for decision_week in scored_weeks:

        week_text = decision_week.strftime(
            "%Y-%m-%d"
        )

        part = predictions[
            predictions["week_start"]
            == week_text
        ]

        if len(part) != VISITS_PER_WEEK:
            raise ValueError(
                f"{week_text}: expected "
                f"{VISITS_PER_WEEK} rows, "
                f"got {len(part)}."
            )

        ranks = sorted(
            pd.to_numeric(
                part["rank"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .tolist()
        )

        if ranks != list(
            range(1, VISITS_PER_WEEK + 1)
        ):
            raise ValueError(
                f"{week_text}: ranks must be "
                "1..15 with no repeats."
            )

        gateway_ids = (
            part["gateway_id"]
            .astype(str)
            .str.replace(
                ":",
                "",
                regex=False,
            )
            .str.upper()
            .str.strip()
        )

        if (
            gateway_ids.nunique()
            != VISITS_PER_WEEK
        ):
            raise ValueError(
                f"{week_text}: duplicate gateway "
                "detected."
            )

    # --------------------------------------------------------
    # Validate scores
    # --------------------------------------------------------

    if predictions["score"].isna().any():
        raise ValueError(
            "One or more score values are missing."
        )

    if not pd.api.types.is_numeric_dtype(
        predictions["score"]
    ):
        raise ValueError(
            "score must be numeric."
        )

    # --------------------------------------------------------
    # Validate reasons
    # --------------------------------------------------------

    reasons = (
        predictions["reason"]
        .astype(str)
        .str.strip()
    )

    if (
        reasons == ""
    ).any():
        raise ValueError(
            "One or more reason fields are empty."
        )

    if (
        reasons.str.len() > 300
    ).any():
        raise ValueError(
            "One or more reason fields exceed "
            "300 characters."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Generate NEXORA 2026 Data Science "
            "gateway visit predictions."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help=(
            "Path to the challenge data directory."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("predictions.csv"),
        help=(
            "Output path for predictions.csv."
        ),
    )

    parser.add_argument(
        "--start-week",
        type=str,
        default=DEFAULT_START_WEEK.strftime("%Y-%m-%d"),
        help=(
            "First decision week (Monday). "
            "Defaults to the official challenge period."
        ),
    )

    parser.add_argument(
        "--num-weeks",
        type=int,
        default=DEFAULT_NUM_WEEKS,
        help=(
            "Number of consecutive decision weeks. "
            "Defaults to 8."
        ),
    )

    args = parser.parse_args()

    start_week = pd.Timestamp(
        args.start_week
    )

    scored_weeks = build_scored_weeks(
        start_week,
        args.num_weeks,
    )

    print("=" * 70)
    print(
        "NEXORA 2026 - DATA SCIENCE PREDICTION GENERATOR"
    )
    print("=" * 70)

    print(
        f"\nData directory: {args.data}"
    )

    print(
        f"Output file: {args.out}"
    )

    print(
        f"Decision period: "
        f"{scored_weeks[0].date()} "
        f"to "
        f"{scored_weeks[-1].date()}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = ChallengeData(
        args.data
    )

    weekly = prepare_weekly_data(
        data
    )

    print(
        "\nWeekly dataset shape:",
        weekly.shape,
    )

    print(
        "Available data:",
        weekly["week_start"].min().date(),
        "to",
        weekly["week_start"].max().date(),
    )

    print(
        "Unique gateways:",
        weekly["gateway_norm"].nunique(),
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = generate_all_predictions(
        weekly,
        scored_weeks,
    )

    # --------------------------------------------------------
    # Internal validation
    # --------------------------------------------------------

    validate_predictions(
        predictions,
        scored_weeks,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        args.out,
        index=False,
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREDICTIONS GENERATED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"\nRows written: {len(predictions)}"
    )

    print(
        "Weeks:",
        predictions["week_start"].nunique(),
    )

    print(
        f"Saved to: {args.out}"
    )

    print("\nFirst 15 predictions:")

    print(
        predictions
        .head(15)
        .to_string(index=False)
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )