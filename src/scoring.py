from __future__ import annotations

import pandas as pd

TOP_K = 15


def percentile_risk(
    series: pd.Series,
    higher_is_worse: bool,
) -> pd.Series:
    """
    Convert a feature into a 0-1 relative risk score.

    Higher returned values always mean higher risk.

    Missing values are treated as neutral risk (0.5),
    because missing information is not evidence of either
    high or low operational risk.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    observed = values.notna()

    # Neutral risk for missing values.
    risk = pd.Series(
        0.5,
        index=series.index,
        dtype=float,
    )

    # Rank only observed values.
    if observed.any():
        ranks = values.loc[observed].rank(
            pct=True,
            method="average",
        )

        if higher_is_worse:
            risk.loc[observed] = ranks
        else:
            risk.loc[observed] = 1.0 - ranks

    return risk


def calculate_risk_scores(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the Data Science visit-priority score.

    Design:
        45% current business risk
        35% business deterioration
        10% persistence
        10% technical confirmation
    """

    df = weekly.copy()

    # --------------------------------------------------------
    # 1. Current business risk
    # --------------------------------------------------------
    # Lower meter-read success = higher risk.
    df["risk_success"] = percentile_risk(
        df["success_rate"],
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # 2. Business deterioration
    # --------------------------------------------------------
    # More negative success_change = higher risk.
    #
    # Missing success_change means there is no newer
    # meter observation available, so it receives neutral risk.
    df["risk_success_change"] = percentile_risk(
        df["success_change"],
        higher_is_worse=False,
    )

    # --------------------------------------------------------
    # 3. Persistence
    # --------------------------------------------------------
    # Longer low-success streak = higher risk.
    df["risk_persistence"] = percentile_risk(
        df["low_success_streak"],
        higher_is_worse=True,
    )

    # --------------------------------------------------------
    # 4. Technical confirmation
    # --------------------------------------------------------
    df["risk_offline"] = percentile_risk(
        df["offline_duration_sec"],
        higher_is_worse=True,
    )

    df["risk_disconnect"] = percentile_risk(
        df["disconnection_cnt"],
        higher_is_worse=True,
    )

    df["risk_reboot"] = percentile_risk(
        df["reboot_cnt"],
        higher_is_worse=True,
    )

    df["risk_reboot_duration"] = percentile_risk(
        df["reboot_duration_sec"],
        higher_is_worse=True,
    )

    df["technical_confirmation"] = (
        df["risk_offline"]
        + df["risk_disconnect"]
        + df["risk_reboot"]
        + df["risk_reboot_duration"]
    ) / 4.0

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    df["business_risk"] = (
        0.50 * df["risk_success"]
        + 0.50 * df["risk_success_change"]
    )

    df["visit_priority_score"] = (
        0.45 * df["risk_success"]
        + 0.35 * df["risk_success_change"]
        + 0.10 * df["risk_persistence"]
        + 0.10 * df["technical_confirmation"]
    )

    return df


def create_reason(row: pd.Series) -> str:
    """
    Create a concise explanation for an operations manager.

    Maximum allowed length: 300 characters.
    """

    reasons = []

    # --------------------------------------------------------
    # Current meter-read performance
    # --------------------------------------------------------
    success = row.get("success_rate")

    if pd.notna(success) and success < 0.80:
        reasons.append(
            f"meter-read success is {success:.0%}"
        )

    # --------------------------------------------------------
    # Recent deterioration
    # --------------------------------------------------------
    change = row.get("success_change")

    if pd.notna(change) and change <= -0.10:
        reasons.append(
            f"success fell {abs(change):.0%} recently"
        )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------
    streak = row.get("low_success_streak")

    if pd.notna(streak) and streak >= 2:
        reasons.append(
            f"{int(streak)} consecutive low-success weeks"
        )

    # --------------------------------------------------------
    # Meter-data freshness
    # --------------------------------------------------------
    meter_age = row.get("meter_age_weeks")

    if (
        pd.isna(change)
        and pd.notna(meter_age)
        and meter_age >= 1
    ):
        age = int(meter_age)

        if age == 1:
            reasons.append(
                "latest meter-read data is 1 week old"
            )
        else:
            reasons.append(
                f"latest meter-read data is {age} weeks old"
            )

    # --------------------------------------------------------
    # Technical signals
    # --------------------------------------------------------
    offline = row.get("offline_duration_sec")

    if pd.notna(offline) and offline > 0:
        reasons.append(
            f"{offline:,.0f}s offline"
        )

    disconnect = row.get("disconnection_cnt")

    if pd.notna(disconnect) and disconnect >= 100:
        reasons.append(
            f"{disconnect:.0f} disconnections"
        )

    reboot = row.get("reboot_cnt")

    if pd.notna(reboot) and reboot >= 10:
        reasons.append(
            f"{reboot:.0f} reboots"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    if not reasons:
        reasons.append(
            "highest relative visit-priority score"
        )

    reason = "; ".join(reasons)

    return reason[:300]


def rank_top_gateways(
    weekly: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Score and rank gateways for one decision week.
    """

    scored = calculate_risk_scores(
        weekly
    )

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

    scored["rank"] = (
        scored.index + 1
    )

    selected = scored.head(
        top_k
    ).copy()

    selected["reason"] = selected.apply(
        create_reason,
        axis=1,
    )

    return selected
