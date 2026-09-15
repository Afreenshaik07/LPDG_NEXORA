from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NEXORA 2026 | Gateway Visit Prioritization",
    page_icon="N",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

PREDICTIONS_FILE = Path("predictions.csv")

VISITS_PER_WEEK = 15
VISIT_COST_EUR = 380
MISSED_FAULT_COST_EUR = 600


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
    }

    section[data-testid="stSidebar"] {
        background: #071a2d;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #10243d;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 800;
        color: #14263d;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .small-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .big-value {
        color: #10243d;
        font-size: 28px;
        font-weight: 800;
    }

    .reason-box {
        background: #eef5ff;
        border-left: 4px solid #246bdf;
        padding: 15px;
        border-radius: 8px;
        line-height: 1.7;
        color: #334155;
    }

    .info-box {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 14px;
    }

    .critical-text {
        color: #bd3021;
        font-weight: 800;
    }

    .high-text {
        color: #a36800;
        font-weight: 800;
    }

    .medium-text {
        color: #29629e;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

@st.cache_data
def load_predictions() -> pd.DataFrame:

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            "predictions.csv was not found in the project root."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    required = {
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["week_start"] = pd.to_datetime(
        df["week_start"],
        errors="coerce",
    )

    df["rank"] = pd.to_numeric(
        df["rank"],
        errors="coerce",
    )

    df["score"] = pd.to_numeric(
        df["score"],
        errors="coerce",
    )

    df["gateway_id"] = df["gateway_id"].astype(str)

    df = df.dropna(
        subset=[
            "week_start",
            "rank",
            "score",
            "gateway_id",
        ]
    ).copy()

    df["rank"] = df["rank"].astype(int)

    return df.sort_values(
        ["week_start", "rank"]
    ).reset_index(drop=True)


# ============================================================
# PARSE REASON
# ============================================================

def parse_value(pattern: str, text: str):
    match = re.search(
        pattern,
        str(text),
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    value = match.group(1).replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def extract_signals(reason: str):

    return {
        "success": parse_value(
            r"meter-read success is\s+([\d.]+)%",
            reason,
        ),
        "streak": parse_value(
            r"([\d.]+)\s+consecutive low-success weeks",
            reason,
        ),
        "meter_age": parse_value(
            r"latest meter-read data is\s+([\d.]+)\s+weeks?\s+old",
            reason,
        ),
        "offline": parse_value(
            r"([\d,]+)s offline",
            reason,
        ),
        "disconnects": parse_value(
            r"([\d,]+)\s+disconnections",
            reason,
        ),
        "reboots": parse_value(
            r"([\d,]+)\s+reboots",
            reason,
        ),
    }


# ============================================================
# LOAD
# ============================================================

try:
    predictions = load_predictions()

except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


weeks = sorted(
    predictions["week_start"].unique()
)

if not weeks:
    st.error("No prediction weeks found.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("NEXORA 2026")

    st.caption(
        "LPDG Innovation Hub\n"
        "Data Science Operations"
    )

    st.divider()

    st.subheader("Decision Control")

    selected_week = st.selectbox(
        "Decision week",
        weeks,
        format_func=lambda x:
            pd.Timestamp(x).strftime(
                "%d %b %Y"
            ),
    )

    st.subheader("View")

    view = st.radio(
        "Select view",
        [
            "Weekly Decision",
            "Gateway Detail",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("DATA STATUS")

    st.metric(
        "Prediction rows",
        len(predictions),
    )

    st.metric(
        "Prediction weeks",
        predictions["week_start"].nunique(),
    )

    st.metric(
        "Gateways ranked",
        predictions["gateway_id"].nunique(),
    )

    st.metric(
        "Weekly capacity",
        VISITS_PER_WEEK,
    )


# ============================================================
# SELECTED WEEK
# ============================================================

week_df = predictions[
    predictions["week_start"]
    == pd.Timestamp(selected_week)
].sort_values("rank").copy()

if len(week_df) == 0:
    st.warning("No predictions for this week.")
    st.stop()


top = week_df.iloc[0]

average_score = week_df["score"].mean()
highest_score = week_df["score"].max()

weekly_cost = VISITS_PER_WEEK * VISIT_COST_EUR

total_cost = (
    predictions["week_start"].nunique()
    * VISITS_PER_WEEK
    * VISIT_COST_EUR
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="small-label">LPDG INNOVATION HUB SELECTION CHALLENGE 2026</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">Gateway Visit Prioritization</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="subtitle">
        Evidence-based decision support for the weekly
        15-visit field-service capacity.
        <br>
        Decision week:
        <strong>
        {pd.Timestamp(selected_week).strftime("%d %B %Y")}
        </strong>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI
# ============================================================

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Visits planned",
        len(week_df),
    )

with k2:
    st.metric(
        "Weekly dispatch cost",
        f"€{weekly_cost:,.0f}",
    )

with k3:
    st.metric(
        "Highest priority",
        "#1",
    )

with k4:
    st.metric(
        "Top score",
        f"{highest_score:.4f}",
    )

with k5:
    st.metric(
        "Average score",
        f"{average_score:.4f}",
    )


# ============================================================
# WEEKLY DECISION
# ============================================================

if view == "Weekly Decision":

    st.markdown(
        '<div class="section-title">Priority Overview</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.35, 0.85]
    )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Top 15 Gateway Ranking"
        )

        display_df = week_df[
            [
                "rank",
                "gateway_id",
                "score",
                "reason",
            ]
        ].copy()

        display_df["priority"] = display_df[
            "rank"
        ].apply(
            lambda x:
                "Critical"
                if x <= 5
                else (
                    "High"
                    if x <= 10
                    else "Medium"
                )
        )

        display_df = display_df[
            [
                "rank",
                "gateway_id",
                "score",
                "priority",
                "reason",
            ]
        ]

        display_df.columns = [
            "Rank",
            "Gateway",
            "Score",
            "Priority",
            "Why selected",
        ]

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            height=570,
        )

    # --------------------------------------------------------
    # TOP GATEWAY
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Highest-Priority Gateway"
        )

        st.markdown(
            f"### {top['gateway_id']}"
        )

        if int(top["rank"]) <= 5:
            st.error("CRITICAL")

        elif int(top["rank"]) <= 10:
            st.warning("HIGH")

        else:
            st.info("MEDIUM")

        st.metric(
            "Priority score",
            f"{float(top['score']):.4f}",
        )

        signals = extract_signals(
            top["reason"]
        )

        st.markdown("#### Operational signals")

        s1, s2 = st.columns(2)

        with s1:
            success = signals["success"]
            st.metric(
                "Meter success",
                f"{success:.0f}%"
                if success is not None
                else "—",
            )

        with s2:
            streak = signals["streak"]
            st.metric(
                "Persistence",
                f"{streak:.0f} weeks"
                if streak is not None
                else "—",
            )

        s3, s4 = st.columns(2)

        with s3:
            age = signals["meter_age"]
            st.metric(
                "Meter age",
                f"{age:.0f} weeks"
                if age is not None
                else "—",
            )

        with s4:
            disconnects = signals["disconnects"]
            st.metric(
                "Disconnections",
                f"{disconnects:,.0f}"
                if disconnects is not None
                else "—",
            )

        st.markdown(
            f"""
            <div class="reason-box">
            <strong>Why selected</strong><br>
            {top["reason"]}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# GATEWAY DETAIL
# ============================================================

elif view == "Gateway Detail":

    st.markdown(
        '<div class="section-title">Gateway Investigation</div>',
        unsafe_allow_html=True,
    )

    options = [
        f"Rank {int(row['rank']):02d} — {row['gateway_id']}"
        for _, row in week_df.iterrows()
    ]

    selected_gateway = st.selectbox(
        "Select gateway",
        options,
    )

    match = re.search(
        r"Rank\s+(\d+)",
        selected_gateway,
    )

    rank = int(match.group(1))

    row = week_df[
        week_df["rank"] == rank
    ].iloc[0]

    signals = extract_signals(
        row["reason"]
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            row["gateway_id"]
        )

        st.metric(
            "Rank",
            f"#{rank}",
        )

        st.metric(
            "Priority score",
            f"{float(row['score']):.4f}",
        )

        if rank <= 5:
            st.error("CRITICAL")
        elif rank <= 10:
            st.warning("HIGH")
        else:
            st.info("MEDIUM")

    with right:

        st.subheader(
            "Operational Signals"
        )

        a, b = st.columns(2)

        with a:
            st.metric(
                "Meter success",
                (
                    f"{signals['success']:.0f}%"
                    if signals["success"] is not None
                    else "—"
                ),
            )

            st.metric(
                "Offline",
                (
                    f"{signals['offline']:,.0f} sec"
                    if signals["offline"] is not None
                    else "—"
                ),
            )

            st.metric(
                "Reboots",
                (
                    f"{signals['reboots']:,.0f}"
                    if signals["reboots"] is not None
                    else "—"
                ),
            )

        with b:
            st.metric(
                "Persistence",
                (
                    f"{signals['streak']:.0f} weeks"
                    if signals["streak"] is not None
                    else "—"
                ),
            )

            st.metric(
                "Disconnections",
                (
                    f"{signals['disconnects']:,.0f}"
                    if signals["disconnects"] is not None
                    else "—"
                ),
            )

            st.metric(
                "Meter age",
                (
                    f"{signals['meter_age']:.0f} weeks"
                    if signals["meter_age"] is not None
                    else "—"
                ),
            )

    st.markdown("### Why this gateway was selected")

    st.info(
        row["reason"]
    )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.markdown(
        '<div class="section-title">Decision Methodology</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "The final score is an ordinal visit-priority score, "
        "not a calibrated failure probability."
    )

    st.subheader(
        "Final Composite Score"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Current success risk",
            "45%",
        )

    with m2:
        st.metric(
            "Recent deterioration",
            "35%",
        )

    with m3:
        st.metric(
            "Persistence",
            "10%",
        )

    with m4:
        st.metric(
            "Technical confirmation",
            "10%",
        )

    st.subheader(
        "Decision logic"
    )

    st.write(
        """
        Current meter-read performance is the strongest
        business-facing signal.

        Recent deterioration identifies gateways whose
        performance is getting worse.

        Persistence captures repeated low-success behavior.

        Technical telemetry provides supporting evidence
        through offline duration, disconnections and
        reboot activity.

        The gateways are ranked by the resulting score
        and the top 15 are selected because field capacity
        is limited to 15 visits per week.
        """
    )


# ============================================================
# ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">Weekly Analytics</div>',
    unsafe_allow_html=True,
)

a1, a2 = st.columns(2)

with a1:

    st.subheader(
        "Priority Mix"
    )

    tier_counts = (
        week_df["rank"]
        .apply(
            lambda x:
                "Critical"
                if x <= 5
                else (
                    "High"
                    if x <= 10
                    else "Medium"
                )
        )
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium"]
        )
        .fillna(0)
    )

    st.bar_chart(
        tier_counts,
        width="stretch",
        height=280,
    )

with a2:

    st.subheader(
        "Average Score by Week"
    )

    weekly_scores = (
        predictions
        .groupby("week_start")["score"]
        .mean()
        .sort_index()
    )

    chart_df = weekly_scores.to_frame(
        name="Average score"
    )

    st.line_chart(
        chart_df,
        width="stretch",
        height=280,
    )


# ============================================================
# BUSINESS IMPACT
# ============================================================

st.markdown(
    '<div class="section-title">Business Impact</div>',
    unsafe_allow_html=True,
)

b1, b2, b3 = st.columns(3)

with b1:
    st.metric(
        "Cost per visit",
        "€380",
    )

with b2:
    st.metric(
        "Weekly planned spend",
        f"€{weekly_cost:,.0f}",
    )

with b3:
    st.metric(
        "8-week planned spend",
        f"€{total_cost:,.0f}",
    )

st.info(
    f"Challenge cost context: €{MISSED_FAULT_COST_EUR} "
    "per unresolved faulty gateway-week."
)


# ============================================================
# FOOTNOTE
# ============================================================

st.caption(
    "NEXORA 2026 • Gateway Visit Prioritization • "
)