from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ============================================================
# PATHS
# ============================================================

INPUT = (
    ROOT
    / "analysis"
    / "outputs"
    / "combined_gateway_week_corrected.csv"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BUSINESS PARAMETERS
# ============================================================

TOP_K = 15

FAILURE_THRESHOLD = 0.60

MISSED_FAILURE_COST = 600

# Minimum consecutive bad weeks to call it an episode.
MIN_EPISODE_WEEKS = 1


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

required_columns = [
    "gateway_norm",
    "week_start",
    "success_rate",
    "success_change",
    "low_success_streak",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce",
)

for col in required_columns[2:]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = (
    df
    .dropna(
        subset=[
            "gateway_norm",
            "week_start",
        ]
    )
    .sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# RELATIVE RISK FUNCTION
# ============================================================

def percentile_risk(
    series: pd.Series,
    higher_is_worse: bool,
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = values.median()

    if pd.isna(median):
        median = 0.0

    values = values.fillna(median)

    ranks = values.rank(
        pct=True,
        method="average",
    )

    if higher_is_worse:
        return ranks

    return 1.0 - ranks


# ============================================================
# RANKING STRATEGIES
# ============================================================

def score_strategy(
    frame: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:

    x = frame.copy()

    risk_success = percentile_risk(
        x["success_rate"],
        higher_is_worse=False,
    )

    risk_change = percentile_risk(
        x["success_change"],
        higher_is_worse=False,
    )

    risk_persistence = percentile_risk(
        x["low_success_streak"],
        higher_is_worse=True,
    )

    technical_parts = []

    for col in [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]:
        technical_parts.append(
            percentile_risk(
                x[col],
                higher_is_worse=True,
            )
        )

    technical = pd.concat(
        technical_parts,
        axis=1,
    ).mean(axis=1)

    if strategy == "A_current_success":

        x["score"] = (
            risk_success
        )

    elif strategy == "B_success_deterioration":

        x["score"] = (
            0.60 * risk_success
            +
            0.40 * risk_change
        )

    elif strategy == "C_current_production":

        x["score"] = (
            0.45 * risk_success
            +
            0.35 * risk_change
            +
            0.10 * risk_persistence
            +
            0.10 * technical
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    return (
        x
        .sort_values(
            [
                "score",
                "gateway_norm",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


STRATEGIES = [
    "A_current_success",
    "B_success_deterioration",
    "C_current_production",
]

STRATEGY_LABELS = {
    "A_current_success":
        "A. Current success only",

    "B_success_deterioration":
        "B. Success + deterioration",

    "C_current_production":
        "C. Current production score",
}


# ============================================================
# IDENTIFY FAILURE EPISODES
#
# An episode begins when success falls below 60% and the
# previous available week was not already below 60%.
#
# This is a historical evaluation label only.
# It is NEVER used as a feature.
# ============================================================

df["failure_week"] = (
    df["success_rate"]
    < FAILURE_THRESHOLD
)

previous_failure = (
    df
    .groupby("gateway_norm")["failure_week"]
    .shift(1)
    .fillna(False)
)

df["episode_start"] = (
    df["failure_week"]
    &
    (~previous_failure)
)


# ============================================================
# ASSIGN EPISODE IDS
# ============================================================

episode_counter = (
    df["episode_start"]
    .astype(int)
    .groupby(df["gateway_norm"])
    .cumsum()
)

df["episode_id"] = np.where(
    df["failure_week"],
    (
        df["gateway_norm"].astype(str)
        + "_E"
        + episode_counter.astype(str)
    ),
    None,
)


# ============================================================
# KEEP ONLY EPISODES THAT EXIST
# ============================================================

episode_starts = df[
    df["episode_start"]
].copy()

if episode_starts.empty:
    raise RuntimeError(
        "No failure episodes found."
    )


# Determine episode end and duration.
episodes = []

for gateway, gateway_df in df.groupby(
    "gateway_norm"
):

    gateway_df = (
        gateway_df
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    failure_mask = (
        gateway_df["failure_week"]
        .tolist()
    )

    weeks = (
        gateway_df["week_start"]
        .tolist()
    )

    i = 0

    while i < len(gateway_df):

        if not failure_mask[i]:
            i += 1
            continue

        start_index = i

        while (
            i + 1 < len(gateway_df)
            and failure_mask[i + 1]
        ):
            i += 1

        end_index = i

        duration = (
            end_index
            - start_index
            + 1
        )

        if duration >= MIN_EPISODE_WEEKS:

            episodes.append(
                {
                    "gateway_norm":
                        gateway,

                    "episode_start":
                        weeks[start_index],

                    "episode_end":
                        weeks[end_index],

                    "episode_weeks":
                        duration,
                }
            )

        i += 1


episodes_df = pd.DataFrame(
    episodes
)

if episodes_df.empty:
    raise RuntimeError(
        "No usable failure episodes found."
    )


# ============================================================
# EVALUATE EACH EPISODE
# ============================================================

episode_results = []


for _, episode in episodes_df.iterrows():

    gateway = episode[
        "gateway_norm"
    ]

    episode_start = episode[
        "episode_start"
    ]

    episode_end = episode[
        "episode_end"
    ]

    # --------------------------------------------------------
    # Important:
    #
    # The strategy for the episode must be based only on
    # information available BEFORE each decision week.
    # --------------------------------------------------------

    for strategy in STRATEGIES:

        first_selection_week = None

        selection_position = None

        # ----------------------------------------------------
        # Check the decision week immediately before the
        # failure episode starts, then subsequent weeks while
        # the episode remains active.
        # ----------------------------------------------------

        candidate_weeks = (
            df[
                (
                    df["week_start"]
                    <= episode_end
                )
            ]["week_start"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        for decision_week in candidate_weeks:

            # We need a decision before or at episode start.
            #
            # Once the episode has started, later selection is
            # still useful, but it is no longer "early".
            #

            if decision_week > episode_end:
                continue

            # ------------------------------------------------
            # STRICT CUTOFF
            # ------------------------------------------------

            historical = df[
                df["week_start"]
                < decision_week
            ].copy()

            if historical.empty:
                continue

            # Latest known feature record per gateway.
            latest = (
                historical
                .sort_values(
                    [
                        "gateway_norm",
                        "week_start",
                    ]
                )
                .groupby(
                    "gateway_norm",
                    as_index=False,
                )
                .tail(1)
                .copy()
            )

            if len(latest) < TOP_K:
                continue

            ranked = score_strategy(
                latest,
                strategy,
            )

            top15 = set(
                ranked.head(TOP_K)[
                    "gateway_norm"
                ]
            )

            if gateway not in top15:
                continue

            first_selection_week = (
                decision_week
            )

            # Position is computed for reporting only.
            ranked["rank"] = (
                ranked.index + 1
            )

            gateway_rank = ranked.loc[
                ranked["gateway_norm"]
                == gateway,
                "rank",
            ]

            if not gateway_rank.empty:
                selection_position = int(
                    gateway_rank.iloc[0]
                )

            break

        # ----------------------------------------------------
        # CLASSIFY TIMING
        # ----------------------------------------------------

        if first_selection_week is None:

            detection_class = (
                "Never selected"
            )

            weeks_before_start = np.nan

        else:

            weeks_before_start = int(
                (
                    episode_start
                    - first_selection_week
                ).days
                / 7
            )

            if weeks_before_start > 0:

                detection_class = (
                    "Caught before episode"
                )

            elif weeks_before_start == 0:

                detection_class = (
                    "Caught at episode start"
                )

            else:

                detection_class = (
                    "Caught after episode started"
                )

        # ----------------------------------------------------
        # Approximate missed failure cost
        #
        # This is used as a comparative timing metric:
        # number of episode weeks before first selection.
        # ----------------------------------------------------

        if first_selection_week is None:

            missed_weeks = int(
                episode["episode_weeks"]
            )

        else:

            if first_selection_week <= episode_start:

                missed_weeks = 0

            else:

                weeks_late = int(
                    (
                        first_selection_week
                        - episode_start
                    ).days
                    / 7
                )

                missed_weeks = min(
                    weeks_late,
                    int(
                        episode["episode_weeks"]
                    ),
                )

        missed_cost = (
            missed_weeks
            * MISSED_FAILURE_COST
        )

        episode_results.append(
            {
                "gateway_norm":
                    gateway,

                "episode_start":
                    episode_start,

                "episode_end":
                    episode_end,

                "episode_weeks":
                    episode["episode_weeks"],

                "strategy":
                    strategy,

                "strategy_label":
                    STRATEGY_LABELS[strategy],

                "first_selection_week":
                    first_selection_week,

                "selection_rank":
                    selection_position,

                "detection_class":
                    detection_class,

                "weeks_before_episode":
                    weeks_before_start,

                "missed_failure_weeks":
                    missed_weeks,

                "estimated_missed_cost":
                    missed_cost,
            }
        )


episode_result_df = pd.DataFrame(
    episode_results
)


# ============================================================
# SUMMARY BY STRATEGY
# ============================================================

summary_rows = []


for strategy in STRATEGIES:

    subset = episode_result_df[
        episode_result_df["strategy"]
        == strategy
    ].copy()

    total = len(subset)

    before = int(
        (
            subset["detection_class"]
            == "Caught before episode"
        ).sum()
    )

    at_start = int(
        (
            subset["detection_class"]
            == "Caught at episode start"
        ).sum()
    )

    after = int(
        (
            subset["detection_class"]
            == "Caught after episode started"
        ).sum()
    )

    never = int(
        (
            subset["detection_class"]
            == "Never selected"
        ).sum()
    )

    early_rate = (
        before / total
        if total > 0
        else np.nan
    )

    at_or_before_rate = (
        (
            before
            + at_start
        )
        / total
        if total > 0
        else np.nan
    )

    mean_missed_cost = (
        subset[
            "estimated_missed_cost"
        ].mean()
    )

    total_missed_cost = (
        subset[
            "estimated_missed_cost"
        ].sum()
    )

    summary_rows.append(
        {
            "strategy":
                strategy,

            "strategy_label":
                STRATEGY_LABELS[strategy],

            "episodes_evaluated":
                total,

            "caught_before_episode":
                before,

            "caught_at_episode_start":
                at_start,

            "caught_after_start":
                after,

            "never_selected":
                never,

            "early_detection_rate":
                early_rate,

            "caught_at_or_before_start_rate":
                at_or_before_rate,

            "mean_missed_failure_cost":
                mean_missed_cost,

            "total_estimated_missed_cost":
                total_missed_cost,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 84)
print("NEXORA 2026 — EPISODE EARLY-DETECTION ANALYSIS")
print("=" * 84)

print()
print(
    f"Failure definition: "
    f"success_rate < {FAILURE_THRESHOLD:.0%}"
)

print(
    f"Failure episodes evaluated: "
    f"{len(episodes_df)}"
)

print(
    f"Top-K field capacity: "
    f"{TOP_K}"
)

print()
print("-" * 84)

display = summary[
    [
        "strategy_label",
        "episodes_evaluated",
        "caught_before_episode",
        "caught_at_episode_start",
        "caught_after_start",
        "never_selected",
        "early_detection_rate",
        "caught_at_or_before_start_rate",
        "mean_missed_failure_cost",
        "total_estimated_missed_cost",
    ]
].copy()

display[
    "early_detection_rate"
] = (
    display[
        "early_detection_rate"
    ].round(3)
)

display[
    "caught_at_or_before_start_rate"
] = (
    display[
        "caught_at_or_before_start_rate"
    ].round(3)
)

display[
    "mean_missed_failure_cost"
] = (
    display[
        "mean_missed_failure_cost"
    ].round(0)
)

display[
    "total_estimated_missed_cost"
] = (
    display[
        "total_estimated_missed_cost"
    ].round(0)
)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# BEST STRATEGY BY EARLY DETECTION
# ============================================================

best_early = summary.loc[
    summary[
        "early_detection_rate"
    ].idxmax()
]

best_cost = summary.loc[
    summary[
        "total_estimated_missed_cost"
    ].idxmin()
]


print()
print("-" * 84)

print(
    "BEST BY EARLY-DETECTION RATE"
)

print(
    best_early[
        "strategy_label"
    ]
)

print(
    f"Early detection rate: "
    f"{best_early['early_detection_rate']:.1%}"
)


print()
print(
    "BEST BY ESTIMATED MISSED-FAILURE COST"
)

print(
    best_cost[
        "strategy_label"
    ]
)

print(
    f"Total estimated missed cost: "
    f"€{best_cost['total_estimated_missed_cost']:,.0f}"
)


# ============================================================
# SAVE
# ============================================================

episodes_output = (
    OUTPUT_DIR
    / "failure_episodes.csv"
)

detail_output = (
    OUTPUT_DIR
    / "episode_early_detection_detail.csv"
)

summary_output = (
    OUTPUT_DIR
    / "episode_early_detection_summary.csv"
)

episodes_df.to_csv(
    episodes_output,
    index=False,
)

episode_result_df.to_csv(
    detail_output,
    index=False,
)

summary.to_csv(
    summary_output,
    index=False,
)


print()
print("=" * 84)
print("OUTPUTS CREATED")
print("=" * 84)

print(episodes_output)
print(detail_output)
print(summary_output)

print()
print("DONE")