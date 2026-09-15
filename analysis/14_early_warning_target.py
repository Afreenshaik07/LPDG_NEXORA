from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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
# CONFIGURATION
# ============================================================

FUTURE_WEEKS = 2

FAILURE_THRESHOLD = 0.60

LOW_SUCCESS_THRESHOLD = 0.80


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(
        f"Required analysis output not found: {INPUT}"
    )

df = pd.read_csv(
    INPUT
)

required_columns = [
    "gateway_norm",
    "week_start",
    "success_rate",
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
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

df["success_rate"] = pd.to_numeric(
    df["success_rate"],
    errors="coerce",
)

df = (
    df
    .dropna(
        subset=[
            "gateway_norm",
            "week_start",
            "success_rate",
        ]
    )
    .copy()
)


# ============================================================
# SORT HISTORICALLY
# ============================================================

df = (
    df
    .sort_values(
        [
            "gateway_norm",
            "week_start",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# FUTURE TARGET
# ============================================================

grouped = df.groupby(
    "gateway_norm",
    group_keys=False,
)


future_1 = grouped["success_rate"].shift(-1)

future_2 = grouped["success_rate"].shift(-2)


df["future_week_1_success"] = future_1

df["future_week_2_success"] = future_2


# Require both future weeks to exist.
df["future_complete"] = (
    df["future_week_1_success"].notna()
    & df["future_week_2_success"].notna()
)


# A future failure episode means that within the
# following two weeks at least one week falls below 60%.
df["future_failure"] = np.where(
    df["future_complete"],
    (
        (df["future_week_1_success"] < FAILURE_THRESHOLD)
        |
        (df["future_week_2_success"] < FAILURE_THRESHOLD)
    ).astype(int),
    np.nan,
)


# ============================================================
# FUTURE SUSTAINED RISK
# ============================================================

# Additional diagnostic:
# both future weeks remain below 80%.
df["future_persistent_low_success"] = np.where(
    df["future_complete"],
    (
        (df["future_week_1_success"] < LOW_SUCCESS_THRESHOLD)
        &
        (df["future_week_2_success"] < LOW_SUCCESS_THRESHOLD)
    ).astype(int),
    np.nan,
)


# ============================================================
# CURRENT SIGNALS
# ============================================================

df["current_low_success"] = (
    df["success_rate"] < LOW_SUCCESS_THRESHOLD
).astype(int)


# ============================================================
# KEEP ONLY USABLE HISTORICAL ROWS
# ============================================================

usable = df[
    df["future_complete"]
].copy()


# ============================================================
# BASIC TARGET STATISTICS
# ============================================================

total = len(usable)

future_failures = int(
    usable["future_failure"].sum()
)

future_failure_rate = (
    future_failures / total
    if total > 0
    else np.nan
)

persistent = int(
    usable["future_persistent_low_success"].sum()
)

persistent_rate = (
    persistent / total
    if total > 0
    else np.nan
)


print()
print("=" * 70)
print("NEXORA 2026 — EARLY WARNING TARGET ANALYSIS")
print("=" * 70)

print()
print(f"Input rows: {len(df):,}")
print(f"Usable historical rows: {total:,}")
print(
    f"Future failure threshold: "
    f"success_rate < {FAILURE_THRESHOLD:.0%}"
)
print(
    f"Future horizon: "
    f"next {FUTURE_WEEKS} complete weeks"
)

print()
print(
    f"Future failure cases: "
    f"{future_failures:,}"
)

print(
    f"Future failure rate: "
    f"{future_failure_rate:.1%}"
)

print()
print(
    f"Future persistent low-success cases: "
    f"{persistent:,}"
)

print(
    f"Future persistent low-success rate: "
    f"{persistent_rate:.1%}"
)


# ============================================================
# CURRENT PERFORMANCE VS FUTURE FAILURE
# ============================================================

comparison = (
    usable
    .groupby("future_failure")[
        "success_rate"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
)

comparison.index = [
    "No future failure",
    "Future failure",
]

print()
print("=" * 70)
print("CURRENT SUCCESS RATE VS FUTURE FAILURE")
print("=" * 70)

print(
    comparison.round(3).to_string()
)


# ============================================================
# SIMPLE CURRENT-STATE THRESHOLDS
# ============================================================

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


rows = []

for threshold in thresholds:

    selected = (
        usable["success_rate"]
        < threshold
    )

    n_selected = int(
        selected.sum()
    )

    captured = int(
        (
            selected
            & (usable["future_failure"] == 1)
        ).sum()
    )

    total_failures = int(
        usable["future_failure"].sum()
    )

    recall = (
        captured / total_failures
        if total_failures > 0
        else np.nan
    )

    precision = (
        captured / n_selected
        if n_selected > 0
        else np.nan
    )

    rows.append(
        {
            "current_success_threshold": threshold,
            "gateways_flagged": n_selected,
            "future_failures_captured": captured,
            "recall": recall,
            "precision": precision,
        }
    )


threshold_df = pd.DataFrame(
    rows
)

print()
print("=" * 70)
print("SIMPLE EARLY-WARNING THRESHOLD TEST")
print("=" * 70)

print(
    threshold_df.round(3).to_string(
        index=False
    )
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = (
    OUTPUT_DIR
    / "early_warning_target.csv"
)

usable.to_csv(
    output_path,
    index=False,
)

threshold_path = (
    OUTPUT_DIR
    / "early_warning_threshold_test.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False,
)

print()
print("=" * 70)
print("OUTPUTS CREATED")
print("=" * 70)

print(output_path)

print(threshold_path)

print()
print("DONE")