from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd


# ============================================================
# DATA LOADER
# ============================================================


class ChallengeData:
    """
    Loads and prepares the challenge data directly from data/.

    Design:
    - Meter-read success provides the business-performance signal.
    - Telemetry provides the technical-health signal.
    - Telemetry-only weeks are retained.
    - Latest known meter performance is carried forward when
      a new meter observation is unavailable.
    - Meter-derived trend features are NOT invented for
      telemetry-only weeks.
    """

    TELEMETRY_COLUMNS: ClassVar[list[str]] = [
        "gateway_id",
        "ts_utc",
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
        "reboot_duration_sec",
    ]

    def __init__(
        self,
        data_dir: str | Path = "data",
    ):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        self.telemetry_dir = (
            self.data_dir / "telemetry"
        )

    # ========================================================
    # GATEWAY ID NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_gateway_id(
        series: pd.Series,
    ) -> pd.Series:
        """
        Convert accepted gateway formats into one format.

        Example:
            06:39:EA:56:02:C1
        becomes:
            0639EA5602C1
        """

        return (
            series.astype(str)
            .str.replace(
                ":",
                "",
                regex=False,
            )
            .str.strip()
            .str.upper()
        )

    # ========================================================
    # LOAD METER READ SUCCESS
    # ========================================================

    def load_meter_read_success(
        self,
    ) -> pd.DataFrame:
        """
        Load weekly meter-read performance.

        Adds:
            success_rate
            gateway_norm
        """

        path = (
            self.data_dir
            / "meter_read_success.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_csv(
            path,
            encoding="cp1252",
        )

        required = [
            "week_start",
            "gateway_id",
            "meters_expected",
            "meters_read",
        ]

        self._check_columns(
            df,
            required,
            "meter_read_success.csv",
        )

        df["week_start"] = pd.to_datetime(
            df["week_start"],
            errors="coerce",
        )

        if df["week_start"].isna().any():
            raise ValueError(
                "meter_read_success.csv contains "
                "invalid week_start values."
            )

        if (
            df["meters_expected"] <= 0
        ).any():
            raise ValueError(
                "meters_expected must be greater than zero."
            )

        if (
            df["meters_read"] < 0
        ).any():
            raise ValueError(
                "meters_read cannot be negative."
            )

        df["success_rate"] = (
            df["meters_read"]
            / df["meters_expected"]
        )

        invalid_rate = (
            (df["success_rate"] < 0)
            |
            (df["success_rate"] > 1)
        )

        if invalid_rate.any():
            raise ValueError(
                "success_rate contains values outside 0..1."
            )

        df["gateway_norm"] = (
            self.normalize_gateway_id(
                df["gateway_id"]
            )
        )

        return df

    # ========================================================
    # LOAD GATEWAY MASTER
    # ========================================================

    def load_gateway_master(
        self,
    ) -> pd.DataFrame:
        """
        Load gateway asset information.
        """

        path = (
            self.data_dir
            / "gateway_master.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_csv(
            path,
            encoding="cp1252",
        )

        required = [
            "gateway_id",
            "tenant",
            "site_type",
            "region",
            "hw_model",
            "antenna_type",
            "fw_version",
            "fw_updated_on",
            "installed_on",
            "decommissioned_on",
            "n_meters_installed",
        ]

        self._check_columns(
            df,
            required,
            "gateway_master.csv",
        )

        df["gateway_norm"] = (
            self.normalize_gateway_id(
                df["gateway_id"]
            )
        )

        df["decommissioned_on"] = pd.to_datetime(
            df["decommissioned_on"],
            errors="coerce",
        )

        return df

    # ========================================================
    # LOAD FIELD VISITS
    # ========================================================

    def load_field_visits(
        self,
    ) -> pd.DataFrame:
        """
        Load historical technician visits.
        """

        path = (
            self.data_dir
            / "field_visits.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_csv(
            path,
            encoding="cp1252",
        )

        required = [
            "visit_id",
            "gateway_id",
            "requested_on",
            "visited_on",
            "reason_reported",
            "outcome",
            "parts_replaced",
            "technician_hours",
        ]

        self._check_columns(
            df,
            required,
            "field_visits.csv",
        )

        df["requested_on"] = pd.to_datetime(
            df["requested_on"],
            errors="coerce",
        )

        df["visited_on"] = pd.to_datetime(
            df["visited_on"],
            errors="coerce",
        )

        df["gateway_norm"] = (
            self.normalize_gateway_id(
                df["gateway_id"]
            )
        )

        return df

    # ========================================================
    # LOAD ENGINEER REVIEW
    # ========================================================

    def load_engineer_review(
        self,
    ) -> pd.DataFrame:
        """
        Load the engineer review spreadsheet.

        This is available for analysis and validation.
        It is not required by the final prediction pipeline.
        """

        path = (
            self.data_dir
            / "engineer_review_2026-02.xlsx"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        df = pd.read_excel(path)

        required = [
            "gateway_id",
            "standort",
            "Kategorie",
            "reviewed_on",
            "reviewer",
            "Bemerkung",
        ]

        self._check_columns(
            df,
            required,
            "engineer_review_2026-02.xlsx",
        )

        df["reviewed_on"] = pd.to_datetime(
            df["reviewed_on"],
            errors="coerce",
        )

        df["gateway_norm"] = (
            self.normalize_gateway_id(
                df["gateway_id"]
            )
        )

        return df

    # ========================================================
    # LOAD TELEMETRY
    # ========================================================

    def load_telemetry_weekly(
        self,
    ) -> pd.DataFrame:
        """
        Read monthly telemetry Parquet files and aggregate
        them to gateway-week level.

        offline_duration_sec:
            Firmware-reported counter/state, therefore use
            the final observed value in the week.

        Event counters:
            Sum across the week.
        """

        if not self.telemetry_dir.exists():
            raise FileNotFoundError(
                "Missing telemetry directory: "
                f"{self.telemetry_dir}"
            )

        files = sorted(
            self.telemetry_dir.glob(
                "month=*/part-*.parquet"
            )
        )

        if not files:
            raise FileNotFoundError(
                "No telemetry Parquet files found."
            )

        weekly_frames = []

        for file in files:

            print(
                "Loading telemetry: "
                f"{file.parent.name}"
            )

            df = pd.read_parquet(
                file,
                columns=self.TELEMETRY_COLUMNS,
            )

            df["ts"] = pd.to_datetime(
                df["ts_utc"],
                utc=True,
                errors="coerce",
            )

            if df["ts"].isna().any():
                raise ValueError(
                    f"Invalid timestamps found in {file}"
                )

            df["gateway_norm"] = (
                self.normalize_gateway_id(
                    df["gateway_id"]
                )
            )

            # Convert UTC timestamps to timezone-naive
            # timestamps for consistent calendar-week grouping.
            ts = (
                df["ts"]
                .dt
                .tz_convert(None)
            )

            # Monday-based calendar week.
            df["week_start"] = (
                ts
                -
                pd.to_timedelta(
                    ts.dt.weekday,
                    unit="D",
                )
            ).dt.normalize()

            # Sort before taking the final weekly
            # offline-duration value.
            df = df.sort_values(
                [
                    "gateway_norm",
                    "week_start",
                    "ts",
                ]
            )

            weekly = (
                df
                .groupby(
                    [
                        "gateway_norm",
                        "week_start",
                    ]
                )
                .agg(
                    offline_duration_sec=(
                        "offline_duration_sec",
                        "last",
                    ),
                    disconnection_cnt=(
                        "disconnection_cnt",
                        "sum",
                    ),
                    reboot_cnt=(
                        "reboot_cnt",
                        "sum",
                    ),
                    reboot_duration_sec=(
                        "reboot_duration_sec",
                        "sum",
                    ),
                )
                .reset_index()
            )

            weekly_frames.append(
                weekly
            )

        telemetry = pd.concat(
            weekly_frames,
            ignore_index=True,
        )

        # Safety aggregation in case the same
        # gateway-week appears more than once.
        telemetry = (
            telemetry
            .groupby(
                [
                    "gateway_norm",
                    "week_start",
                ],
                as_index=False,
            )
            .agg(
                offline_duration_sec=(
                    "offline_duration_sec",
                    "last",
                ),
                disconnection_cnt=(
                    "disconnection_cnt",
                    "sum",
                ),
                reboot_cnt=(
                    "reboot_cnt",
                    "sum",
                ),
                reboot_duration_sec=(
                    "reboot_duration_sec",
                    "sum",
                ),
            )
        )

        return telemetry

    # ========================================================
    # BUILD WEEKLY DATASET
    # ========================================================

    def build_weekly_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Build the complete gateway-week dataset.

        Key design choice:

        Use an OUTER merge so telemetry-only weeks are retained.

        Meter-derived features are based only on genuinely
        observed meter-read weeks. When meter observations stop,
        the latest known business performance is carried forward,
        but new meter trends are NOT invented.
        """

        meter = (
            self.load_meter_read_success()
        )

        master = (
            self.load_gateway_master()
        )

        telemetry = (
            self.load_telemetry_weekly()
        )

        master_small = master[
            [
                "gateway_norm",
                "tenant",
                "site_type",
                "region",
                "hw_model",
                "antenna_type",
                "fw_version",
                "n_meters_installed",
                "decommissioned_on",
            ]
        ].copy()

        # ----------------------------------------------------
        # Meter data
        # ----------------------------------------------------

        meter_small = meter[
            [
                "gateway_norm",
                "week_start",
                "meters_expected",
                "meters_read",
                "success_rate",
            ]
        ].copy()

        # ----------------------------------------------------
        # Combine all gateway-weeks
        # ----------------------------------------------------

        weekly = telemetry.merge(
            meter_small,
            on=[
                "gateway_norm",
                "week_start",
            ],
            how="outer",
        )

        # ----------------------------------------------------
        # Add gateway master information
        # ----------------------------------------------------

        weekly = weekly.merge(
            master_small,
            on="gateway_norm",
            how="left",
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        weekly = (
            weekly
            .sort_values(
                [
                    "gateway_norm",
                    "week_start",
                ]
            )
            .reset_index(drop=True)
        )

        # ====================================================
        # METER DATA FRESHNESS
        # ====================================================

        # True only when a real meter-read observation exists
        # for THIS exact gateway-week.
        weekly["meter_observed"] = (
            weekly["success_rate"].notna()
        )

        # Last genuinely observed meter week per gateway.
        last_meter_by_gateway = (
            weekly.loc[
                weekly["meter_observed"],
                [
                    "gateway_norm",
                    "week_start",
                ],
            ]
            .groupby(
                "gateway_norm"
            )["week_start"]
            .max()
        )

        weekly["last_meter_week"] = (
            weekly["gateway_norm"]
            .map(
                last_meter_by_gateway
            )
        )

        # Number of weeks since the last genuine
        # meter observation.
        weekly["meter_age_weeks"] = (
            (
                weekly["week_start"]
                - weekly["last_meter_week"]
            )
            .dt.days
            / 7
        ).round()

        # ====================================================
        # CARRY FORWARD LATEST KNOWN BUSINESS PERFORMANCE
        # ====================================================

        weekly["success_rate"] = (
            weekly
            .groupby(
                "gateway_norm"
            )["success_rate"]
            .ffill()
        )

        # Gateways with no meter history receive the fleet
        # median as a neutral business baseline.
        fleet_median_success = (
            meter["success_rate"].median()
        )

        weekly["success_rate"] = (
            weekly["success_rate"]
            .fillna(
                fleet_median_success
            )
        )

        # ====================================================
        # METER-DERIVED TREND
        # ====================================================

        # IMPORTANT:
        # We calculate trend ONLY BETWEEN ACTUAL METER
        # OBSERVATIONS.
        #
        # Therefore:
        #
        # Jan actual -> Jan actual:
        #     real success_change
        #
        # Feb telemetry-only week:
        #     success_change = NaN
        #
        # Mar telemetry-only week:
        #     success_change = NaN
        #
        # This prevents stale meter data from pretending
        # that a new business deterioration was observed.

        observed_meter = weekly[
            weekly["meter_observed"]
        ][
            [
                "gateway_norm",
                "week_start",
                "success_rate",
            ]
        ].copy()

        observed_meter = (
            observed_meter
            .sort_values(
                [
                    "gateway_norm",
                    "week_start",
                ]
            )
        )

        observed_meter[
            "previous_observed_success"
        ] = (
            observed_meter
            .groupby(
                "gateway_norm"
            )["success_rate"]
            .shift(1)
        )

        observed_meter[
            "observed_success_change"
        ] = (
            observed_meter["success_rate"]
            -
            observed_meter[
                "previous_observed_success"
            ]
        )

        observed_changes = observed_meter[
            [
                "gateway_norm",
                "week_start",
                "previous_observed_success",
                "observed_success_change",
            ]
        ].copy()

        weekly = weekly.merge(
            observed_changes,
            on=[
                "gateway_norm",
                "week_start",
            ],
            how="left",
        )

        weekly["previous_success_rate"] = (
            weekly[
                "previous_observed_success"
            ]
        )

        weekly["success_change"] = (
            weekly[
                "observed_success_change"
            ]
        )

        # Remove temporary columns.
        weekly = weekly.drop(
            columns=[
                "previous_observed_success",
                "observed_success_change",
            ]
        )

        # ====================================================
        # TELEMETRY CHANGES
        # ====================================================

        grouped = weekly.groupby(
            "gateway_norm",
            group_keys=False,
        )

        weekly["previous_offline"] = (
            grouped[
                "offline_duration_sec"
            ].shift(1)
        )

        weekly["previous_disconnections"] = (
            grouped[
                "disconnection_cnt"
            ].shift(1)
        )

        weekly["previous_reboots"] = (
            grouped[
                "reboot_cnt"
            ].shift(1)
        )

        weekly["offline_change"] = (
            weekly["offline_duration_sec"]
            -
            weekly["previous_offline"]
        )

        weekly["disconnection_change"] = (
            weekly["disconnection_cnt"]
            -
            weekly["previous_disconnections"]
        )

        weekly["reboot_change"] = (
            weekly["reboot_cnt"]
            -
            weekly["previous_reboots"]
        )

        # ====================================================
        # LOW-SUCCESS PERSISTENCE
        # ====================================================

        # Persistence is calculated ONLY from actual meter
        # observations.
        #
        # Later telemetry-only weeks carry forward the last
        # genuinely observed persistence value.

        observed_for_persistence = weekly[
            weekly["meter_observed"]
        ][
            [
                "gateway_norm",
                "week_start",
                "success_rate",
            ]
        ].copy()

        observed_for_persistence = (
            observed_for_persistence
            .sort_values(
                [
                    "gateway_norm",
                    "week_start",
                ]
            )
        )

        persistence_rows = []

        for (
            gateway,
            gateway_df,
        ) in observed_for_persistence.groupby(
            "gateway_norm"
        ):

            streak = 0

            for _, row in gateway_df.iterrows():

                if row["success_rate"] < 0.80:
                    streak += 1
                else:
                    streak = 0

                persistence_rows.append(
                    {
                        "gateway_norm":
                            gateway,

                        "week_start":
                            row["week_start"],

                        "low_success_streak":
                            streak,
                    }
                )

        persistence_df = pd.DataFrame(
            persistence_rows,
            columns=[
                "gateway_norm",
                "week_start",
                "low_success_streak",
            ],
        )

        weekly = weekly.merge(
            persistence_df,
            on=[
                "gateway_norm",
                "week_start",
            ],
            how="left",
        )

        # Carry the latest genuine persistence value into
        # later telemetry-only weeks.
        weekly["low_success_streak"] = (
            weekly
            .groupby(
                "gateway_norm"
            )["low_success_streak"]
            .ffill()
            .fillna(0)
        )

        # ====================================================
        # CLEAN NUMERIC COLUMNS
        # ====================================================

        numeric_columns = [
            "offline_duration_sec",
            "disconnection_cnt",
            "reboot_cnt",
            "reboot_duration_sec",
            "meters_expected",
            "meters_read",
            "success_rate",
            "previous_success_rate",
            "previous_offline",
            "previous_disconnections",
            "previous_reboots",
            "success_change",
            "offline_change",
            "disconnection_change",
            "reboot_change",
            "low_success_streak",
            "meter_age_weeks",
        ]

        for column in numeric_columns:

            if column in weekly.columns:

                weekly[column] = pd.to_numeric(
                    weekly[column],
                    errors="coerce",
                )

        return weekly

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    @staticmethod
    def _check_columns(
        df: pd.DataFrame,
        required: list[str],
        file_name: str,
    ) -> None:

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{file_name} is missing column(s): "
                f"{', '.join(missing)}"
            )