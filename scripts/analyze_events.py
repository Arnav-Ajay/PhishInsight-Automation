"""Aggregate KPIs from the cleaned phishing event log."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EVENT_TYPES = ["sent", "opened", "clicked", "reported", "ignored"]


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.floor("d")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    for event in EVENT_TYPES:
        if event not in df:
            df[event] = 0
    return df


def _group_event_counts(df: pd.DataFrame, group_by: Iterable[str]) -> pd.DataFrame:
    counts = (
        df.groupby(list(group_by) + ["event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    counts = _ensure_event_columns(counts)
    return counts


def _compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["open_rate"] = result["opened"] / result["sent"].replace(0, np.nan)
    result["click_rate"] = result["clicked"] / result["opened"].replace(0, np.nan)
    result["report_rate"] = result["reported"] / result["clicked"].replace(0, np.nan)
    positive_actions = (
        result[["ignored", "clicked", "reported"]].sum(axis=1).replace(0, np.nan)
    )
    result["false_positive_rate"] = result["ignored"] / positive_actions
    result["phishrisk_index"] = (
        result["click_rate"].fillna(0) * 0.55
        + (0.3 - result["report_rate"].fillna(0)).clip(lower=0) * 0.45
    )
    result["campaign_size"] = result["sent"]
    rate_cols = ["open_rate", "click_rate", "report_rate", "false_positive_rate", "phishrisk_index"]
    result[rate_cols] = result[rate_cols].fillna(0).clip(lower=0)
    return result


def build_kpi_frame(df: pd.DataFrame, group_by: Iterable[str]) -> pd.DataFrame:
    frame = _group_event_counts(df, group_by)
    frame = _compute_rates(frame)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(list(group_by))


def compute_kpi_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result = {}
    result["overall_daily"] = build_kpi_frame(df, ["date"])
    result["department_daily"] = build_kpi_frame(df, ["department", "date"])
    result["region_daily"] = build_kpi_frame(df, ["region", "date"])
    result["department_overall"] = build_kpi_frame(df, ["department"])
    result["region_overall"] = build_kpi_frame(df, ["region"])
    result["campaign_summary"] = build_kpi_frame(df, ["campaign_id"])
    return result


def _persist_frames(frames: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        destination = output_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create KPI slices for the PoC dashboard.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clean_phishing.csv"),
        help="Cleaned dataset produced by preprocess_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analytics"),
        help="Destination folder for KPI CSVs.",
    )
    args = parser.parse_args()
    clean_df = load_clean_data(args.input)
    frames = compute_kpi_frames(clean_df)
    _persist_frames(frames, args.output_dir)
    print(f"Saved KPI slices to {args.output_dir}")


if __name__ == "__main__":
    main()
