"""Identify statistical anomalies in click/report behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze_events import compute_kpi_frames, load_clean_data


def flag_zscore_anomalies(
    df: pd.DataFrame,
    metric: str,
    group_by: Iterable[str],
    window: int = 7,
    z_threshold: float = 2.5,
    pct_threshold: float = 0.15,
) -> pd.DataFrame:
    flagged_frames: list[pd.DataFrame] = []
    for _, group in df.groupby(list(group_by), dropna=False):
        sorted_group = group.sort_values("date").copy()
        sorted_group["rolling_mean"] = sorted_group[metric].rolling(window, min_periods=3).mean()
        sorted_group["rolling_std"] = sorted_group[metric].rolling(window, min_periods=3).std()
        sorted_group["z_score"] = (sorted_group[metric] - sorted_group["rolling_mean"]) / sorted_group["rolling_std"]
        sorted_group["pct_change"] = sorted_group[metric].pct_change()
        mask = (
            sorted_group["rolling_std"].gt(0)
            & sorted_group["z_score"].abs().ge(z_threshold)
            & sorted_group["pct_change"].abs().ge(pct_threshold)
        )
        if not mask.any():
            continue
        block = sorted_group.loc[mask].copy()
        key_values = sorted_group.iloc[0][list(group_by)]
        descriptor = "|".join(f"{col}={key_values[col]}" for col in list(group_by))
        block["metric"] = metric
        block["anomaly_direction"] = np.where(block["pct_change"] >= 0, "spike", "drop")
        block["group"] = descriptor
        flagged_frames.append(block)
    if not flagged_frames:
        columns = df.columns.tolist() + ["metric", "anomaly_direction", "group"]
        return pd.DataFrame(columns=columns)
    return pd.concat(flagged_frames, ignore_index=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Detect anomalies in daily KPIs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/clean_phishing.csv"),
        help="Cleaned data produced by preprocess_data.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/anomalies.csv"),
        help="CSV to persist flagged anomalies.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Rolling window length (days) used to compute baselines.",
    )
    args = parser.parse_args()
    clean_df = load_clean_data(args.input)
    frames = compute_kpi_frames(clean_df)
    dept_anomalies = flag_zscore_anomalies(
        frames["department_daily"], "click_rate", ["department"], window=args.window
    )
    region_anomalies = flag_zscore_anomalies(
        frames["region_daily"], "report_rate", ["region"], window=args.window
    )
    combined = pd.concat([dept_anomalies, region_anomalies], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    if combined.empty:
        print("No anomalies detected in the current window.")
    else:
        print(f"Flagged {len(combined)} anomalies saved to {args.output}")


if __name__ == "__main__":
    main()
