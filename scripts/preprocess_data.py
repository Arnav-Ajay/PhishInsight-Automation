"""Preprocess the synthetic phishing events for downstream analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_type"] = df["event_type"].astype(str).str.lower().str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["email_id", "employee_id", "event_type", "timestamp"])
    df["date"] = df["timestamp"].dt.floor("d")
    df["weekday"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour
    df["campaign_id"] = df["campaign_id"].fillna(df["timestamp"].dt.strftime("campaign-%Y%U"))
    df["event_order"] = (
        df.groupby(["email_id", "employee_id"])["timestamp"].rank(method="dense").astype(int)
    )
    df["region"] = df["region"].str.upper()
    return df


def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Clean the synthetic phishing CSV for analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/synthetic_phishing.csv"),
        help="Raw synthetic events CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clean_phishing.csv"),
        help="Path for the cleaned CSV.",
    )
    args = parser.parse_args()
    df = load_raw_data(args.input)
    clean_df = preprocess(df)
    save_clean_data(clean_df, args.output)
    print(f"Cleaned {len(clean_df)} rows (deduped) written to {args.output}")


if __name__ == "__main__":
    main()
