"""Generate realistic synthetic phishing data for PhishInsight Automation PoC."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
from uuid import uuid4

DEPARTMENTS = [
    {
        "name": "Finance",
        "region": "NA",
        "send_weight": 0.22,
        "open_rate": 0.7,
        "click_rate": 0.22,
        "report_rate": 0.2,
        "false_positive_rate": 0.06,
    },
    {
        "name": "Operations",
        "region": "EMEA",
        "send_weight": 0.2,
        "open_rate": 0.65,
        "click_rate": 0.24,
        "report_rate": 0.14,
        "false_positive_rate": 0.04,
    },
    {
        "name": "IT",
        "region": "APAC",
        "send_weight": 0.18,
        "open_rate": 0.72,
        "click_rate": 0.3,
        "report_rate": 0.12,
        "false_positive_rate": 0.03,
    },
    {
        "name": "Sales",
        "region": "LATAM",
        "send_weight": 0.16,
        "open_rate": 0.6,
        "click_rate": 0.18,
        "report_rate": 0.1,
        "false_positive_rate": 0.05,
    },
    {
        "name": "People",
        "region": "NA",
        "send_weight": 0.12,
        "open_rate": 0.8,
        "click_rate": 0.15,
        "report_rate": 0.25,
        "false_positive_rate": 0.09,
    },
    {
        "name": "Security",
        "region": "EMEA",
        "send_weight": 0.12,
        "open_rate": 0.86,
        "click_rate": 0.09,
        "report_rate": 0.35,
        "false_positive_rate": 0.02,
    },
]

SUBJECT_TEMPLATES = [
    "Urgent: Expense report flagged for review",
    "Unusual sign-in detected – action required",
    "Your password expires today – update now",
    "Reimbursement approval pending",
    "Vendor invoice attached for verification",
    "Corporate travel itinerary changed",
    "New IT policy update needs acknowledgement",
    "Payroll information requires confirmation",
    "Pending security compliance task",
    "Verify device enrollment status",
]


def _ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _create_anomaly_schedule(
    start_date: datetime, days_span: int, rng: np.random.Generator, count: int = 5
) -> Dict[str, Dict[str, float]]:
    """Generate (department, date) pairs with probability adjustments to simulate spikes/drops."""
    anomalies: Dict[str, Dict[str, float]] = {}
    department_names = [dept["name"] for dept in DEPARTMENTS]
    for _ in range(count):
        dept = rng.choice(department_names)
        offset_days = int(rng.integers(0, max(1, days_span)))
        anomaly_date = (start_date + timedelta(days=offset_days)).date()
        key = f"{dept}::{anomaly_date.isoformat()}"
        if key in anomalies:
            continue
        anomalies[key] = {
            "click_delta": float(rng.uniform(0.15, 0.35)),
            "report_delta": float(rng.uniform(-0.25, -0.08)),
            "tag": rng.choice(["click_spike", "report_drop"]),
        }
    return anomalies


def _random_timestamp(base: datetime, scale_minutes: float, rng: np.random.Generator) -> datetime:
    """Sample a future timestamp offset from base using an exponential distribution."""
    offset_minutes = rng.exponential(scale=scale_minutes)
    return base + timedelta(minutes=int(offset_minutes))


def _build_record(
    email_id: str,
    employee_id: str,
    subject: str,
    profile: Dict,
    timestamp: datetime,
    event_type: str,
    anomaly_tag: str | None = None,
) -> Dict:
    return {
        "email_id": email_id,
        "employee_id": employee_id,
        "department": profile["name"],
        "region": profile["region"],
        "campaign_id": f"campaign-{timestamp.strftime('%Y%m')}",
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "subject": subject,
        "anomaly_tag": anomaly_tag or "",
    }


def generate_synthetic_data(num_events: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.utcnow() - timedelta(days=45)
    end = datetime.utcnow()
    days_span = (end - start).days
    anomalies = _create_anomaly_schedule(start, days_span, rng)
    weights = np.array([dept["send_weight"] for dept in DEPARTMENTS], dtype=float)
    weights /= weights.sum()

    records: List[Dict] = []
    template_choices = len(SUBJECT_TEMPLATES)
    total_minutes = max(1, int((end - start).total_seconds() / 60))

    for event_number in range(num_events):
        dept_idx = rng.choice(len(DEPARTMENTS), p=weights)
        profile = DEPARTMENTS[dept_idx]
        send_offset = rng.integers(0, total_minutes)
        send_time = start + timedelta(minutes=int(send_offset))
        date_key = send_time.date().isoformat()
        anomaly_key = f"{profile['name']}::{date_key}"
        anomaly = anomalies.get(anomaly_key)
        open_rate = profile["open_rate"]
        click_prob = profile["click_rate"]
        report_prob = profile["report_rate"]
        if anomaly:
            open_rate = min(1.0, open_rate + anomaly["click_delta"] / 2)
            click_prob = min(1.0, click_prob + anomaly["click_delta"])
            report_prob = max(0.02, report_prob + anomaly["report_delta"])

        email_id = str(uuid4())
        employee_id = f"user_{rng.integers(1, 6000):04d}"
        subject = SUBJECT_TEMPLATES[rng.integers(template_choices)]
        anomaly_tag = anomaly["tag"] if anomaly else ""

        records.append(
            _build_record(
                email_id,
                employee_id,
                subject,
                profile,
                send_time,
                "sent",
                anomaly_tag=anomaly_tag,
            )
        )

        opened = rng.random() < open_rate
        if opened:
            open_time = _random_timestamp(send_time, 90, rng)
            records.append(
                _build_record(
                    email_id,
                    employee_id,
                    subject,
                    profile,
                    open_time,
                    "opened",
                    anomaly_tag=anomaly_tag,
                )
            )
        else:
            open_time = send_time

        clicked = opened and (rng.random() < click_prob)
        if clicked:
            click_time = _random_timestamp(open_time, 30, rng)
            records.append(
                _build_record(
                    email_id,
                    employee_id,
                    subject,
                    profile,
                    click_time,
                    "clicked",
                    anomaly_tag=anomaly_tag,
                )
            )
        else:
            click_time = open_time

        reported = False
        if clicked:
            reported = rng.random() < report_prob
        else:
            reported = rng.random() < profile["false_positive_rate"]

        if reported:
            report_time = _random_timestamp(click_time, 20 + rng.uniform(0, 20), rng)
            records.append(
                _build_record(
                    email_id,
                    employee_id,
                    subject,
                    profile,
                    report_time,
                    "reported",
                    anomaly_tag=anomaly_tag,
                )
            )

        if not (opened or clicked or reported):
            ignored_time = _random_timestamp(send_time, 120, rng)
            records.append(
                _build_record(
                    email_id,
                    employee_id,
                    subject,
                    profile,
                    ignored_time,
                    "ignored",
                    anomaly_tag=anomaly_tag,
                )
            )

    df = pd.DataFrame.from_records(records)
    return df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create synthetic phishing response events for PhishInsight Automation PoC."
    )
    parser.add_argument("--rows", "-n", type=int, default=25000, help="Number of sent events to seed.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to make the dataset reproducible."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_phishing.csv"),
        help="Location to write the synthetic CSV.",
    )
    args = parser.parse_args()
    df = generate_synthetic_data(args.rows, seed=args.seed)
    _ensure_output_path(args.output)
    df.to_csv(args.output, index=False)
    summary = df["event_type"].value_counts(normalize=True).mul(100).round(1)
    print(f"Synthetic phishing dataset saved to {args.output}")
    print("Event mix (%):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
