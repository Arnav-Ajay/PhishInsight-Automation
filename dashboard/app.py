"""Streamlit dashboard for PhishInsight Automation (PoC)."""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PAGE_DESC = (
    "PhishInsight Automation (PoC) showcases synthetic phishing simulation KPI tracking, "
    "anomaly detection, and campaign storytelling tailored for enterprise security teams."
)

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(relative: str, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    path = ROOT / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or [])


@st.cache_data(show_spinner=False)
def load_datasets() -> dict[str, pd.DataFrame]:
    data = {
        "overall": _read_csv("data/analytics/overall_daily.csv", parse_dates=["date"]),
        "department_daily": _read_csv("data/analytics/department_daily.csv", parse_dates=["date"]),
        "region_daily": _read_csv("data/analytics/region_daily.csv", parse_dates=["date"]),
        "department_overall": _read_csv("data/analytics/department_overall.csv"),
        "region_overall": _read_csv("data/analytics/region_overall.csv"),
        "anomalies": _read_csv("data/anomalies.csv", parse_dates=["date"]),
    }
    return data


def _run_pipeline() -> None:
    scripts = [
        ROOT / "scripts" / "generate_data.py",
        ROOT / "scripts" / "preprocess_data.py",
        ROOT / "scripts" / "analyze_events.py",
        ROOT / "scripts" / "detect_anomalies.py",
    ]
    for script in scripts:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def _filter_by_selection(
    df: pd.DataFrame, date_window: tuple[date, date], departments: list[str], regions: list[str]
) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    start, end = date_window
    filtered = filtered[(filtered["date"] >= pd.Timestamp(start)) & (filtered["date"] <= pd.Timestamp(end))]
    if "department" in filtered.columns and departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if "region" in filtered.columns and regions:
        filtered = filtered[filtered["region"].isin(regions)]
    return filtered


def _format_delta(current: float, baseline: float, as_percent: bool = True) -> str:
    if baseline == 0 or pd.isna(baseline):
        return ""
    diff = current - baseline
    if as_percent:
        return f"{diff:+.1%}"
    return f"{diff:+.2f}"


def _build_trend_chart(overall: pd.DataFrame, anomalies: pd.DataFrame, date_window: tuple[date, date]) -> go.Figure:
    if overall.empty:
        return go.Figure()
    start, end = date_window
    filtered = overall[
        (overall["date"] >= pd.Timestamp(start))
        & (overall["date"] <= pd.Timestamp(end))
    ]
    fig = px.line(
        filtered,
        x="date",
        y=["click_rate", "report_rate"],
        labels={"value": "Rate", "variable": "Metric"},
        markers=True,
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(legend_title_text="")
    if not anomalies.empty:
        window_anomalies = anomalies[
            (anomalies["date"] >= pd.Timestamp(start))
            & (anomalies["date"] <= pd.Timestamp(end))
        ]
        for metric_name in ["click_rate", "report_rate"]:
            subset = window_anomalies[window_anomalies["metric"] == metric_name]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset[metric_name],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond", line=dict(width=1, color="DarkSlateGrey")),
                    name=f"{metric_name.replace('_', ' ').title()} Anomaly",
                    hovertemplate="%{x|%b %d}: %{y:.1%}<br>%{customdata[0]}<br>Δ %{customdata[1]:.1%}",
                    customdata=subset[["group", "pct_change"]].to_numpy(),
                )
            )
    fig.update_yaxes(tickformat=".0%")
    return fig


def _show_anomaly_panel(anomalies: pd.DataFrame) -> None:
    st.subheader("Flagged anomalies")
    if anomalies.empty:
        st.info("No anomalies detected in the current window.")
        return
    st.dataframe(
        anomalies[
            ["date", "group", "metric", "anomaly_direction", "z_score", "pct_change"]
        ].rename(columns={"group": "scope"}),
        height=220,
    )
    csv = anomalies.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download anomalies",
        data=csv,
        file_name="phishinsight_anomalies.csv",
        mime="text/csv",
    )


def _show_department_board(dept_overall: pd.DataFrame, selected_departments: list[str]) -> None:
    st.subheader("Department performance")
    if dept_overall.empty:
        st.write("Department breakdown will appear once analytics run.")
        return
    df = dept_overall.copy()
    if selected_departments:
        df = df[df["department"].isin(selected_departments)]
    df = df.sort_values("phishrisk_index", ascending=False).head(6)
    st.table(df[["department", "click_rate", "report_rate", "phishrisk_index"]].assign(
        click_rate=lambda d: d["click_rate"].map("{:.1%}".format),
        report_rate=lambda d: d["report_rate"].map("{:.1%}".format),
        phishrisk_index=lambda d: d["phishrisk_index"].map("{:.2f}".format),
    ))


def _show_region_board(region_overall: pd.DataFrame, selected_regions: list[str]) -> None:
    st.subheader("Regional spotlight")
    if region_overall.empty:
        st.write("Regional breakdown will appear once analytics run.")
        return
    df = region_overall.copy()
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]
    fig = px.bar(
        df,
        x="region",
        y="phishrisk_index",
        color="region",
        title="PhishRisk Index by region",
        labels={"phishrisk_index": "PhishRisk"},
    )
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="PhishInsight Automation", layout="wide", page_icon="🛡️")
    st.title("PhishInsight Automation (PoC)")
    st.caption(PAGE_DESC)

    sidebar = st.sidebar
    refresh_requested = sidebar.button("Refresh data (run pipeline)")
    if refresh_requested:
        with st.spinner("Regenerating data..."):
            try:
                _run_pipeline()
                load_datasets.clear()
                st.success("Data refreshed — dashboard will show new metrics shortly.")
            except subprocess.CalledProcessError as exc:
                st.error(f"Pipeline failed: {exc}")
    data = load_datasets()
    overall = data["overall"]
    dept_daily = data["department_daily"]
    region_daily = data["region_daily"]
    dept_overall = data["department_overall"]
    region_overall = data["region_overall"]
    anomalies = data["anomalies"]

    if overall.empty:
        st.warning("Run `generate_data.py` and the preprocessing pipeline to seed the dashboard.")
        return

    min_date = overall["date"].min().date()
    max_date = overall["date"].max().date()
    sidebar.header("Filters & controls")
    selected_departments = sidebar.multiselect(
        "Department(s)",
        options=sorted(dept_daily["department"].unique()),
        default=list(sorted(dept_daily["department"].unique())),
    )
    selected_regions = sidebar.multiselect(
        "Region(s)",
        options=sorted(region_daily["region"].unique()),
        default=list(sorted(region_daily["region"].unique())),
    )
    date_range = sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range
    latest = overall.sort_values("date").iloc[-1]
    previous = overall.sort_values("date").iloc[-2] if len(overall) >= 2 else latest
    cols = st.columns(5)
    metrics = [
        ("Open rate", "open_rate"),
        ("Click rate", "click_rate"),
        ("Report rate", "report_rate"),
        ("False positive rate", "false_positive_rate"),
        ("PhishRisk index", "phishrisk_index"),
    ]
    for col, (label, key) in zip(cols, metrics):
        value = latest.get(key, 0)
        baseline = previous.get(key, 0)
        is_rate = key != "phishrisk_index"
        delta = _format_delta(value, baseline, as_percent=is_rate)
        formatted = f"{value:.1%}" if is_rate else f"{value:.2f}"
        col.metric(label, formatted, delta=delta)

    st.markdown("### Campaign trends")
    trend_fig = _build_trend_chart(overall, anomalies, (start_date, end_date))
    st.plotly_chart(trend_fig, width="stretch")

    st.markdown("### Drill-down insights")
    st.write("Department and regional performance with anomaly highlights.")
    left, right = st.columns([2, 1])
    with left:
        _show_department_board(dept_overall, selected_departments)
        _show_anomaly_panel(anomalies)
    with right:
        _show_region_board(region_overall, selected_regions)


if __name__ == "__main__":
    main()
