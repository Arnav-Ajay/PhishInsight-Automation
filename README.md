# PhishInsight Automation (PoC)

Try It Live: https://phishinsight-automation.streamlit.app/

## 1. Executive Summary

PhishInsight Automation (PoC) is a demonstration of how a security program can generate synthetic phishing-response activity, compute critical awareness KPIs, detect anomalies, and communicate the findings through a polished Streamlit dashboard. It mirrors the workflow of enterprise phishing simulation programs by focusing on automation, analytics, and visual storytelling.

## 2. Project Flow & Components

- **Synthetic data layer:** `scripts/generate_data.py` produces 25k+ rows of phishing event activity with departments, regions, campaign IDs, subjects, and anomaly flags so the dataset captures realistic human behavior patterns.
- **Preprocessing:** `scripts/preprocess_data.py` normalizes text, parses timestamps, deduplicates rows, and derives `date`, `weekday`, `hour`, and `event_order` metadata to ensure reliable grouping.
- **Analytics:** `scripts/analyze_events.py` counts each funnel transition (`sent`, `opened`, `clicked`, `reported`, `ignored`), computes open/click/report/false-positive rates, and persists KPI slices (overall, department, region, campaign) for the dashboard filters.
- **Anomaly detection:** `scripts/detect_anomalies.py` applies rolling z-score analysis on click/report rates and writes anomalous rows to `data/anomalies.csv` for operational review.
- **Presentation:** `dashboard/app.py` (Streamlit + Plotly) surfaces KPI cards, trend charts with anomaly markers, department/regional breakdowns, and allows users to refresh the entire pipeline from the UI.

## 3. Folder Structure

```
phishinsight-automation/
├── data/                      # synthetic exports + cleaned + analytics
│   └── analytics/             # KPI tables consumed by the dashboard
├── scripts/                   # ETL + analytics + anomaly modules
├── dashboard/                 # Streamlit app
├── requirements.txt           # runtime dependencies
└── README.md                  # project overview & deployment notes
```

## 4. Getting Started (Local)

1. Pull/download this repository.
2. Install dependencies: `pip install -r requirements.txt`.  
3. Run the pipeline:
   ```bash
   python scripts/generate_data.py --rows 30000
   python scripts/preprocess_data.py
   python scripts/analyze_events.py
   python scripts/detect_anomalies.py
   ```
4. Launch the dashboard: `streamlit run dashboard/app.py`.  
5. Use the sidebar to filter by department/region/date and click “Refresh data (run pipeline)” to regenerate every stage (the UI clears caches and re-reads the newest CSVs).

## 5. Dashboard Highlights

- KPI banner with open, click, report, false-positive rates and the PhishRisk Index.  
- Trend chart that overlays click/report performance with anomaly diamonds (includes percent-change hover info).  
- Department ranking table that surfaces top PhishRisk departments for leadership review.  
- Regional PhishRisk bar chart for geo-level visibility.  
- Anomaly table + download button for compliance review and audit exports.

## 6. Automation & Observability

- “Refresh data (run pipeline)” in the sidebar shells out to `scripts/generate_data.py` → `scripts/preprocess_data.py` → `scripts/analyze_events.py` → `scripts/detect_anomalies.py`, then clears Streamlit cache so the UI reflects the latest CSV exports.  
- Everything is persisted under `data/` (raw, cleaned, analytics, anomalies), making the datasets available for downstream tools (Power BI, SQL, notebooks).
