# PhishInsight Automation (PoC)

## Project Overview

PhishInsight Automation (PoC) simulates the analytics lifecycle of an enterprise phishing awareness program: synthetic phishing events are generated, processed, and aggregated, then visualized through a recruiter-friendly Streamlit dashboard. The PoC highlights campaign KPIs, department and region performance, and anomaly markers that mimic real-world RBC phishing insights.

## Architecture Highlights

- **Data Flow:** `data/synthetic_phishing.csv` -> `scripts/preprocess_data.py` -> KPI aggregation + anomaly detection -> `dashboard/app.py` (Streamlit/Plotly).  
- **Automation Hooks:** Streamlit buttons trigger regeneration; optional Azure Functions + Blob Storage can refresh data automatically.  
- **Metrics:** Open rate, click rate, report rate, false positive rate, and PhishRisk Index; anomalies surfaced via z-score / rolling baseline.

## Folder Structure

- `data/` (synthetic CSV snapshots)  
- `scripts/` (ETL and analytics modules)  
- `dashboard/` (Streamlit)  
- `README.md` (project overview and deployment notes)