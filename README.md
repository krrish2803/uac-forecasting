# HHS UAC Program: Predictive Forecasting Dashboard

A complete forecasting pipeline for the HHS Unaccompanied Alien Children (UAC) Program, predicting daily **Children in HHS Care** and **Discharged from HHS** volumes. Includes 6 models, interactive Streamlit dashboard, and full research paper.

**Live Dashboard:** [uac-forecasting.streamlit.app](https://uac-forecasting.streamlit.app)

## Project Overview

| Metric | Value |
|--------|-------|
| Data Period | Jan 2023 - Dec 2025 |
| Daily Observations | 1,061 |
| Engineered Features | 46 |
| Models Compared | 6 |
| Best Care Load Model | Random Forest (MAE: 6.07) |
| Best Discharge Model | Gradient Boosting (MAE: 1.88) |

## Key Findings

- **Machine learning outperforms statistical methods** -- RF beats SARIMA by 66% on care load forecasting
- **Discharge rates increase 17.8% on weekends** -- batch processing and family reunification schedules
- **Net pressure signal** (transfers out minus discharges) is a strong early warning indicator
- **Exponential smoothing fails** due to structural breaks in the 79% declining series

## Project Structure

```
uac_forecasting/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
├── src/
│   ├── preprocessing.py          # Data cleaning, interpolation, source mask
│   ├── features.py               # 46-feature engineering pipeline
│   ├── models.py                 # 6 models x 2 targets, walk-forward eval
│   └── evaluate.py               # Residual diagnostics, feature importance
├── data/
│   ├── HHS_Unaccompanied_Alien_Children_Program.csv   # Raw data
│   ├── processed_features.csv    # Cleaned 1061x46 feature matrix
│   └── interpolation_source.csv  # Reported vs interpolated mask
├── outputs/
│   ├── models/                   # Trained .joblib models
│   ├── forecasts/                # Model comparison + forecast CSVs
│   ├── plots/                    # Feature importance, CI, residual plots
│   └── Research_Paper_HHS_UAC_Forecasting.pdf
├── reports/
│   └── evaluation_report.md
├── research_paper.md
├── executive_summary.md
├── requirements.txt
├── setup.sh
└── generate_pdf.py
```

## Models

| Model | Description | in_hhs MAE | discharged MAE |
|-------|-------------|------------|----------------|
| Naive Persistence | Yesterday = today | 6.36 | 2.00 |
| Moving Average (7d) | 7-day rolling average | 42.21 | 2.89 |
| SARIMA | Auto-selected, m=7 | 17.86 | 3.77 |
| Exp. Smoothing | Holt-Winters | 108.95 | 4.68 |
| **Random Forest** | 200 trees, lag features | **6.07** | 2.09 |
| **Gradient Boosting** | 200 estimators | 9.82 | **1.88** |

## Setup

```bash
# Clone
git clone https://github.com/krrish2803/uac-forecasting.git
cd uac-forecasting

# Install
pip install -r requirements.txt

# Run dashboard
streamlit run app/streamlit_app.py
```

## Dashboard Features

- **Care Load Forecast** -- 1-30 day horizon with confidence intervals
- **Discharge Demand Forecast** -- model comparison with CI bands
- **Scenario Builder** -- adjust intake volume (-50% to +100%), view capacity breach probability
- **Model Comparison** -- side-by-side MAE/RMSE across all models
- **KPI Cards** -- current values, surge lead time, forecast stability
- **Capacity Threshold** -- adjustable, defaults to historical peak (11,516)

## Data

Source: HHS Office of Refugee Resettlement daily UAC Program reporting.

**Note:** 33% of calendar days (355/1075) had missing values reconstructed via linear interpolation. A source mask (`interpolation_source.csv`) tracks which days are reported vs interpolated.

## Reports

- [Research Paper](research_paper.md) -- full methodology, results, diagnostics
- [Executive Summary](executive_summary.md) -- 1-page operational briefing
- [Evaluation Report](reports/evaluation_report.md) -- model performance details
- [PDF Version](outputs/Research_Paper_HHS_UAC_Forecasting.pdf)

## Author

Krish Yad Unified Mentor Project -- July 2026
