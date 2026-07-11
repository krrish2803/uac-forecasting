# Executive Summary: UAC Care Load & Placement Demand Forecasting

**Prepared for:** HHS Program Decision-Makers | **Date:** July 2026

---

## The Big Picture

We built a forecasting system that predicts how many children will be in HHS care and how many will be discharged each day, using 3 years of daily data (Jan 2023 – Dec 2025). The system uses machine learning models that outperform traditional forecasting methods by up to 66%.

## Key Numbers

| Metric | Value |
|--------|-------|
| **Current Care Load** | ~2,470 children in HHS care |
| **Forecast Accuracy** | 99.7% (1-day ahead) |
| **Surge Lead Time** | 7 days — the system reliably predicts capacity changes a full week in advance |
| **Best Model for Care Load** | Random Forest (MAE: 6.1 children) |
| **Best Model for Discharges** | Gradient Boosting (MAE: 1.9 children) |

## What the Data Shows

- Care load has dropped 79% from its peak (11,500 in Jan 2023 → ~2,470 now)
- Discharges run about 181 children per day on average
- **Weekend discharges are 18% higher** than weekdays — plan staffing accordingly
- A "net pressure signal" (children transferred in vs. discharged) serves as an early warning for capacity stress

## Capacity Breach Risk

The dashboard's "Capacity Breach Probability" is calculated against a default threshold of **11,516 children** — the historical peak observed in December 2023. This threshold is adjustable in the sidebar to match your operational capacity limits. With current trends, the probability of exceeding this threshold within the next 14 days is near zero. However, any increase in intake volume (e.g., +25% scenario) would raise breach probability significantly. The dashboard allows you to model "what-if" scenarios instantly.

## Recommendations

1. **Use the 7-day care load forecast** for weekly staffing and resource allocation decisions
2. **Monitor the net pressure indicator** — when it turns red (positive), capacity stress is imminent
3. **Run scenario analyses** before policy changes to anticipate downstream care load impact
4. **Re-train models weekly** to maintain accuracy as trends evolve — given the 79% structural decline in the series, weekly retraining ensures the model adapts to the current regime rather than learning from stale historical patterns

## How to Use This System

The interactive dashboard (`streamlit run app/streamlit_app.py`) provides:
- Real-time care load and discharge forecasts with confidence bands
- Side-by-side model comparison
- Scenario builder — adjust intake volume and see the impact on future care load
- KPI cards updated daily

---

*For technical details, see the full research paper (`research_paper.md`).*
