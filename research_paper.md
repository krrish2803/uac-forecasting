# Predictive Forecasting of Care Load & Placement Demand: HHS Unaccompanied Alien Children Program

## Abstract

This study develops a predictive forecasting system for the Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program, targeting two critical operational metrics: daily children in HHS care and daily discharges from HHS care. Using 1,061 daily observations spanning January 2023 through December 2025, we engineered 46 predictive features including lag variables, rolling statistics, calendar effects, and a novel net pressure signal. We compared six forecasting approaches: naive persistence, 7-day moving average, SARIMA, exponential smoothing, random forest, and gradient boosting. Random forest achieved the best performance for care load forecasting (MAE: 6.07, RMSE: 7.72), while gradient boosting excelled at discharge prediction (MAE: 1.88, RMSE: 2.31). Results demonstrate that machine learning models leveraging engineered features outperform traditional statistical methods for this operational forecasting task.

## 1. Background

The HHS UAC Program manages care and placement of unaccompanied minors apprehended at U.S. borders. Daily operational decisions — staffing, facility capacity, resource allocation — depend on accurate forecasting of care load and discharge volumes. This project develops a data-driven forecasting pipeline to support these decisions.

### 1.1 Objectives

1. Build a complete daily forecasting system for "Children in HHS Care" and "Children Discharged from HHS Care"
2. Compare statistical and machine learning approaches
3. Develop operational KPIs: Surge Lead Time, Capacity Breach Probability, Forecast Stability
4. Create an interactive dashboard for scenario planning

## 2. Exploratory Data Analysis

### 2.1 Data Provenance

The dataset is sourced from the HHS Office of Refugee Resettlement's daily UAC Program reporting, published as a CSV file. It covers **January 12, 2023 through December 21, 2025** (1,075 calendar days). Reporting frequency is nominally daily, but 355 days (33.0%) have no reported values and were reconstructed via linear interpolation. The dataset includes five operational metrics: children apprehended and placed in CBP custody, children currently in CBP custody, children transferred out of CBP custody, children in HHS care, and children discharged from HHS care. We treat the data as a daily time series and do not adjust for reporting lag or backfill corrections.

### 2.2 Data Overview

- **Source:** HHS UAC Program daily reporting (CSV)
- **Period:** January 12, 2023 — December 21, 2025
- **Raw rows:** 720 (355 calendar days had missing/interpolated values)
- **Processed:** 1,061 daily observations (after forward-fill gap handling and interpolation)
- **Variables:** Children apprehended, in CBP custody, transferred out of CBP, in HHS care, discharged from HHS

### 2.2 Key EDA Findings

- **Dramatic decline in care load:** HHS care population dropped from a peak of ~11,500 (Jan 2023) to ~2,400 (Dec 2025), a 79% reduction
- **Apprehension volumes collapsed:** Daily apprehensions fell from 100-260+ (2023-2024) to 1-20 per day in late 2025
- **Seasonal patterns:** Summer months (June-August) consistently showed higher care loads in 2023-2024
- **Weekend effect:** Discharge rates average 203.1 on weekends vs 172.4 on weekdays (+17.8% on weekends)
- **Net pressure signal:** Transferred-out minus discharged shows capacity stress moments, particularly during January-February 2025 transitions

### 2.3 Time Series Decomposition

Seasonal decomposition (period=7) of the "Children in HHS Care" series reveals:
- **Trend:** Strong, consistent downward trend from 11,500 to 2,400
- **Seasonality:** Weekly pattern with modest amplitude (~50 children)
- **Residual:** Noise increases during transition periods (policy changes, seasonal surges)

## 3. Methodology

### 3.1 Feature Engineering

46 features were engineered across five categories:

| Category | Count | Examples |
|----------|-------|----------|
| Base numeric | 5 | apprehended, in_cbp, transferred_out, in_hhs, discharged |
| Lag features | 15 | t-1, t-7, t-14 for all 5 numeric columns |
| Rolling statistics | 20 | 7-day and 14-day rolling mean and std for all 5 columns |
| Calendar features | 5 | day_of_week, is_weekend, month, is_month_start, is_month_end |
| Pressure signal | 1 | net_pressure = transferred_out - discharged |

### 3.2 Train/Test Split

- **Training:** 2023-01-26 to 2025-12-07 (1,047 days)
- **Test:** 2025-12-08 to 2025-12-21 (14 days, held-out)

### 3.3 Models

1. **Naive Persistence:** Yesterday's value = today's forecast (baseline)
2. **Moving Average (7-day):** Rolling 7-day average
3. **SARIMA:** Auto-selected (pmdarima, seasonal=True, m=7)
4. **Exponential Smoothing:** Holt-Winters with additive trend and weekly seasonality
5. **Random Forest:** 200 trees on lag/rolling/calendar features
6. **Gradient Boosting:** 200 estimators with staged predictions for confidence intervals

### 3.4 Confidence Intervals

- **SARIMA / Exp. Smoothing:** Native analytical intervals
- **RF / GBR:** Bootstrap via tree predictions (RF) and staged prediction variance (GBR)

### 3.5 Capacity Threshold Assumption

The "Capacity Breach Probability" metric reported in the dashboard and executive summary uses a default threshold of **11,516 children** — the historical peak observed on December 20, 2023. This value represents the maximum care load recorded during the dataset period and serves as an illustrative upper bound. In practice, operational capacity thresholds should be set by HHS stakeholders based on shelter bed counts, staffing levels, and policy mandates. The dashboard allows users to adjust this threshold interactively.

### 3.6 Exponential Smoothing Debugging

The initial exponential smoothing model (additive trend, additive seasonal, m=7) produced an MAE of 120.48 — a severe outlier. A systematic search over 5 configurations (additive/multiplicative trend, additive/multiplicative/no seasonal, with/without Box-Cox transformation) was performed. The best-fitting model selected multiplicative seasonal with additive trend (AIC=8177.9), improving MAE to 108.95. However, this remains 10-18x worse than ML models. The root cause is that the HHS care load series underwent a structural break: a 79% decline from 11,500 to 2,400 over the training period. Exponential smoothing assumes a stable local level and trend, but the series' trajectory changed fundamentally due to policy-driven apprehension volume shifts. The model extrapolates a declining linear trend that overshoots the actual stabilization near 2,400, producing large systematic errors at the 14-day horizon.

## 4. Results

### 4.1 Model Comparison — Children in HHS Care

| Model | MAE (1-day) | MAE (7-day) | MAE (14-day) | MAE (overall) | RMSE |
|-------|-------------|-------------|---------------|---------------|------|
| Naive Persistence | 11.00 | 6.43 | 6.36 | 6.36 | 7.19 |
| Moving Average (7d) | 21.43 | 27.14 | 42.21 | 42.21 | 45.39 |
| SARIMA | 11.42 | 10.45 | 17.86 | 17.86 | 19.72 |
| Exp. Smoothing | 16.68 | 72.87 | 108.95 | 108.95 | 117.90 |
| **Random Forest** | **14.61** | **7.55** | **6.07** | **6.07** | **7.72** |
| Gradient Boosting | 5.21 | 12.30 | 9.82 | 9.82 | 12.83 |

**Winner:** Random Forest — lowest overall MAE (6.07), demonstrating that lag-based ML features capture the persistent care load dynamics better than pure time-series approaches.

### 4.2 Model Comparison — Discharged from HHS

| Model | MAE (1-day) | MAE (7-day) | MAE (14-day) | MAE (overall) | RMSE |
|-------|-------------|-------------|---------------|---------------|------|
| Naive Persistence | 8.00 | 2.29 | 2.00 | 2.00 | 3.01 |
| Moving Average (7d) | 6.57 | 2.43 | 2.89 | 2.89 | 3.39 |
| SARIMA | 9.64 | 3.57 | 3.77 | 3.77 | 4.45 |
| Exp. Smoothing | 10.44 | 5.56 | 4.68 | 4.68 | 5.32 |
| Random Forest | 5.72 | 2.96 | 2.09 | 2.09 | 2.60 |
| **Gradient Boosting** | **4.84** | **2.65** | **1.88** | **1.88** | **2.31** |

**Winner:** Gradient Boosting — lowest MAE (1.88), effectively modeling the nonlinear discharge patterns.

### 4.3 Feature Importance (Top 5)

**Random Forest (in_hhs):**
1. in_hhs_lag1 (t-1 value): dominant predictor
2. in_hhs_rmean7 (7-day rolling mean): captures trend
3. in_hhs_lag7 (t-7): weekly seasonality
4. in_hhs_rstd7 (7-day volatility): captures instability
5. net_pressure: operational signal

**Gradient Boosting (discharged):**
1. discharged_lag1: immediate autoregressive signal
2. discharged_rmean7: weekly average
3. day_of_week: captures weekend/weekday patterns
4. net_pressure: flow-through pressure
5. transferred_out_lag1: CBP transfer signal

### 4.4 Reported-Only vs Full Test Set MAE

To assess the impact of interpolated values on model evaluation, we recomputed MAE on only the 10 originally-reported days in the 14-day test set (29% of test days were interpolated).

| Model | in_hhs Full | in_hhs Reported-Only | discharged Full | discharged Reported-Only |
|-------|------------|---------------------|-----------------|------------------------|
| Naive Persistence | 6.36 | 6.43 | 2.00 | 2.53 |
| Moving Average (7d) | 42.21 | 41.03 | 2.89 | 2.84 |
| SARIMA | 17.86 | 18.07 | 3.77 | 3.98 |
| Exp. Smoothing | 108.95 | 102.95 | 4.68 | 5.12 |
| Random Forest | 6.07 | 6.35 | 2.09 | 2.19 |
| Gradient Boosting | 9.82 | 9.34 | 1.88 | 1.88 |

**Key finding:** Reported-only MAE tracks full-set MAE closely (within 5-10% for most models), confirming that interpolated days do not materially distort the model comparison. Gradient Boosting on discharged is identical (1.88) across both evaluation sets.

### 4.5 Residual Diagnostics

Residual plots were generated for the best model per target: Random Forest (in_hhs) and Gradient Boosting (discharged). See `outputs/plots/residuals_in_hhs.png` and `outputs/plots/residuals_discharged.png`.

- **in_hhs (Random Forest):** Residuals are tightly clustered around zero (IQR threshold: 7.45). One outlier detected on 2025-12-14 (residual=14.7, reported day) — this aligns with a holiday-period dip. No systematic clustering by interpolation status.
- **discharged (Gradient Boosting):** Four outliers detected (IQR threshold: 3.00): 2025-12-08 (-4.8, reported), 2025-12-10 (-3.7, reported), 2025-12-13 (-3.5, interpolated), 2025-12-14 (-3.2, reported). The cluster around Dec 8-14 suggests the model slightly overpredicts discharge volume during the early-to-mid December period, possibly reflecting seasonal slowdowns not fully captured by calendar features. Three of four outlier days are originally reported (not interpolated), ruling out interpolation artifact as the cause.

### 4.6 Feature Importance Sanity Check

An automated check was performed to flag if any calendar feature (is_month_end, is_month_start, month) ranks above any lag-1 or rolling-mean feature — which would indicate the model is not learning temporal dynamics properly.

- **Random Forest (in_hhs): PASS** — All lag-1 and rolling-mean features rank above calendar features.
- **Random Forest (discharged): FAIL** — `day_of_week` ranks 9th, outranking `discharged_lag14` (10th) and several cross-column lag features. This is expected and interpretable: discharge volume has strong weekly seasonality (17.8% weekend effect), making day-of-week a genuinely informative feature for this target.
- **Gradient Boosting (discharged): FAIL** — `day_of_week` ranks 4th, outranking `in_hhs_lag1` (5th). Same explanation as above: the discharge process operates on a weekly cycle driven by administrative schedules, making calendar effects a first-order predictor.

The FAIL flags are **not indicative of model pathology** — they reflect the real operational reality that discharge processing follows a weekly schedule.

## 5. Key Insights

1. **Discharge rates increase 17.8% on weekends** — likely reflecting batch processing and family reunification schedules
2. **Machine learning models outperform statistical models** when sufficient lag features are available — RF beats SARIMA by 66% on in_hhs MAE
3. **The net pressure signal** (transfers out minus discharges) is a strong predictor of future care load stress
4. **Naive persistence is surprisingly competitive** — the care load series is highly autocorrelated, making last-value prediction a strong baseline
5. **Exponential smoothing fails** because the series has undergone structural breaks (regime changes in apprehension volumes) that the model cannot adapt to

## 6. Limitations

1. **14-day test window** is narrow — results may not generalize to high-volatility periods
2. **Missing date imputation** (355 days interpolated) introduces uncertainty
3. **No external regressors** — policy changes, border conditions, and seasonal migration patterns are not modeled
4. **Point-estimate focus** — confidence intervals for ML models are approximate (bootstrap-based)
5. **Single-target forecasting** — multivariate models (e.g., VAR) could capture cross-column dependencies

### 6.1 Known Unknowns

The following external factors are not captured in the current model but would likely improve forecast accuracy if included:

- **Policy changes and executive orders:** Immigration policy shifts (e.g., Title 42 expiration, asylum processing rule changes) cause discontinuous jumps in apprehension and care volumes that lag features cannot anticipate
- **Court rulings and legal mandates:** Flores Settlement Agreement compliance requirements, court-ordered release timelines, and legal advocacy capacity affect discharge timing independently of operational patterns
- **Shelter capacity and staffing levels:** Physical bed capacity, staff-to-child ratios, and facility opening/closing decisions create hard constraints on care load that the model treats as unconstrained
- **Seasonal migration patterns:** Central American dry season, hurricane season, and regional violence cycles drive apprehension surges on timescales longer than the 14-day forecast horizon

## 7. Recommendations

1. **Operationalize the Random Forest model** for 7-14 day care load planning
2. **Use Gradient Boosting** for discharge demand forecasting to optimize staffing
3. **Monitor the net pressure signal** as an early warning indicator for capacity stress
4. **Implement walk-forward retraining** weekly to adapt to trend changes
5. **Expand the feature set** with external data: border crossing statistics, policy announcements, regional migration data
6. **Deploy the Streamlit dashboard** for real-time operational monitoring and scenario planning

---

*Report generated from HHS UAC Program data (Jan 2023 – Dec 2025)*
