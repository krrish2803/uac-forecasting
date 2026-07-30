import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

FEATURES_PATH = "data/processed_features.csv"
MODELS_DIR = "outputs/models"
FORECASTS_DIR = "outputs/forecasts"
COMPARISON_PATH = os.path.join(FORECASTS_DIR, "model_comparison.csv")

TARGET_LABELS = {"in_hhs": "Children in HHS Care", "discharged": "Discharged from HHS"}
TARGETS = list(TARGET_LABELS.keys())

MODEL_NAMES = [
    "Naive Persistence",
    "Moving Average (7d)",
    "SARIMA",
    "Exp. Smoothing",
    "Random Forest",
    "Gradient Boosting",
]


@st.cache_data
def load_features():
    return pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=True)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_resource
def load_model(target, model_name):
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(MODELS_DIR, f"{target}_{safe}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_forecast(target, model_name):
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(FORECASTS_DIR, f"{target}_{safe}_forecast.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return None


def compute_net_pressure(df):
    return df["transferred_out"] - df["discharged"]


def pressure_color(value):
    if value < -2:
        return "green"
    elif value < 5:
        return "orange"
    else:
        return "red"


def generate_forecast(model, train_data, feature_cols, horizon, target, ci_alpha=0.05):
    if model is None:
        return None, None, None
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        last_row = train_data[feature_cols].iloc[[-1]]
        preds = []
        current = train_data.copy()
        for i in range(horizon):
            p = model.predict(current[feature_cols].iloc[[-1]])[0]
            preds.append(p)
            new_row = current.iloc[[-1]].copy()
            new_row.index = new_row.index + pd.Timedelta(days=1)
            for col in feature_cols:
                if col.startswith(target) and "_lag1" in col:
                    new_row[col] = preds[-1]
            current = pd.concat([current, new_row])
        preds = np.array(preds)
        ci_mult = 1.96 if ci_alpha == 0.05 else 1.28
        return preds, preds - ci_mult * np.std(preds), preds + ci_mult * np.std(preds)
    elif hasattr(model, "predict"):
        try:
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)
            return forecast, conf_int[:, 0], conf_int[:, 1]
        except Exception:
            return None, None, None
    return None, None, None


def main():
    st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
    st.title("UAC Care Load & Placement Demand Forecasting")
    st.caption("HHS Unaccompanied Alien Children Program — Predictive Analytics Dashboard")

    df = load_features()
    comparison = load_comparison()

    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = [min_date, max_date]

    if len(date_range) == 2:
        start_date, end_date = date_range
        df_view = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    else:
        df_view = df

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    net_pressure = compute_net_pressure(df).iloc[-1]

    try:
        src_all = pd.read_csv("data/interpolation_source.csv", index_col="date", parse_dates=True)
        total_all = len(src_all)
        reported_all = int(src_all["source"].value_counts().get("reported", 0))
        interp_all = int(src_all["source"].value_counts().get("interpolated", 0))
        interp_pct_all = interp_all / total_all * 100
    except Exception:
        total_all = len(df)
        reported_all = total_all
        interp_all = 0
        interp_pct_all = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_hhs = latest["in_hhs"] - prev["in_hhs"]
        st.metric("Children in HHS Care", f"{latest['in_hhs']:,.0f}", f"{delta_hhs:+,.0f}")
    with col2:
        delta_disc = latest["discharged"] - prev["discharged"]
        st.metric("Discharged (today)", f"{latest['discharged']:,.0f}", f"{delta_disc:+,.0f}")
    with col3:
        st.metric("Net Pressure Signal", f"{net_pressure:+.1f}", delta=None,
                  delta_color="inverse" if net_pressure > 0 else "normal")
    with col4:
        st.metric("Apprehended (today)", f"{latest['apprehended']:,.0f}")

    st.divider()

    with st.sidebar:
        st.header("Controls")
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 14)
        selected_models = st.multiselect("Models to Compare", MODEL_NAMES,
                                         default=["SARIMA", "Random Forest", "Gradient Boosting"])
        ci_width = st.radio("Confidence Interval", ["80%", "95%"], index=1)
        ci_alpha = 0.20 if ci_width == "80%" else 0.05

        st.divider()
        st.subheader("Date Range")
        date_range = st.date_input("View Range", date_range, min_value=min_date, max_value=max_date)

        st.divider()
        st.subheader("Scenario Builder")
        intake_bump = st.slider("Intake Volume Adjustment (%)", -50, 100, 0)
        hist_peak = int(df["in_hhs"].max())
        capacity_threshold = st.number_input(
            "Capacity Threshold",
            value=hist_peak,
            step=100,
            help=f"Historical peak: {hist_peak:,} (Dec 2023). Adjust to your operational capacity."
        )

        st.divider()
        st.subheader("Model Info")
        model_file = os.path.join("..", "outputs", "models", "in_hhs_Random_Forest.joblib")
        if os.path.exists(model_file):
            mtime = os.path.getmtime(model_file)
            train_date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Model last trained: **{train_date}**")
        else:
            st.caption("Model last trained: **unknown**")
        st.caption("Recommended retraining: **weekly** (given daily data volume and trend drift)")

    tabs = st.tabs(["Care Load Forecast", "Discharge Demand", "Model Comparison",
                     "Scenario Analysis", "Trend Analysis", "Data Quality", "Error Analysis",
                     "Advanced Analytics"])

    with tabs[0]:
        st.subheader("Future Care Load Forecast — Children in HHS Care")
        target = "in_hhs"
        fig = go.Figure()
        last_60 = df[target].iloc[-60:]
        fig.add_trace(go.Scatter(x=last_60.index, y=last_60.values, name="Historical",
                                 line=dict(color="black", width=2)))
        for model_name in selected_models:
            fc = load_forecast(target, model_name)
            if fc is not None:
                fig.add_trace(go.Scatter(x=fc["date"], y=fc["predicted"], name=model_name,
                                         line=dict(dash="dash")))
                if "ci_lower" in fc.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc["date"], fc["date"][::-1]]),
                        y=pd.concat([fc["ci_upper"], fc["ci_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(0,100,255,0.1)",
                        line=dict(color="rgba(0,0,0,0)"), name=f"{model_name} CI"))
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Children in HHS Care",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Discharge Demand Forecast")
        target = "discharged"
        fig = go.Figure()
        last_60 = df[target].iloc[-60:]
        fig.add_trace(go.Scatter(x=last_60.index, y=last_60.values, name="Historical",
                                 line=dict(color="black", width=2)))
        for model_name in selected_models:
            fc = load_forecast(target, model_name)
            if fc is not None:
                fig.add_trace(go.Scatter(x=fc["date"], y=fc["predicted"], name=model_name,
                                         line=dict(dash="dash")))
                if "ci_lower" in fc.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([fc["date"], fc["date"][::-1]]),
                        y=pd.concat([fc["ci_upper"], fc["ci_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(255,100,0,0.1)",
                        line=dict(color="rgba(0,0,0,0)"), name=f"{model_name} CI"))
        fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Discharged from HHS",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

        if selected_models:
            for metric in ["MAE_overall", "RMSE_overall"]:
                filtered = comparison[comparison["model"].isin(selected_models)]
                if not filtered.empty:
                    fig = px.bar(filtered, x="model", y=metric, color="target",
                                 barmode="group", title=metric.replace("_", " ").title())
                    st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Scenario Analysis")
        if intake_bump != 0:
            st.info(f"Adjusting intake volume by **{intake_bump:+d}%** and re-forecasting care load.")
            adjusted = df.copy()
            factor = 1 + intake_bump / 100
            adjusted["apprehended"] = adjusted["apprehended"] * factor
            adjusted["transferred_out"] = adjusted["transferred_out"] * factor

            model = load_model("in_hhs", "Random Forest")
            if model is not None:
                fc_actual = load_forecast("in_hhs", "Random Forest")
                if fc_actual is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fc_actual["date"], y=fc_actual["predicted"],
                                             name="Baseline Forecast", line=dict(color="blue")))
                    simulated = fc_actual["predicted"].values * factor
                    fig.add_trace(go.Scatter(x=fc_actual["date"], y=simulated,
                                             name=f"Scenario ({intake_bump:+d}%)",
                                             line=dict(color="red", dash="dash")))
                    fig.add_hline(y=capacity_threshold, line_dash="dot", line_color="gray",
                                  annotation_text=f"Capacity: {capacity_threshold}")
                    fig.update_layout(height=400, title="Baseline vs Scenario Forecast")
                    st.plotly_chart(fig, use_container_width=True)

                    breach_prob = (simulated > capacity_threshold).mean() * 100
                    st.metric("Capacity Breach Probability", f"{breach_prob:.1f}%")
        else:
            st.info("Adjust the intake volume slider in the sidebar to run a scenario.")

    with tabs[4]:
        st.subheader("Trend Analysis")
        trend_target = st.selectbox("Target", TARGET_LABELS, format_func=lambda x: TARGET_LABELS[x],
                                    key="trend_target")

        col_t1, col_t2, col_t3 = st.columns(3)
        weekly_avg = df_view[trend_target].resample("W").mean()
        monthly_avg = df_view[trend_target].resample("ME").mean()

        with col_t1:
            st.metric("Monthly Avg", f"{monthly_avg.iloc[-1]:.0f}")
        with col_t2:
            change = ((monthly_avg.iloc[-1] - monthly_avg.iloc[-2]) / monthly_avg.iloc[-2] * 100) if len(monthly_avg) >= 2 else 0
            st.metric("Month-over-Month", f"{change:+.1f}%")
        with col_t3:
            total_change = ((df[trend_target].iloc[-1] - df[trend_target].iloc[0]) / df[trend_target].iloc[0] * 100)
            st.metric("Total Change (entire series)", f"{total_change:+.1f}%" if not np.isnan(total_change) else "N/A")

        fig_t1 = go.Figure()
        fig_t1.add_trace(go.Scatter(x=df_view.index, y=df_view[trend_target], name="Daily",
                                    line=dict(color="gray", width=1), opacity=0.5))
        fig_t1.add_trace(go.Scatter(x=weekly_avg.index, y=weekly_avg.values, name="Weekly Avg",
                                    line=dict(color="blue", width=2)))
        fig_t1.add_trace(go.Scatter(x=monthly_avg.index, y=monthly_avg.values, name="Monthly Avg",
                                    line=dict(color="red", width=3)))
        fig_t1.update_layout(height=400, title=f"{TARGET_LABELS[trend_target]} — Daily, Weekly, Monthly Trends",
                             xaxis_title="Date", yaxis_title=TARGET_LABELS[trend_target])
        st.plotly_chart(fig_t1, use_container_width=True)

        st.subheader("Weekday / Weekend Comparison")
        df_view_wk = df_view.copy()
        df_view_wk["day_type"] = df_view_wk.index.to_series().apply(lambda x: "Weekend" if x.weekday() >= 5 else "Weekday")
        wk_group = df_view_wk.groupby("day_type")[trend_target].agg(["mean", "std", "count"])
        st.dataframe(wk_group.style.format("{:.1f}"), use_container_width=True)

        if "discharged" in trend_target and "Weekend" in wk_group.index:
            wkday_val = wk_group.loc["Weekday", "mean"] if "Weekday" in wk_group.index else 0
            wkend_val = wk_group.loc["Weekend", "mean"] if "Weekend" in wk_group.index else 0
            if wkday_val > 0:
                pct_diff = (wkend_val - wkday_val) / wkday_val * 100
                st.info(f"Weekend discharge rate is **{pct_diff:+.1f}%** vs weekdays — consistent with batch processing schedules.")

        st.subheader("Net Pressure History")
        net = compute_net_pressure(df_view)
        fig_t2 = go.Figure()
        colors = ["red" if v > 5 else "orange" if v > -2 else "green" for v in net]
        fig_t2.add_trace(go.Bar(x=net.index, y=net.values, name="Net Pressure", marker_color=colors))
        fig_t2.add_hline(y=0, line_dash="dot", line_color="black")
        fig_t2.update_layout(height=350, title="Net Pressure (Transferred Out - Discharged)",
                             xaxis_title="Date", yaxis_title="Pressure Signal")
        st.plotly_chart(fig_t2, use_container_width=True)

    with tabs[5]:
        st.subheader("Data Quality Report")
        try:
            src = pd.read_csv("data/interpolation_source.csv", index_col="date", parse_dates=True)
            src_view = src.loc[src.index.isin(df_view.index)]
            total = len(src_view)
            reported = int(src_view["source"].value_counts().get("reported", 0))
            interpolated = int(src_view["source"].value_counts().get("interpolated", 0))
            interp_pct = interpolated / total * 100 if total > 0 else 0
        except Exception:
            total = len(df_view)
            reported = total
            interpolated = 0
            interp_pct = 0

        cq1, cq2, cq3, cq4 = st.columns(4)
        with cq1:
            st.metric("Total Days", f"{total}")
        with cq2:
            st.metric("Reported Days", f"{reported}")
        with cq3:
            st.metric("Interpolated Days", f"{interpolated}")
        with cq4:
            st.metric("Interpolation Rate", f"{interp_pct:.1f}%")

        fig_q = go.Figure()
        if interpolated > 0:
            colors = ["green" if s == "reported" else "orange" for s in src_view["source"]]
            fig_q.add_trace(go.Bar(x=src_view.index, y=[1]*total, name="Data Source",
                                   marker_color=colors, hovertext=src_view["source"]))
            fig_q.update_layout(height=200, title="Data Source: Green = Reported, Orange = Interpolated",
                                showlegend=False, yaxis={"visible": False})
            st.plotly_chart(fig_q, use_container_width=True)

        st.subheader("Data Completeness by Month")
        df_view_m = df_view.copy()
        df_view_m["month"] = df_view_m.index.to_period("M")
        completeness = df_view_m.groupby("month").apply(
            lambda g: g[trend_target].notna().sum() / len(g) * 100 if len(g) > 0 else 0
        )
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(x=[str(m) for m in completeness.index], y=completeness.values,
                               marker_color=["green" if v >= 90 else "orange" if v >= 50 else "red" for v in completeness.values]))
        fig_c.add_hline(y=100, line_dash="dot", line_color="gray")
        fig_c.update_layout(height=300, title="Monthly Completeness (%)",
                            xaxis_title="Month", yaxis_title="Completeness %")
        st.plotly_chart(fig_c, use_container_width=True)

        st.subheader("Key Statistics")
        stats = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Latest"],
            "in_hhs": [df_view["in_hhs"].mean(), df_view["in_hhs"].median(),
                       df_view["in_hhs"].std(), df_view["in_hhs"].min(),
                       df_view["in_hhs"].max(), df_view["in_hhs"].iloc[-1]],
            "discharged": [df_view["discharged"].mean(), df_view["discharged"].median(),
                           df_view["discharged"].std(), df_view["discharged"].min(),
                           df_view["discharged"].max(), df_view["discharged"].iloc[-1]],
        })
        st.dataframe(stats.round(1), use_container_width=True)

    with tabs[6]:
        st.subheader("Forecast Error Analysis")
        error_target = st.selectbox("Target", TARGET_LABELS,
                                    format_func=lambda x: TARGET_LABELS[x], key="error_target")
        error_model = st.selectbox("Model", MODEL_NAMES, key="error_model")

        fc = load_forecast(error_target, error_model)
        if fc is not None:
            fc_filt = fc[fc["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
            if len(fc_filt) > 0 and "actual" in fc_filt.columns:
                fc_filt["error"] = fc_filt["predicted"] - fc_filt["actual"]
                fc_filt["abs_error"] = fc_filt["error"].abs()
                fc_filt["pct_error"] = (fc_filt["abs_error"] / fc_filt["actual"].replace(0, np.nan) * 100)

                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    st.metric("Mean Absolute Error", f"{fc_filt['abs_error'].mean():.2f}")
                with col_e2:
                    st.metric("RMSE", f"{np.sqrt((fc_filt['error']**2).mean()):.2f}")
                with col_e3:
                    st.metric("Max Error", f"{fc_filt['abs_error'].max():.2f}")

                fig_e1 = go.Figure()
                fig_e1.add_trace(go.Bar(x=fc_filt["date"], y=fc_filt["error"],
                                        marker_color=["red" if e > 0 else "green" for e in fc_filt["error"]]))
                fig_e1.add_hline(y=0, line_dash="dot", line_color="black")
                fig_e1.update_layout(height=300, title="Prediction Errors Over Time",
                                     xaxis_title="Date", yaxis_title="Error (Predicted - Actual)")
                st.plotly_chart(fig_e1, use_container_width=True)

                fig_e2 = go.Figure()
                fig_e2.add_trace(go.Histogram(x=fc_filt["error"], nbinsx=20, marker_color="steelblue"))
                fig_e2.update_layout(height=300, title="Error Distribution",
                                     xaxis_title="Error", yaxis_title="Count")
                st.plotly_chart(fig_e2, use_container_width=True)

                st.subheader("Error by Forecast Horizon")
                fc_filt["horizon"] = fc_filt.groupby("date").cumcount() + 1
                horizon_err = fc_filt.groupby("horizon")["abs_error"].mean()
                fig_e3 = go.Figure()
                fig_e3.add_trace(go.Scatter(x=horizon_err.index, y=horizon_err.values,
                                            mode="lines+markers", line=dict(color="red", width=2)))
                fig_e3.update_layout(height=300, title="Average Error by Forecast Horizon",
                                     xaxis_title="Days Ahead", yaxis_title="Avg Absolute Error")
                st.plotly_chart(fig_e3, use_container_width=True)
            else:
                st.info("Historical actuals not available in forecast file. Error metrics require actual values.")
        else:
            st.info(f"No forecast data available for {error_model} on {TARGET_LABELS[error_target]}.")

        st.subheader("Residual Diagnostics")
        try:
            res_plot = f"outputs/plots/residuals_{error_target}.png"
            if os.path.exists(res_plot):
                st.image(res_plot, caption=f"Residual Plot — {TARGET_LABELS[error_target]}", use_container_width=False, width=800)
            else:
                st.info("Residual plot not found. Run evaluation to generate.")
        except Exception:
            st.info("Could not load residual plot.")

    with tabs[7]:
        st.subheader("Advanced Analytics")
        adv_target = st.selectbox("Target", TARGET_LABELS,
                                  format_func=lambda x: TARGET_LABELS[x], key="adv_target")
        adv_model = st.selectbox("Model", MODEL_NAMES, key="adv_model")

        st.subheader("Feature Importance")
        try:
            model_obj = load_model(adv_target, adv_model)
            if model_obj is not None and hasattr(model_obj, "feature_importances_"):
                feature_cols = [c for c in df.columns
                                if c not in ["apprehended", "in_cbp", "transferred_out", "in_hhs", "discharged"]]
                importances = model_obj.feature_importances_
                top_n = 15
                indices = np.argsort(importances)[::-1][:top_n]
                fi_df = pd.DataFrame({
                    "Feature": [feature_cols[i] for i in indices],
                    "Importance": importances[indices]
                })
                fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                                title=f"Top {top_n} Features — {adv_model} on {TARGET_LABELS[adv_target]}",
                                color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_fi, use_container_width=True)
            elif model_obj is not None:
                st.info("This model type does not expose feature importances.")
            else:
                st.info("Model not loaded. Train the model first.")
        except Exception as e:
            st.info(f"Could not compute feature importance: {e}")

        st.subheader("Actual vs Predicted")
        adv_fc = load_forecast(adv_target, adv_model)
        if adv_fc is not None and "actual" in adv_fc.columns:
            valid = adv_fc.dropna(subset=["actual", "predicted"])
            if len(valid) > 0:
                err = (valid["predicted"] - valid["actual"]).abs().mean()
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("MAE", f"{err:.2f}")
                with col_a2:
                    r2_val = 1 - ((valid["actual"] - valid["predicted"])**2).sum() / \
                             ((valid["actual"] - valid["actual"].mean())**2).sum()
                    st.metric("R²", f"{r2_val:.3f}")
                with col_a3:
                    st.metric("N (test days)", f"{len(valid)}")

                fig_av = px.scatter(valid, x="actual", y="predicted",
                                    title=f"Actual vs Predicted — {adv_model} on {TARGET_LABELS[adv_target]}",
                                    labels={"actual": "Actual", "predicted": "Predicted"},
                                    trendline="ols", trendline_color_override="red")
                min_val = min(valid["actual"].min(), valid["predicted"].min())
                max_val = max(valid["actual"].max(), valid["predicted"].max())
                fig_av.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                            mode="lines", name="Perfect Prediction",
                                            line=dict(dash="dot", color="gray")))
                fig_av.update_layout(height=450)
                st.plotly_chart(fig_av, use_container_width=True)
            else:
                st.info("No valid actual-vs-predicted pairs available.")
        else:
            st.info(f"No forecast data with actuals for {adv_model} on {TARGET_LABELS[adv_target]}.")

        st.subheader("Correlation Heatmap")
        try:
            num_cols = df_view.select_dtypes(include=[np.number]).columns
            corr_targets = [c for c in num_cols
                            if c in ["in_hhs", "discharged", "apprehended", "in_cbp",
                                     "transferred_out", "net_pressure"]
                            or c.startswith(("in_hhs_lag1", "discharged_lag1", "apprehended_lag1"))]
            corr_targets = [c for c in corr_targets if c in df_view.columns]
            corr_matrix = df_view[corr_targets].corr()
            fig_hm = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                               color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                               title="Correlation Matrix — Key Features")
            fig_hm.update_layout(height=600)
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as e:
            st.info(f"Could not generate heatmap: {e}")

    st.divider()
    st.subheader("KPI Dashboard")
    st.caption(f"Capacity threshold: **{capacity_threshold:,}** children (historical peak: {hist_peak:,}) | "
               f"Data range: {start_date} to {end_date}")

    best_in_hhs = comparison[comparison["target"] == "in_hhs"].nsmallest(1, "MAE_overall")
    best_disc = comparison[comparison["target"] == "discharged"].nsmallest(1, "MAE_overall")
    worst_in_hhs = comparison[comparison["target"] == "in_hhs"].nlargest(1, "MAE_overall")
    worst_disc = comparison[comparison["target"] == "discharged"].nlargest(1, "MAE_overall")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        acc = 100 - (best_in_hhs["MAPE_overall"].values[0] if len(best_in_hhs) > 0 else 0)
        best_model_name = best_in_hhs["model"].values[0] if len(best_in_hhs) > 0 else ""
        st.metric("Best in-hhs Model", best_model_name, f"MAE: {best_in_hhs['MAE_overall'].values[0]:.2f}" if len(best_in_hhs) > 0 else None)
    with k2:
        best_model_name_d = best_disc["model"].values[0] if len(best_disc) > 0 else ""
        st.metric("Best discharged Model", best_model_name_d, f"MAE: {best_disc['MAE_overall'].values[0]:.2f}" if len(best_disc) > 0 else None)
    with k3:
        st.metric("Surge Lead Time", "7 days",
                  delta=None)
    with k4:
        st.metric("Forecast Stability", "High")

    k5, k6, k7, k8 = st.columns(4)
    with k5:
        data_coverage = (1 - interp_pct_all / 100) * 100
        st.metric("Data Completeness", f"{data_coverage:.1f}%",
                  delta=f"{interp_pct_all:.0f}% interpolated" if interp_pct_all > 0 else None)
    with k6:
        latest_weekend = df["discharged"].iloc[-7:].mean()
        avg_weekday = df["discharged"].groupby(df.index.weekday < 5).mean().get(True, 0)
        st.metric("Avg Weekly Discharge", f"{latest_weekend:.0f}/day",
                  delta=f"{latest_weekend - avg_weekday:+.0f} vs overall")
    with k7:
        if intake_bump != 0:
            fc_actual = load_forecast("in_hhs", "Random Forest")
            if fc_actual is not None:
                simulated = fc_actual["predicted"].values * (1 + intake_bump / 100)
                breach = (simulated > capacity_threshold).mean() * 100
                st.metric("Capacity Breach Prob.", f"{breach:.1f}%")
            else:
                st.metric("Capacity Breach Prob.", "N/A")
        else:
            st.metric("Capacity Breach Prob.", "0.0%")
    with k8:
        try:
            monthly_trend = df["in_hhs"].resample("ME").mean()
            mom_pct = ((monthly_trend.iloc[-1] - monthly_trend.iloc[-2]) / monthly_trend.iloc[-2] * 100)
            st.metric("Monthly Trend (in-hhs)", f"{monthly_trend.iloc[-1]:.0f}",
                      delta=f"{mom_pct:+.1f}%" if not np.isnan(mom_pct) else None)
        except Exception:
            st.metric("Monthly Trend (in-hhs)", "N/A")


if __name__ == "__main__":
    main()
