import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

PLOTS_DIR = "outputs/plots"
MODELS_DIR = "outputs/models"
FORECASTS_DIR = "outputs/forecasts"
REPORTS_DIR = "reports"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def plot_feature_importances(importances, target, model_name, top_n=15):
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_imp)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), values, color="#2196F3")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance: {model_name} ({target})")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{target}_{model_name.replace(' ', '_')}_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_forecast_comparison(actual, predictions_dict, target, dates):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, "k-", linewidth=2, label="Actual", marker="o", markersize=3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    for (name, preds), color in zip(predictions_dict.items(), colors):
        ax.plot(dates, preds, label=name, color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(target.replace("_", " ").title())
    ax.set_title(f"Forecast Comparison: {target.replace('_', ' ').title()}")
    ax.legend(fontsize=8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{target}_forecast_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confidence_intervals(actual, preds, ci_lower, ci_upper, target, model_name, dates):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(dates, ci_lower, ci_upper, alpha=0.2, color="#2196F3", label="95% CI")
    ax.plot(dates, actual, "k-", linewidth=2, label="Actual")
    ax.plot(dates, preds, "r--", linewidth=1.5, label="Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel(target.replace("_", " ").title())
    ax.set_title(f"Confidence Intervals: {model_name} ({target})")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{target}_{model_name.replace(' ', '_')}_ci.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_residuals(actual, predicted, dates, target, model_name):
    residuals = actual - predicted
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Time series of residuals
    ax = axes[0]
    ax.plot(dates, residuals, "o-", color="#E91E63", markersize=4, linewidth=1.2)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Residuals Over Time: {model_name} ({target})")
    ax.tick_params(axis="x", rotation=45)

    # Highlight clustering: flag dates with |residual| > 1.5 * IQR
    q75 = np.percentile(np.abs(residuals), 75)
    q25 = np.percentile(np.abs(residuals), 25)
    iqr = q75 - q25
    threshold = 1.5 * iqr
    outliers = np.abs(residuals) > threshold
    if outliers.any():
        ax.scatter(dates[outliers], residuals[outliers], color="red", s=80, zorder=5, label="Outlier (IQR)")
        ax.legend()

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=10, color="#9C27B0", alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution: {model_name} ({target})")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"residuals_{target}.png")
    plt.savefig(path, dpi=150)
    plt.close()

    # Report clustering
    clusters = []
    abs_res = np.abs(residuals)
    for i in range(len(residuals)):
        if abs_res[i] > threshold:
            clusters.append((dates[i], residuals[i]))
    return path, clusters, threshold


def feature_importance_sanity_check(importances, target, model_name):
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feature_ranking = [f for f, _ in sorted_imp]

    calendar_features = {"is_month_end", "is_month_start", "month", "day_of_week", "is_weekend"}
    lag_rolling_features = {f for f in feature_ranking if "_lag1" in f or "_rmean" in f}

    # Find positions of calendar features in the ranking
    calendar_positions = {}
    lag_positions = {}
    for i, f in enumerate(feature_ranking):
        if f in calendar_features:
            calendar_positions[f] = i
        if f in lag_rolling_features:
            lag_positions[f] = i

    # Check if any calendar feature ranks higher (lower index) than any lag-1 or rolling-mean feature
    fail = False
    fail_details = []
    for cal_f, cal_pos in calendar_positions.items():
        for lr_f, lr_pos in lag_positions.items():
            if cal_pos < lr_pos:
                fail = True
                fail_details.append(f"  {cal_f} (rank {cal_pos+1}) > {lr_f} (rank {lr_pos+1})")

    status = "FAIL" if fail else "PASS"
    print(f"\n  Feature Sanity Check ({model_name}, {target}): {status}")
    print(f"    Calendar feature positions: {calendar_positions}")
    print(f"    Lag/rolling feature positions: {dict(sorted(lag_positions.items(), key=lambda x: x[1]))}")
    if fail:
        print(f"    SUSPICIOUS: Calendar features outrank lag/rolling features:")
        for d in fail_details:
            print(d)
    else:
        print(f"    OK: Lag/rolling features properly outrank calendar features")

    return status


def generate_markdown_report(comparison_df, all_results):
    lines = []
    lines.append("# UAC Forecasting - Model Evaluation Report\n")
    lines.append("## Model Comparison Summary\n")
    lines.append(comparison_df.to_string(index=False))
    lines.append("")

    for target in all_results:
        lines.append(f"\n## Target: {target.replace('_', ' ').title()}\n")
        lines.append("### Per-Horizon Metrics\n")
        for model_name, res in all_results[target].items():
            lines.append(f"\n#### {model_name}\n")
            for horizon, m in res["metrics"].items():
                lines.append(f"- **{horizon}**: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, MAPE={m['MAPE']:.1f}%")
        if "Random Forest" in all_results[target]:
            imp = all_results[target]["Random Forest"]["importances"]
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("\n### Top 10 Features (Random Forest)\n")
            for f, v in sorted_imp:
                lines.append(f"- {f}: {v:.4f}")
        if "Gradient Boosting" in all_results[target]:
            imp = all_results[target]["Gradient Boosting"]["importances"]
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("\n### Top 10 Features (Gradient Boosting)\n")
            for f, v in sorted_imp:
                lines.append(f"- {f}: {v:.4f}")

    report_path = os.path.join(REPORTS_DIR, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def run_evaluation(all_results, comparison_df):
    print("=" * 60)
    print("PHASE 2: Evaluation & Visualization")
    print("=" * 60)

    # Load test data for residual plots
    features = pd.read_csv("data/processed_features.csv", index_col="date", parse_dates=True)
    test = features.iloc[-14:]
    source_mask = pd.read_csv("data/interpolation_source.csv", index_col="date", parse_dates=True)["source"]

    # Best models per target
    best_models = {"in_hhs": "Random Forest", "discharged": "Gradient Boosting"}

    for target in all_results:
        preds_dict = {}
        for model_name, res in all_results[target].items():
            preds_dict[model_name] = res["predictions"]
            if "importances" in res:
                path = plot_feature_importances(res["importances"], target, model_name)
                print(f"  Saved: {path}")

            # Feature importance sanity check
            if "importances" in res:
                feature_importance_sanity_check(res["importances"], target, model_name)

            if "conf_int" in res:
                fc_df = pd.read_csv(
                    f"{FORECASTS_DIR}/{target}_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_forecast.csv"
                )
                test_dates = pd.to_datetime(fc_df["date"])
                path = plot_confidence_intervals(
                    test[target].values, res["predictions"],
                    res["conf_int"][:, 0], res["conf_int"][:, 1],
                    target, model_name, test_dates,
                )
                print(f"  Saved: {path}")

        # Residual diagnostics for best model per target
        if target in best_models:
            best_name = best_models[target]
            if best_name in all_results[target]:
                res = all_results[target][best_name]
                path, clusters, threshold = plot_residuals(
                    test[target].values, res["predictions"], test.index, target, best_name
                )
                print(f"\n  Residual plot: {path}")
                print(f"  IQR threshold: {threshold:.2f}")
                if clusters:
                    print(f"  Outlier dates ({len(clusters)}):")
                    for d, r in clusters:
                        interp_status = source_mask.get(d, "unknown")
                        print(f"    {d.date()}: residual={r:.1f} (source={interp_status})")
                else:
                    print(f"  No residual outliers detected")

    report_path = generate_markdown_report(comparison_df, all_results)
    print(f"\nReport saved to: {report_path}")

    comparison_path = os.path.join(FORECASTS_DIR, "model_comparison.csv")
    print(f"Comparison table saved to: {comparison_path}")

    return report_path


if __name__ == "__main__":
    from models import run_modeling
    all_results, comparison_df = run_modeling()
    run_evaluation(all_results, comparison_df)
