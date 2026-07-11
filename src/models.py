import pandas as pd
import numpy as np
import joblib
import warnings
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

warnings.filterwarnings("ignore")

FEATURES_PATH = "data/processed_features.csv"
SOURCE_PATH = "data/interpolation_source.csv"
MODELS_DIR = "outputs/models"
FORECASTS_DIR = "outputs/forecasts"

TARGETS = ["in_hhs", "discharged"]


def load_data():
    return pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=True)


def load_source_mask():
    src = pd.read_csv(SOURCE_PATH, index_col="date", parse_dates=True)
    return src["source"]


def time_train_test_split(df, test_days=14):
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test


def naive_persistence(train, test, target):
    preds = test[f"{target}_lag1"].values
    return preds


def moving_average_forecast(train, test, target, window=7):
    last_vals = train[target].rolling(window=window).mean().iloc[-1]
    preds = np.full(len(test), last_vals)
    return preds


def fit_sarima(train, target, seasonal=True, m=7):
    model = pm.auto_arima(
        train[target], seasonal=seasonal, m=m,
        suppress_warnings=True, stepwise=True, error_action="ignore",
    )
    return model


def sarima_forecast(model, test, target, alpha=0.05):
    n = len(test)
    forecast, conf_int = model.predict(n_periods=n, return_conf_int=True, alpha=alpha)
    return forecast, conf_int


def fit_exponential_smoothing(train, target):
    configs = [
        {"trend": "add", "seasonal": "add", "sp": 7, "bc": False},
        {"trend": "add", "seasonal": "mul", "sp": 7, "bc": False},
        {"trend": "add", "seasonal": "add", "sp": 7, "bc": True},
        {"trend": "add", "seasonal": None, "sp": None, "bc": False},
        {"trend": "mul", "seasonal": "add", "sp": 7, "bc": False},
    ]
    best_model = None
    best_aic = np.inf
    for cfg in configs:
        try:
            kwargs = {
                "trend": cfg["trend"], "seasonal_periods": cfg["sp"],
                "initialization_method": "estimated", "use_boxcox": cfg["bc"],
            }
            if cfg["seasonal"] is not None:
                kwargs["seasonal"] = cfg["seasonal"]
            model = ExponentialSmoothing(train[target], **kwargs).fit(optimized=True)
            aic = model.aic
            print(f"    Config: trend={cfg['trend']}, seasonal={cfg['seasonal']}, "
                  f"boxcox={cfg['bc']} -> AIC={aic:.1f}")
            if aic < best_aic:
                best_aic = aic
                best_model = model
        except Exception as e:
            print(f"    Config failed: {cfg} -> {e}")
    return best_model


def exp_smoothing_forecast(model, test, target, alpha=0.05):
    n = len(test)
    forecast = model.forecast(n)
    residuals = model.resid.dropna()
    sigma = residuals.std()
    z = 1.96
    lower = forecast - z * sigma
    upper = forecast + z * sigma
    return forecast.values, np.column_stack([lower.values, upper.values])


def get_feature_columns(df, target):
    exclude = [c for c in df.columns if c.startswith(target)]
    base_excludes = ["apprehended", "in_cbp", "transferred_out", "in_hhs", "discharged"]
    feature_cols = [c for c in df.columns if c not in exclude and c not in base_excludes]
    lag_cols = [f"{target}_lag1", f"{target}_lag7", f"{target}_lag14"]
    rolling_cols = [f"{target}_rmean7", f"{target}_rmean14", f"{target}_rstd7", f"{target}_rstd14"]
    all_cols = lag_cols + rolling_cols + feature_cols
    return [c for c in all_cols if c in df.columns]


def fit_ml_model(train, target, model_cls, **kwargs):
    feature_cols = get_feature_columns(train, target)
    X_train = train[feature_cols].values
    y_train = train[target].values
    model = model_cls(**kwargs)
    model.fit(X_train, y_train)
    return model, feature_cols


def ml_forecast(model, test, target, feature_cols):
    X_test = test[feature_cols].values
    point = model.predict(X_test)
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    if isinstance(model, RandomForestRegressor):
        preds_all = np.array([tree.predict(X_test) for tree in model.estimators_])
        lower = np.percentile(preds_all, 2.5, axis=0)
        upper = np.percentile(preds_all, 97.5, axis=0)
    elif isinstance(model, GradientBoostingRegressor):
        staged = np.array(list(model.staged_predict(X_test)))
        lower = point - 1.96 * staged.std(axis=0)
        upper = point + 1.96 * staged.std(axis=0)
    else:
        lower = point - 100
        upper = point + 100
    return point, np.column_stack([lower, upper])


def walk_forward_validation(df, target, test_days=14, min_train_days=200, n_folds=3):
    results = []
    actual_len = len(df)
    for fold in range(n_folds):
        split_point = actual_len - test_days - (n_folds - 1 - fold) * (test_days // n_folds)
        if split_point < min_train_days:
            split_point = min_train_days
        train = df.iloc[:split_point]
        test = df.iloc[split_point : split_point + test_days]
        if len(test) == 0:
            continue
        results.append((train, test))
    return results


def compute_metrics(y_true, y_pred, horizon_label="all"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return {"horizon": horizon_label, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def per_horizon_metrics(y_true, y_pred):
    metrics = {}
    if len(y_true) >= 1:
        metrics["1-day"] = compute_metrics(y_true[:1], y_pred[:1], "1-day")
    if len(y_true) >= 7:
        metrics["7-day"] = compute_metrics(y_true[:7], y_pred[:7], "7-day")
    if len(y_true) >= 14:
        metrics["14-day"] = compute_metrics(y_true[:14], y_pred[:14], "14-day")
    metrics["overall"] = compute_metrics(y_true, y_pred, "overall")
    return metrics


def reported_only_metrics(y_true, y_pred, source_mask, dates):
    mask = source_mask.reindex(dates).values == "reported"
    n = int(mask.sum())
    if n == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n": 0}
    yt = y_true[mask]
    yp = y_pred[mask]
    m = compute_metrics(yt, yp, "reported_only")
    m["n"] = n
    return m


def run_modeling():
    print("=" * 60)
    print("PHASE 2: Model Training & Evaluation")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FORECASTS_DIR, exist_ok=True)

    df = load_data()
    source_mask = load_source_mask()
    train, test = time_train_test_split(df, test_days=14)

    print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} days)")
    print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} days)")

    test_source = source_mask.reindex(test.index)
    n_reported = (test_source == "reported").sum()
    n_interp = (test_source == "interpolated").sum()
    print(f"Test set: {n_reported} reported, {n_interp} interpolated ({n_interp/len(test)*100:.0f}%)")

    print("\n--- Walk-Forward Folds: Interpolation Breakdown ---")
    folds = walk_forward_validation(df, "in_hhs", test_days=14, n_folds=3)
    for i, (tr, te) in enumerate(folds):
        te_src = source_mask.reindex(te.index)
        pct_interp = (te_src == "interpolated").sum() / len(te) * 100
        print(f"  Fold {i+1}: {len(te)} days, {pct_interp:.0f}% interpolated "
              f"({te.index.min().date()} to {te.index.max().date()})")

    all_results = {}
    reported_only_table = []

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")
        target_results = {}

        print("\n[1/6] Naive Persistence...")
        preds = naive_persistence(train, test, target)
        metrics = per_horizon_metrics(test[target].values, preds)
        target_results["Naive Persistence"] = {"predictions": preds, "metrics": metrics}
        print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")

        print("[2/6] Moving Average (7-day)...")
        preds = moving_average_forecast(train, test, target, window=7)
        metrics = per_horizon_metrics(test[target].values, preds)
        target_results["Moving Average (7d)"] = {"predictions": preds, "metrics": metrics}
        print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")

        print("[3/6] SARIMA...")
        try:
            sarima_model = fit_sarima(train, target, seasonal=True, m=7)
            preds, conf = sarima_forecast(sarima_model, test, target, alpha=0.05)
            metrics = per_horizon_metrics(test[target].values, preds)
            target_results["SARIMA"] = {
                "predictions": preds, "conf_int": conf,
                "metrics": metrics, "model": sarima_model,
            }
            print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")
        except Exception as e:
            print(f"  SARIMA failed: {e}")

        print("[4/6] Exponential Smoothing (multi-config search)...")
        try:
            es_model = fit_exponential_smoothing(train, target)
            if es_model is not None:
                print(f"    Best model AIC: {es_model.aic:.1f}")
                preds, conf = exp_smoothing_forecast(es_model, test, target)
                metrics = per_horizon_metrics(test[target].values, preds)
                target_results["Exp. Smoothing"] = {
                    "predictions": preds, "conf_int": conf,
                    "metrics": metrics, "model": es_model,
                }
                print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")
            else:
                print("  Exp Smoothing: all configs failed")
        except Exception as e:
            print(f"  Exp Smoothing failed: {e}")

        print("[5/6] Random Forest...")
        rf_model, feature_cols = fit_ml_model(
            train, target, RandomForestRegressor, n_estimators=200, random_state=42, n_jobs=-1
        )
        preds, conf = ml_forecast(rf_model, test, target, feature_cols)
        metrics = per_horizon_metrics(test[target].values, preds)
        importances = dict(zip(feature_cols, rf_model.feature_importances_))
        target_results["Random Forest"] = {
            "predictions": preds, "conf_int": conf,
            "metrics": metrics, "model": rf_model,
            "feature_cols": feature_cols, "importances": importances,
        }
        print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")

        print("[6/6] Gradient Boosting...")
        gb_model, feature_cols = fit_ml_model(
            train, target, GradientBoostingRegressor, n_estimators=200, random_state=42
        )
        preds, conf = ml_forecast(gb_model, test, target, feature_cols)
        metrics = per_horizon_metrics(test[target].values, preds)
        importances = dict(zip(feature_cols, gb_model.feature_importances_))
        target_results["Gradient Boosting"] = {
            "predictions": preds, "conf_int": conf,
            "metrics": metrics, "model": gb_model,
            "feature_cols": feature_cols, "importances": importances,
        }
        print(f"  Overall MAE: {metrics['overall']['MAE']:.2f}")

        all_results[target] = target_results

        for name, res in target_results.items():
            model_path = os.path.join(MODELS_DIR, f"{target}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
            if "model" in res:
                joblib.dump(res["model"], model_path)
            forecast_df = pd.DataFrame(
                {"date": test.index, "actual": test[target].values, "predicted": res["predictions"]}
            )
            if "conf_int" in res:
                forecast_df["ci_lower"] = res["conf_int"][:, 0]
                forecast_df["ci_upper"] = res["conf_int"][:, 1]
            forecast_path = os.path.join(FORECASTS_DIR, f"{target}_{name.replace(' ', '_').replace('(', '').replace(')', '')}_forecast.csv")
            forecast_df.to_csv(forecast_path, index=False)

        # Reported-only MAE comparison
        print(f"\n--- {target}: Full vs Reported-Only Test Set MAE ---")
        print(f"{'Model':<24} {'Full MAE':>10} {'Reported-Only MAE':>18} {'Reported N':>10}")
        print("-" * 66)
        for name, res in target_results.items():
            full_mae = res["metrics"]["overall"]["MAE"]
            ro = reported_only_metrics(
                test[target].values, res["predictions"], source_mask, test.index
            )
            print(f"{name:<24} {full_mae:>10.2f} {ro['MAE']:>18.2f} {ro['n']:>10}")
            reported_only_table.append({
                "target": target, "model": name,
                "full_MAE": round(full_mae, 2),
                "reported_only_MAE": round(ro["MAE"], 2),
                "reported_n": int(ro["n"]),
            })

    comparison_rows = []
    for target, results in all_results.items():
        for name, res in results.items():
            row = {"target": target, "model": name}
            for h_name, m in res["metrics"].items():
                row[f"MAE_{h_name}"] = round(m["MAE"], 2)
                row[f"RMSE_{h_name}"] = round(m["RMSE"], 2)
                row[f"MAPE_{h_name}"] = round(m["MAPE"], 2) if not np.isnan(m["MAPE"]) else None
            comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = os.path.join(FORECASTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    ro_df = pd.DataFrame(reported_only_table)
    ro_path = os.path.join(FORECASTS_DIR, "reported_only_comparison.csv")
    ro_df.to_csv(ro_path, index=False)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    print(f"\n{'='*60}")
    print("REPORTED-ONLY vs FULL TEST SET MAE")
    print(f"{'='*60}")
    print(ro_df.to_string(index=False))

    return all_results, comparison_df


if __name__ == "__main__":
    run_modeling()
