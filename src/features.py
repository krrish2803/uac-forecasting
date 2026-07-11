import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

PROCESSED_PATH = "data/processed_features.csv"
OUTPUT_PATH = "data/processed_features.csv"


def add_lag_features(df, cols, lags):
    for c in cols:
        for lag in lags:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    return df


def add_rolling_features(df, cols, windows):
    for c in cols:
        for w in windows:
            df[f"{c}_rmean{w}"] = df[c].rolling(window=w, min_periods=1).mean()
            df[f"{c}_rstd{w}"] = df[c].rolling(window=w, min_periods=1).std()
    return df


def add_calendar_features(df):
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df.index.month
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"] = df.index.is_month_end.astype(int)
    return df


def add_pressure_signal(df):
    df["net_pressure"] = df["transferred_out"] - df["discharged"]
    return df


def build_features(df):
    numeric_cols = ["apprehended", "in_cbp", "transferred_out", "in_hhs", "discharged"]
    df = add_lag_features(df, numeric_cols, [1, 7, 14])
    df = add_rolling_features(df, numeric_cols, [7, 14])
    df = add_pressure_signal(df)
    df = add_calendar_features(df)
    df = df.dropna()
    return df


def run_feature_engineering():
    print("=" * 60)
    print("PHASE 1: Feature Engineering")
    print("=" * 60)

    df = pd.read_csv(PROCESSED_PATH, index_col="date", parse_dates=True)
    print(f"\nInput data: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    featured_df = build_features(df)
    print(f"\nAfter feature engineering: {featured_df.shape}")
    print(f"Total features: {featured_df.shape[1]}")

    featured_df.to_csv(OUTPUT_PATH)

    feature_names = [c for c in featured_df.columns]
    lag_features = [c for c in feature_names if "_lag" in c]
    rolling_features = [c for c in feature_names if "_rmean" in c or "_rstd" in c]
    calendar_features = ["day_of_week", "is_weekend", "month", "is_month_start", "is_month_end"]
    pressure_features = ["net_pressure"]
    base_features = ["apprehended", "in_cbp", "transferred_out", "in_hhs", "discharged"]

    print(f"\nFeature Summary:")
    print(f"  Base numeric features:    {len(base_features)}")
    print(f"  Lag features (t-1,7,14):  {len(lag_features)}")
    print(f"  Rolling features (7,14):  {len(rolling_features)}")
    print(f"  Calendar features:        {len(calendar_features)}")
    print(f"  Pressure signal:          {len(pressure_features)}")
    print(f"  TOTAL:                    {len(feature_names)}")

    print(f"\nData range after dropna: {featured_df.index.min().date()} to {featured_df.index.max().date()}")
    print(f"Rows available for modeling: {len(featured_df)}")

    print(f"\nFirst 5 rows preview:")
    print(featured_df.head())

    return featured_df


if __name__ == "__main__":
    run_feature_engineering()
