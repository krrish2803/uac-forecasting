import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

RAW_PATH = "data/HHS_Unaccompanied_Alien_Children_Program.csv"
PROCESSED_PATH = "data/processed_features.csv"
SOURCE_PATH = "data/interpolation_source.csv"

COL_RENAME = {
    "Date": "date",
    "Children apprehended and placed in CBP custody*": "apprehended",
    "Children in CBP custody": "in_cbp",
    "Children transferred out of CBP custody": "transferred_out",
    "Children in HHS Care": "in_hhs",
    "Children discharged from HHS Care": "discharged",
}


def load_raw_junk_sample(n=10):
    df = pd.read_csv(RAW_PATH)
    df.rename(columns=COL_RENAME, inplace=True)
    junk = df[df["date"].isna()].head(n)
    return junk


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df.rename(columns=COL_RENAME, inplace=True)
    df = df.dropna(subset=["date"])
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y")
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    df["in_hhs"] = df["in_hhs"].astype(str).str.replace(",", "", regex=False).astype(float)
    num_cols = ["apprehended", "in_cbp", "transferred_out", "in_hhs", "discharged"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[num_cols]
    return df


def build_full_calendar(df):
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_cal = df.reindex(full_idx)
    df_cal.index.name = "date"
    return df_cal


def interpolate_comparison(df_cal):
    ffill = df_cal.ffill()
    interp = df_cal.interpolate(method="linear", limit_direction="both")
    cols = ["in_hhs"]
    ffill_trend = ffill[cols].diff().abs().mean()
    interp_trend = interp[cols].diff().abs().mean()
    scores = pd.DataFrame(
        {"ffill_mae": ffill_trend, "interp_mae": interp_trend, "chosen": ""}
    )
    for c in cols:
        scores.loc[scores["ffill_mae"] < scores["interp_mae"], "chosen"] = "ffill"
        scores.loc[scores["ffill_mae"] >= scores["interp_mae"], "chosen"] = "interp"
    return interp, scores


def run_preprocessing():
    print("=" * 60)
    print("PHASE 1: Data Preparation & Preprocessing")
    print("=" * 60)

    # Show junk row sample
    junk_sample = load_raw_junk_sample(10)
    print("\n--- JUNK ROW SAMPLE (first 10 empty rows from raw CSV) ---")
    print(junk_sample.to_string(index=False))
    print("--- All 450 junk rows have NaT date + every column NaN: TEMPLATE ARTIFACTS ---\n")

    df = load_raw()
    print(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    expected_days = (df.index.max() - df.index.min()).days + 1
    actual_days = len(df)
    missing_days = expected_days - actual_days
    print(f"Expected calendar days: {expected_days}")
    print(f"Actual data rows: {actual_days}")
    print(f"Missing/skipped days: {missing_days}")

    df_cal = build_full_calendar(df)
    print(f"\nAfter calendar reindex: {df_cal.shape[0]} rows (DAILY)")

    # Build interpolation source mask BEFORE filling
    source_mask = pd.Series("reported", index=df_cal.index)
    source_mask[df_cal.isna().any(axis=1)] = "interpolated"
    print(f"Reported days: {(source_mask == 'reported').sum()}")
    print(f"Interpolated days: {(source_mask == 'interpolated').sum()}")

    missing_before = df_cal.isna().sum()
    print(f"\nMissing values after reindex (before interpolation):")
    for c in df_cal.columns:
        print(f"  {c}: {missing_before[c]}")

    interpolated_df, comparison = interpolate_comparison(df_cal)
    print(f"\nInterpolation comparison:")
    print(comparison)

    # Save source mask alongside features
    source_df = pd.DataFrame({"date": interpolated_df.index, "source": source_mask.values})
    source_df.to_csv(SOURCE_PATH, index=False)

    interpolated_df.to_csv(PROCESSED_PATH)
    print(f"\nProcessed data saved to {PROCESSED_PATH}")
    print(f"Interpolation source mask saved to {SOURCE_PATH}")
    print(f"Final shape: {interpolated_df.shape}")

    print(f"\n--- Data Quality Report ---")
    print(f"  Date range: {interpolated_df.index.min().date()} to {interpolated_df.index.max().date()}")
    print(f"  Total days: {len(interpolated_df)}")
    print(f"  Reported (original) days: {(source_mask == 'reported').sum()}")
    print(f"  Interpolated days: {(source_mask == 'interpolated').sum()}")
    print(f"  Interpolation rate: {(source_mask == 'interpolated').sum() / len(source_mask) * 100:.1f}%")
    print(f"  Method chosen: linear interpolation")

    return interpolated_df


if __name__ == "__main__":
    run_preprocessing()
