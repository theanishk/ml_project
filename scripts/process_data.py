import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# The "Bad List" identified from your matrix
DROP_COUNTRIES = ["AUS", "NZL", "TUR"]


def process_data():
    print("--- STARTING DATA PROCESSING ---")

    # 1. Load Data
    input_path = os.path.join("..", "data", "raw data", "oecd_mei_final.csv")
    try:
        df = pd.read_csv(input_path)
        df["Date"] = pd.to_datetime(df["Date"])
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found.")
        return

    # 2. Filter Countries
    # Remove the ones with missing data
    df = df[~df["Country"].isin(DROP_COUNTRIES)]
    print(
        f"Filtered out {DROP_COUNTRIES}. Remaining Countries: {df['Country'].nunique()}"
    )

    # 3. Pivot (Long -> Wide)
    # Index: Date, Country | Columns: Variables
    df_wide = df.pivot_table(
        index=["Date", "Country"], columns="Variable", values="Value"
    )
    df_wide = df_wide.sort_index()

    print(f"Data Matrix Shape: {df_wide.shape}")
    print(f"Variables available: {df_wide.columns.tolist()}")

    # ---------------------------------------------------------
    # 4. FEATURE ENGINEERING (The Transformations)
    # ---------------------------------------------------------
    print("\n--- APPLYING ECONOMIC TRANSFORMATIONS ---")

    def transform_country(group):
        # A. INFLATION (YoY)
        # Log difference of CPI over 12 months
        # Replace zero/negative values with NaN to avoid log warnings
        cpi_ratio = group["CPI"] / group["CPI"].shift(12)
        cpi_ratio = cpi_ratio.replace([0, -np.inf, np.inf], np.nan)
        group["Inflation_YoY"] = np.log(cpi_ratio.clip(lower=1e-10)) * 100

        cpi_ratio_mom = group["CPI"] / group["CPI"].shift(1)
        cpi_ratio_mom = cpi_ratio_mom.replace([0, -np.inf, np.inf], np.nan)
        group["Inflation_MoM"] = np.log(cpi_ratio_mom.clip(lower=1e-10)) * 100

        # B. REAL ECONOMIC GROWTH (IP)
        ip_ratio = group["IP"] / group["IP"].shift(12)
        ip_ratio = ip_ratio.replace([0, -np.inf, np.inf], np.nan)
        group["IP_Growth_YoY"] = np.log(ip_ratio.clip(lower=1e-10)) * 100

        # C. YIELD CURVE (The Recession Predictor)
        group["Term_Spread"] = group["LongRate"] - group["ShortRate"]

        # D. REAL RATES (Monetary Stance)
        group["Real_Short_Rate"] = group["ShortRate"] - group["Inflation_YoY"]

        # E. REAL STOCK RETURNS
        stock_ratio = group["Stocks"] / group["Stocks"].shift(1)
        stock_ratio = stock_ratio.replace([0, -np.inf, np.inf], np.nan)
        stock_ret = np.log(stock_ratio.clip(lower=1e-10)) * 100
        group["Real_Stock_Return"] = stock_ret - group["Inflation_MoM"]

        return group

    # Apply transformations country-by-country
    df_final = df_wide.groupby("Country", group_keys=False).apply(transform_country)

    # ---------------------------------------------------------
    # 6. CLEANUP & SAVE
    # ---------------------------------------------------------
    # Drop the rows that became NaN due to shifting (first 12 months)
    features = [
        "Term_Spread",
        "Real_Short_Rate",
        "Real_Stock_Return",
        "IP_Growth_YoY",
    ]

    df_clean = df_final.dropna(subset=features)

    # Final Validation
    print("\n--- FINAL DATASET REPORT ---")
    print(f"Total Observations: {len(df_clean)}")
    print(f"Features: {features}")

    # Create output directory if it doesn't exist
    output_dir = os.path.join("..", "data", "cleaned data")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "final_oecd_mei.csv")
    df_clean.to_csv(output_path)
    print(f"\n✅ SUCCESS: Model-Ready data saved to '{output_path}'")


if __name__ == "__main__":
    process_data()
