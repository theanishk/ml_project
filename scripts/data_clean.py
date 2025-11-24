############################################
#           SCRIPT TO CLEAN DATA           #
############################################

import os
import pandas as pd

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw data")
CLEANED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned data")

# OECD countries ISO-2 codes
OECD_ISO2 = {
    "AU",
    "AT",
    "BE",
    "CA",
    "CL",
    "CO",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IS",
    "IE",
    "IL",
    "IT",
    "JP",
    "KR",
    "LV",
    "LT",
    "LU",
    "MX",
    "NL",
    "NZ",
    "NO",
    "PL",
    "PT",
    "SK",
    "SI",
    "ES",
    "SE",
    "CH",
    "TR",
    "GB",
    "US",
}

# ISO-2 to ISO-3 mapping
ISO2_TO_ISO3 = {
    "AU": "AUS",
    "AT": "AUT",
    "BE": "BEL",
    "CA": "CAN",
    "CL": "CHL",
    "CO": "COL",
    "CZ": "CZE",
    "DK": "DNK",
    "EE": "EST",
    "FI": "FIN",
    "FR": "FRA",
    "DE": "DEU",
    "GR": "GRC",
    "HU": "HUN",
    "IS": "ISL",
    "IE": "IRL",
    "IL": "ISR",
    "IT": "ITA",
    "JP": "JPN",
    "KR": "KOR",
    "LV": "LVA",
    "LT": "LTU",
    "LU": "LUX",
    "MX": "MEX",
    "NL": "NLD",
    "NZ": "NZL",
    "NO": "NOR",
    "PL": "POL",
    "PT": "PRT",
    "SK": "SVK",
    "SI": "SVN",
    "ES": "ESP",
    "SE": "SWE",
    "CH": "CHE",
    "TR": "TUR",
    "GB": "GBR",
    "US": "USA",
}


def clean_bis_credit(df) -> pd.DataFrame:
    """
    Cleans the BIS credit dataset by removing metadata rows,
    selecting only OECD countries with credit metric:
    "Credit to Private Non-Financial Sector, All sectors, % of GDP"
    """
    # Find the first row containing a valid BIS code pattern
    bis_code_row = None
    for idx, row in df.iterrows():
        # Check if any cell in this row matches BIS code pattern (starts with Q:)
        if any(str(cell).startswith("Q:") for cell in row):
            bis_code_row = idx
            break

    if bis_code_row is None:
        # Debug: Print first few rows to diagnose
        print("\n⚠ DEBUG: No BIS codes found. Printing first 5 rows:")
        print(df.head())
        print(f"\nFirst row values:")
        if len(df) > 0:
            for i, val in enumerate(df.iloc[0]):
                print(f"  Col {i}: {str(val)[:50]}")
        raise ValueError("No BIS codes found in dataset. Check file structure.")

    # Remove metadata rows
    df_cleaned = df.iloc[bis_code_row:].reset_index(drop=True)

    # First row is now the header with BIS codes
    df_cleaned.columns = df_cleaned.iloc[0]
    df_cleaned = df_cleaned.iloc[1:].reset_index(drop=True)

    # Convert date column to datetime
    date_col = df_cleaned.columns[0]
    df_cleaned[date_col] = pd.to_datetime(
        df_cleaned[date_col], format="%d.%m.%Y", errors="coerce"
    )
    df_cleaned = df_cleaned.dropna(subset=[date_col])

    # Filter date range: Jan 1990 to Dec 2024
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")
    df_cleaned = df_cleaned[
        (df_cleaned[date_col] >= start_date) & (df_cleaned[date_col] <= end_date)
    ]

    # Select only OECD countries with specific credit metric
    # Target pattern: Q:CC:P:A:M:770:A (or similar variants)
    # P = Private non-financial sector
    # A = All sectors
    # M = Market value
    # 770 = % of GDP
    selected_cols = [date_col]
    for col in df_cleaned.columns[1:]:
        code_str = str(col)
        if code_str.startswith("Q:"):
            parts = code_str.split(":")
            if len(parts) >= 6:
                country_code = parts[1]
                sector = parts[2]
                lending = parts[3]
                valuation = parts[4]
                unit = parts[5]

                # Filter for OECD countries AND the specific metric
                if (
                    country_code in OECD_ISO2
                    and sector == "P"
                    and lending == "A"
                    and valuation == "M"
                    and unit == "770"
                ):
                    selected_cols.append(col)

    print(f"  Found {len(selected_cols) - 1} matching columns")
    df_cleaned = df_cleaned[selected_cols].copy()

    # Show which OECD countries have no data for this metric
    found_countries = {col.split(":")[1] for col in selected_cols[1:]}
    missing_countries = OECD_ISO2 - found_countries
    if missing_countries:
        missing_iso3 = sorted([ISO2_TO_ISO3[code] for code in missing_countries])
        print(
            f"  Missing data for {len(missing_countries)} countries: {', '.join(missing_iso3)}"
        )

    # Rename columns from ISO-2 codes to ISO-3 codes
    rename_dict = {}
    for col in df_cleaned.columns:
        if col != date_col:
            iso2_code = col.split(":")[1]
            iso3_code = ISO2_TO_ISO3.get(iso2_code)
            if iso3_code:
                rename_dict[col] = iso3_code
    df_cleaned = df_cleaned.rename(columns=rename_dict)

    # Convert to year-month format (ignore day)
    df_cleaned[date_col] = df_cleaned[date_col].dt.to_period("M")
    df_cleaned = df_cleaned.set_index(date_col)

    print("\nCleaned BIS credit data.")
    print(f"  Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
    print(f"  OECD countries: {len(df_cleaned.columns)}")
    print(f"  Shape: {df_cleaned.shape}")

    return df_cleaned


def clean_bis_reer(df) -> pd.DataFrame:
    """
    Cleans the BIS REER dataset by removing metadata rows,
    selecting only OECD countries with REER metric.
    Row 0: Country names
    Row 1: REER codes (RBXX format, e.g., RBAR for Argentina)
    Row 2+: Data with dates
    """
    # REER codes for OECD countries (RB + ISO-2)
    REER_OECD_CODES = {f"RB{iso2}" for iso2 in OECD_ISO2}

    # First row is headers (Date + country names)
    # Second row contains REER codes
    if len(df) < 3:
        raise ValueError("REER dataset must have at least 3 rows (headers + data).")

    # Get the REER code row (second row, index 1)
    reer_codes_row = df.iloc[1]
    date_col = df.columns[0]

    # Find columns with OECD REER codes
    selected_cols = [date_col]
    for col in df.columns[1:]:
        code_str = str(reer_codes_row[col]).strip()
        if code_str in REER_OECD_CODES:
            selected_cols.append(col)

    print(f"  Found {len(selected_cols) - 1} OECD REER countries")

    # Keep only relevant columns and remove header rows (first 2 rows)
    df_cleaned = df[selected_cols].iloc[2:].reset_index(drop=True)
    df_cleaned.columns = selected_cols

    # Convert date column to datetime (format: MM-YYYY)
    df_cleaned[date_col] = pd.to_datetime(
        df_cleaned[date_col], format="%m-%Y", errors="coerce"
    )
    df_cleaned = df_cleaned.dropna(subset=[date_col])

    # Filter date range: Jan 1990 to Dec 2024
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")
    df_cleaned = df_cleaned[
        (df_cleaned[date_col] >= start_date) & (df_cleaned[date_col] <= end_date)
    ]

    # Rename columns from REER codes to ISO-3 codes
    rename_dict = {}
    for col in df_cleaned.columns:
        if col != date_col:
            iso2_code = str(reer_codes_row[col]).strip()[2:]  # Remove 'RB' prefix
            iso3_code = ISO2_TO_ISO3.get(iso2_code)
            if iso3_code:
                rename_dict[col] = iso3_code
    df_cleaned = df_cleaned.rename(columns=rename_dict)

    # Convert to year-month format
    df_cleaned[date_col] = df_cleaned[date_col].dt.to_period("M")
    df_cleaned = df_cleaned.set_index(date_col)

    # Convert data to numeric
    for col in df_cleaned.columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

    print("\nCleaned BIS REER data.")
    print(f"  Date range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
    print(f"  OECD countries: {len(df_cleaned.columns)}")
    print(f"  Shape: {df_cleaned.shape}")

    return df_cleaned


def aggregate_to_monthly(
    df, date_col, value_col, start_date, end_date, halflife_days=20
) -> pd.DataFrame:
    """
    Aggregates daily time series data to monthly using EOM (end-of-month) methods.

    Returns columns:
    - {value_col}_raw_eom: End-of-month raw value
    - {value_col}_ema_eom: End-of-month EMA value
    """
    df = df.copy()
    df = df.rename(columns={date_col: "date", value_col: "value"})

    # Convert date to datetime BEFORE setting as index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.set_index("date").sort_index()

    # Filter date range
    df = df.loc[start_date:end_date]

    # Forward-fill daily gaps (weekends/holidays)
    df = df.ffill()

    # EOM raw
    eom_raw = df["value"].resample("ME").last().rename(f"{value_col}_raw_eom")

    # EMA daily
    ema = df["value"].ewm(halflife=halflife_days, adjust=False).mean()
    eom_ema = ema.resample("ME").last().rename(f"{value_col}_ema_eom")

    # Combine results
    out = pd.concat([eom_raw, eom_ema], axis=1)

    return out


def clean_oil_prices(df) -> pd.DataFrame:
    """
    Cleans oil price (DCOILBRENTEU) dataset.
    Aggregates daily data to monthly EOM values.
    """
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")

    df = df.astype({df.columns[0]: str, df.columns[1]: float})
    df_agg = aggregate_to_monthly(
        df, df.columns[0], df.columns[1], start_date, end_date, halflife_days=20
    )

    # Add period column and rename
    df_agg["Period"] = df_agg.index.strftime("%Y-%m")
    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.rename(
        columns={"DCOILBRENTEU_raw_eom": "raw_eom", "DCOILBRENTEU_ema_eom": "ema_eom"}
    )
    df_agg = df_agg[["Period", "raw_eom", "ema_eom"]]

    print("\nCleaned Oil Price data (DCOILBRENTEU).")
    print(f"  Monthly observations: {len(df_agg)}")
    print(f"  Shape: {df_agg.shape}")

    return df_agg


def clean_fedfunds(df) -> pd.DataFrame:
    """
    Cleans Federal Funds Rate (FEDFUNDS) dataset.
    Data is already clean, just needs date conversion and monthly alignment.
    Columns: observation_date, FEDFUNDS
    """
    df_cleaned = df.copy()

    # Rename columns for consistency
    df_cleaned.columns = ["Date", "FEDFUNDS"]

    # Convert date column to datetime
    df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=["Date"])

    # Convert rate to numeric
    df_cleaned["FEDFUNDS"] = pd.to_numeric(df_cleaned["FEDFUNDS"], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=["FEDFUNDS"])

    # Filter date range: Jan 1990 to Dec 2024
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")
    df_cleaned = df_cleaned[
        (df_cleaned["Date"] >= start_date) & (df_cleaned["Date"] <= end_date)
    ]

    # Convert to year-month and take last value of each month
    df_cleaned["YearMonth"] = df_cleaned["Date"].dt.to_period("M")
    monthly_data = df_cleaned.groupby("YearMonth")["FEDFUNDS"].last().reset_index()

    # Rename YearMonth to Period and format
    monthly_data["Period"] = monthly_data["YearMonth"].astype(str)
    monthly_data = monthly_data[["Period", "FEDFUNDS"]]

    print("\nCleaned Federal Funds Rate data.")
    print(f"  Monthly observations: {len(monthly_data)}")
    print(f"  Shape: {monthly_data.shape}")

    return monthly_data


def clean_indpro(df) -> pd.DataFrame:
    """
    Cleans Industrial Production (INDPRO) dataset.
    Data is already clean, just needs date conversion and monthly alignment.
    Columns: observation_date, INDPRO
    """
    df_cleaned = df.copy()

    # Rename columns for consistency
    df_cleaned.columns = ["Date", "INDPRO"]

    # Convert date column to datetime
    df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=["Date"])

    # Convert index to numeric
    df_cleaned["INDPRO"] = pd.to_numeric(df_cleaned["INDPRO"], errors="coerce")
    df_cleaned = df_cleaned.dropna(subset=["INDPRO"])

    # Filter date range: Jan 1990 to Dec 2024
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")
    df_cleaned = df_cleaned[
        (df_cleaned["Date"] >= start_date) & (df_cleaned["Date"] <= end_date)
    ]

    # Convert to year-month and take last value of each month
    df_cleaned["YearMonth"] = df_cleaned["Date"].dt.to_period("M")
    monthly_data = df_cleaned.groupby("YearMonth")["INDPRO"].last().reset_index()

    # Rename YearMonth to Period and format
    monthly_data["Period"] = monthly_data["YearMonth"].astype(str)
    monthly_data = monthly_data[["Period", "INDPRO"]]

    print("\nCleaned Industrial Production data.")
    print(f"  Monthly observations: {len(monthly_data)}")
    print(f"  Shape: {monthly_data.shape}")

    return monthly_data


def clean_vixcls(df) -> pd.DataFrame:
    """
    Cleans VIX (VIXCLS) dataset.
    Aggregates daily data to monthly EOM values.
    """
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")

    df = df.astype({df.columns[0]: str, df.columns[1]: float})
    df_agg = aggregate_to_monthly(
        df, df.columns[0], df.columns[1], start_date, end_date, halflife_days=20
    )

    # Add period column and rename
    df_agg["Period"] = df_agg.index.strftime("%Y-%m")
    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.rename(
        columns={"VIXCLS_raw_eom": "raw_eom", "VIXCLS_ema_eom": "ema_eom"}
    )
    df_agg = df_agg[["Period", "raw_eom", "ema_eom"]]

    print("\nCleaned VIX data (VIXCLS).")
    print(f"  Monthly observations: {len(df_agg)}")
    print(f"  Shape: {df_agg.shape}")

    return df_agg


def clean_term_spread(df) -> pd.DataFrame:
    """
    Cleans Term Spread (T10Y2Y) dataset.
    Aggregates daily data to monthly EOM values.
    """
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")

    df = df.astype({df.columns[0]: str, df.columns[1]: float})
    df_agg = aggregate_to_monthly(
        df, df.columns[0], df.columns[1], start_date, end_date, halflife_days=20
    )

    # Add period column and rename
    df_agg["Period"] = df_agg.index.strftime("%Y-%m")
    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.rename(
        columns={"T10Y2Y_raw_eom": "raw_eom", "T10Y2Y_ema_eom": "ema_eom"}
    )
    df_agg = df_agg[["Period", "raw_eom", "ema_eom"]]

    print("\nCleaned Term Spread data (T10Y2Y).")
    print(f"  Monthly observations: {len(df_agg)}")
    print(f"  Shape: {df_agg.shape}")

    return df_agg


def load_and_clean_bis_credit(
    raw_file_path: str, output_file_path: str
) -> pd.DataFrame:
    """
    Load and clean BIS credit data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING BIS CREDIT DATA")
    print("=" * 50)
    df_bis = pd.read_excel(
        raw_file_path, sheet_name=1, header=None
    )  # Read 2nd sheet (index 1)
    df_cleaned = clean_bis_credit(df_bis)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def load_and_clean_bis_reer(raw_file_path: str, output_file_path: str) -> pd.DataFrame:
    """
    Load and clean BIS REER data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING BIS REER DATA")
    print("=" * 50)
    df_bis_reer = pd.read_excel(raw_file_path, header=None)
    df_cleaned = clean_bis_reer(df_bis_reer)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def load_and_clean_oil_prices(
    raw_file_path: str, output_file_path: str
) -> pd.DataFrame:
    """
    Load and clean oil price data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING OIL PRICE DATA")
    print("=" * 50)
    df_oil = pd.read_csv(raw_file_path)
    df_cleaned = clean_oil_prices(df_oil)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def load_and_clean_fedfunds(raw_file_path: str, output_file_path: str) -> pd.DataFrame:
    """
    Load and clean Federal Funds Rate data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING FEDERAL FUNDS RATE DATA")
    print("=" * 50)
    df_fedfunds = pd.read_csv(raw_file_path)
    df_cleaned = clean_fedfunds(df_fedfunds)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def load_and_clean_indpro(raw_file_path: str, output_file_path: str) -> pd.DataFrame:
    """
    Load and clean Industrial Production data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING INDUSTRIAL PRODUCTION DATA")
    print("=" * 50)
    df_indpro = pd.read_csv(raw_file_path)
    df_cleaned = clean_indpro(df_indpro)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def load_and_clean_vixcls(raw_file_path: str, output_file_path: str) -> pd.DataFrame:
    """
    Load and clean VIX data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING VIX DATA")
    print("=" * 50)
    df_vix = pd.read_csv(raw_file_path)
    df_cleaned = clean_vixcls(df_vix)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def clean_term_spread(df) -> pd.DataFrame:
    """
    Cleans Term Spread (T10Y2Y) dataset.
    Aggregates daily data to monthly EOM values.
    """
    start_date = pd.to_datetime("1990-01-01")
    end_date = pd.to_datetime("2024-12-31")

    df = df.astype({df.columns[0]: str, df.columns[1]: float})
    df_agg = aggregate_to_monthly(
        df, df.columns[0], df.columns[1], start_date, end_date, halflife_days=20
    )

    # Add period column and rename
    df_agg["Period"] = df_agg.index.strftime("%Y-%m")
    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.rename(
        columns={"T10Y2Y_raw_eom": "raw_eom", "T10Y2Y_ema_eom": "ema_eom"}
    )
    df_agg = df_agg[["Period", "raw_eom", "ema_eom"]]

    print("\nCleaned Term Spread data (T10Y2Y).")
    print(f"  Monthly observations: {len(df_agg)}")
    print(f"  Shape: {df_agg.shape}")

    return df_agg


def load_and_clean_term_spread(
    raw_file_path: str, output_file_path: str
) -> pd.DataFrame:
    """
    Load and clean Term Spread data from file.
    """
    print("\n" + "=" * 50)
    print("CLEANING TERM SPREAD DATA")
    print("=" * 50)
    df_spread = pd.read_csv(raw_file_path)
    df_cleaned = clean_term_spread(df_spread)
    df_cleaned.to_csv(output_file_path)
    print(f"✓ Saved to: {output_file_path}")
    return df_cleaned


def main(
    credit=True,
    reer=True,
    oil=True,
    fedfunds=True,
    indpro=True,
    vix=True,
    term_spread=True,
):
    """
    Main orchestrator function.

    Args:
        credit: Whether to clean BIS credit data
        reer: Whether to clean BIS REER data
        oil: Whether to clean oil price data
        fedfunds: Whether to clean Federal Funds Rate data
        indpro: Whether to clean Industrial Production data
        vix: Whether to clean VIX data
        term_spread: Whether to clean Term Spread data
    """
    results = {}

    if credit:
        bis_credit_file = os.path.join(RAW_DATA_DIR, "BIS_credit.xlsx")
        cleaned_credit_path = os.path.join(
            CLEANED_DATA_DIR, "bis_credit_data_cleaned.csv"
        )
        results["credit"] = load_and_clean_bis_credit(
            bis_credit_file, cleaned_credit_path
        )

    if reer:
        bis_reer_file = os.path.join(RAW_DATA_DIR, "BIS_reer.xlsx")
        cleaned_reer_path = os.path.join(CLEANED_DATA_DIR, "bis_reer_data_cleaned.csv")
        results["reer"] = load_and_clean_bis_reer(bis_reer_file, cleaned_reer_path)

    if oil:
        oil_file = os.path.join(RAW_DATA_DIR, "DCOILBRENTEU.csv")
        cleaned_oil_path = os.path.join(CLEANED_DATA_DIR, "oil_prices_cleaned.csv")
        results["oil"] = load_and_clean_oil_prices(oil_file, cleaned_oil_path)

    if fedfunds:
        fedfunds_file = os.path.join(RAW_DATA_DIR, "FEDFUNDS.csv")
        cleaned_fedfunds_path = os.path.join(CLEANED_DATA_DIR, "fedfunds_cleaned.csv")
        results["fedfunds"] = load_and_clean_fedfunds(
            fedfunds_file, cleaned_fedfunds_path
        )

    if indpro:
        indpro_file = os.path.join(RAW_DATA_DIR, "INDPRO.csv")
        cleaned_indpro_path = os.path.join(CLEANED_DATA_DIR, "indpro_cleaned.csv")
        results["indpro"] = load_and_clean_indpro(indpro_file, cleaned_indpro_path)

    if vix:
        vix_file = os.path.join(RAW_DATA_DIR, "VIXCLS.csv")
        cleaned_vix_path = os.path.join(CLEANED_DATA_DIR, "vixcls_cleaned.csv")
        results["vix"] = load_and_clean_vixcls(vix_file, cleaned_vix_path)

    if term_spread:
        spread_file = os.path.join(RAW_DATA_DIR, "T10Y2Y.csv")
        cleaned_spread_path = os.path.join(CLEANED_DATA_DIR, "term_spread_cleaned.csv")
        results["term_spread"] = load_and_clean_term_spread(
            spread_file, cleaned_spread_path
        )

    print("\n" + "=" * 50)
    print("COMPLETED")
    print("=" * 50)
    return results


if __name__ == "__main__":
    results = main(
        credit=True,
        reer=True,
        oil=True,
        fedfunds=True,
        indpro=True,
        vix=True,
        term_spread=True,
    )
