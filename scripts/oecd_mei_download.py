import dbnomics
import pandas as pd
import os

# 1. Define Countries
COUNTRIES = [
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHE",
    "CHL",
    "COL",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HUN",
    "IRL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LUX",
    "MEX",
    "NLD",
    "NOR",
    "NZL",
    "POL",
    "PRT",
    "SWE",
    "TUR",
    "USA",
]

# 2. Define the Mapping (Subject Codes)
# We map the code to a readable name
VAR_MAP = {
    # Dataset 1: MEI (Macro)
    "PRMNTO01": "IP",
    "CPALTT01": "CPI",
    "IRSTCI01": "ShortRate",
    "IRLTLT01": "LongRate",
    "SPASTT01": "Stocks",
    "LRUNTTT": "Unemployment",
    # Dataset 2: MEI_CLI (Sentiment)
    "BSCICP03": "Biz_Confidence",
    "CSCICP03": "Cons_Confidence",
}


def fetch_and_merge():
    print("--- STARTING TWIN-ENGINE DOWNLOAD ---")

    # ---------------------------------------------------------
    # ENGINE 1: Fetch Macro Data (MEI)
    # ---------------------------------------------------------
    print("1. Fetching Macro Data (MEI)...")
    mei_subjects = [
        "PRMNTO01",
        "CPALTT01",
        "IRSTCI01",
        "IRLTLT01",
        "SPASTT01",
        "LRUNTTT",
    ]

    try:
        df_mei = dbnomics.fetch_series(
            "OECD",
            "MEI",
            max_nb_series=2000,
            dimensions={
                "LOCATION": COUNTRIES,
                "SUBJECT": mei_subjects,
                "FREQUENCY": ["M"],
            },
        )
        df_mei["Dataset"] = "MEI"
        print(f"   -> Found {len(df_mei)} rows in MEI.")
    except Exception as e:
        print(f"   -> MEI Error: {e}")
        df_mei = pd.DataFrame()

    # ---------------------------------------------------------
    # ENGINE 2: Fetch Sentiment Data (MEI_CLI)
    # ---------------------------------------------------------
    print("2. Fetching Sentiment Data (MEI_CLI)...")
    cli_subjects = ["BSCICP03", "CSCICP03"]

    try:
        df_cli = dbnomics.fetch_series(
            "OECD",
            "MEI_CLI",
            max_nb_series=2000,
            dimensions={
                "LOCATION": COUNTRIES,
                "SUBJECT": cli_subjects,
                "FREQUENCY": ["M"],
            },
        )
        df_cli["Dataset"] = "MEI_CLI"
        print(f"   -> Found {len(df_cli)} rows in MEI_CLI.")
    except Exception as e:
        print(f"   -> MEI_CLI Error: {e}")
        df_cli = pd.DataFrame()

    # ---------------------------------------------------------
    # MERGE & CLEAN
    # ---------------------------------------------------------
    if df_mei.empty and df_cli.empty:
        print("CRITICAL: No data found in either dataset.")
        return

    # Stack them on top of each other
    df = pd.concat([df_mei, df_cli], ignore_index=True)

    # Clean Columns (Handle case sensitivity)
    # DBnomics usually gives 'SUBJECT' and 'LOCATION' in uppercase for these datasets
    cols_to_keep = ["period", "value", "LOCATION", "SUBJECT", "MEASURE", "Dataset"]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    df = df.rename(
        columns={
            "period": "Date",
            "value": "Value",
            "LOCATION": "Country",
            "SUBJECT": "Subject_Code",
            "MEASURE": "Measure_Code",
        }
    )

    # Map Variable Names
    df["Variable"] = df["Subject_Code"].map(VAR_MAP)

    # ---------------------------------------------------------
    # INTELLIGENT FILTERING
    # ---------------------------------------------------------
    print("3. Filtering for best measures...")
    # We prioritize specific measures to avoid duplicates
    # Confidence (CLI) is usually 'IXOBSA' (Amplitude Adjusted)
    # Rates are 'ST' or 'PA'
    # Stocks are 'IX' or 'IXOBSA'

    # We allow ALL these valid codes
    valid_measures = ["IX", "IXOBSA", "LV", "ST", "STSA", "PA", "PC", "GP"]
    # Note: Added 'PC' (Percent) just in case Unemployment uses it

    df_clean = df[df["Measure_Code"].isin(valid_measures)].copy()

    # Remove duplicates: Prioritize Seasonally Adjusted (IXOBSA/STSA)
    df_clean = df_clean.sort_values(
        by=["Country", "Variable", "Date", "Measure_Code"],
        ascending=[True, True, True, False],
    )
    df_clean = df_clean.drop_duplicates(
        subset=["Country", "Variable", "Date"], keep="first"
    )

    # Final Check
    found_vars = df_clean["Variable"].unique()
    print(f"\nSUCCESS! Found {len(found_vars)} variables: {found_vars}")

    # Create directory if it doesn't exist
    os.makedirs("../data/raw data", exist_ok=True)

    df_clean.to_csv("../data/raw data/oecd_mei_final.csv", index=False)
    print(
        "Saved to '../data/raw data/oecd_mei_final.csv'. Run the Diagnostics script now to check coverage."
    )


if __name__ == "__main__":
    fetch_and_merge()
