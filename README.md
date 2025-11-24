# Predicting Economic Recessions Using a Panel Based Machine Learning Framework

## Overview
This repository implements a machine learning framework for predicting economic recessions using a Multi-Factor Composite Index (MFCI) approach. The project combines macroeconomic indicators from multiple sources to create country-specific composite indices, then uses these along with global factors to train classification models.

**Key Components:**
- Data cleaning and integration from FRED, OECD, and BIS sources
- MFCI construction using Principal Component Analysis (PCA)
- Machine learning models (Random Forest, XGBoost, SVC) for recession prediction

## Repository Structure

```
ml_project/
├── data/
│   ├── cleaned data/          # Cleaned datasets ready for analysis
│   │   ├── bis_credit_data_cleaned.csv
│   │   ├── bis_reer_data_cleaned.csv
│   │   ├── fedfunds_cleaned.csv
│   │   ├── final_oecd_mei.csv
│   │   ├── indpro_cleaned.csv
│   │   ├── oil_prices_cleaned.csv
│   │   ├── term_spread_cleaned.csv
│   │   └── vixcls_cleaned.csv
│   ├── mfci/                  # MFCI computation outputs
│   │   ├── country_loadings.csv
│   │   ├── country_mfci.csv
│   │   ├── global_cycles.csv
│   │   ├── mfci_complete_dataset.csv
│   │   └── mfci_metadata.csv
│   └── raw data/              # Original downloaded datasets
├── notebooks/
│   ├── mfci.ipynb            # Data integration & MFCI creation
│   └── ml_model.ipynb        # Machine learning experiments
└── scripts/
    ├── data_clean.py         # Cleans FRED and BIS data
    ├── oecd_mei_download.py  # Downloads OECD MEI data via DBnomics API
    ├── process_data.py       # Processes OECD data (feature engineering)
    └── check.py              # Data coverage diagnostics
```

## Quick Start

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Pipeline

Run the data pipeline scripts in order:

**Step 1: Download OECD data**
```bash
python scripts/oecd_mei_download.py
```
- Downloads Main Economic Indicators (MEI) from OECD via DBnomics API
- Output: `data/raw data/oecd_mei_final.csv`
- Includes: CPI, Industrial Production, Interest Rates, Stock Indices

**Step 2: Process OECD data**
```bash
python scripts/process_data.py
```
- Transforms raw OECD data into model-ready format
- Applies feature engineering (YoY growth rates, real rates, term spreads)
- Output: `data/cleaned data/final_oecd_mei.csv`

**Step 3: Clean FRED and BIS data**
```bash
python scripts/data_clean.py
```
- Cleans FRED datasets (Federal Funds Rate, Industrial Production, VIX, Oil Prices, Term Spread)
- Cleans BIS datasets (Credit to GDP ratios, Real Effective Exchange Rates)
- Outputs: Multiple cleaned CSV files in `data/cleaned data/`

**Step 4: (Optional) Check data coverage**
```bash
python scripts/check.py
```
- Diagnoses data availability across countries and variables
- Helps identify missing data issues

### 3. MFCI Creation

Open and run the MFCI notebook:

```bash
jupyter notebook notebooks/mfci.ipynb
```

**What this notebook does:**
1. **Data Integration**: Merges OECD MEI, BIS, and FRED datasets
2. **Feature Engineering**: Creates stationary transformations (growth rates, real rates)
3. **Country-Level MFCI**: Uses PCA to extract first principal component from ~10 indicators per country
4. **Global Factors**: Extracts common global cycles from country MFCIs
5. **Outputs**: Saves complete dataset to `data/mfci/`

**Key outputs:**
- `mfci_complete_dataset.csv` — Full dataset with MFCI and all variables
- `country_mfci.csv` — Country-specific MFCI time series
- `global_cycles.csv` — Global factors (first 5 principal components)
- `country_loadings.csv` — Factor loadings showing country exposures

### 4. Machine Learning Models

Open and run the ML notebook:

```bash
jupyter notebook notebooks/ml_model.ipynb
```

**What this notebook does:**
- Loads MFCI dataset with recession labels
- Trains classification models (Random Forest, XGBoost, SVC)
- Performs hyperparameter tuning and cross-validation
- Evaluates model performance (accuracy, precision, recall, ROC curves)
- Analyzes feature importance

## Data Sources

### FRED (Federal Reserve Economic Data)
- [DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU) — Brent crude oil prices (daily → monthly)
- [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) — Federal funds effective rate (monthly)
- [INDPRO](https://fred.stlouisfed.org/series/INDPRO) — US industrial production index (monthly)
- [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y) — 10Y-2Y Treasury term spread (daily → monthly)
- [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) — CBOE VIX volatility index (daily → monthly)

### OECD
- [Main Economic Indicators (MEI)](https://data.oecd.org/mei.htm) via [DBnomics API](https://db.nomics.world/)
- Includes: CPI, Industrial Production, Short/Long Rates, Stock Indices, Unemployment
- Coverage: 28 OECD countries, 1995-2023

### BIS (Bank for International Settlements)
- [Credit Statistics](https://www.bis.org/statistics/totcredit.htm) — Credit to non-financial sector (% of GDP, quarterly)
- [Real Effective Exchange Rates](https://www.bis.org/statistics/eer.htm) — Broad REER indices (monthly)

## MFCI Methodology

The Multi-Factor Composite Index follows the BIS/IMF research framework:

### Four Pillars:
1. **Real Economy**: Industrial production growth, unemployment
2. **Prices & Monetary**: Inflation, real interest rates, yield curve
3. **Financial Markets**: Stock returns, VIX, credit spreads
4. **Credit & External**: Credit growth, real effective exchange rates, oil prices

### Construction Steps:
1. **Standardization**: Transform raw variables to stationary form (YoY growth, MoM changes)
2. **PCA per Country**: Extract first principal component from ~10 indicators
3. **Smoothing**: 3-month centered rolling average
4. **Normalization**: Z-score standardization (mean=0, std=1)
5. **Global Factors**: PCA on cross-section of country MFCIs

### Interpretation:
- **Positive MFCI** → Favorable macro-financial conditions
- **Negative MFCI** → Stressed conditions (potential recession signal)
- **Global Factor 1** → Common international business cycle (explains ~33% of variance)

## Technical Notes

### Requirements
- Python 3.8+
- Key packages: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `dbnomics`
- For interactive notebooks: `jupyter`, `ipykernel`

### Data Coverage
- **Time Period**: 1995-01 to 2023-12 (348 months)
- **Countries**: 28 OECD countries (excludes AUS, NZL, TUR due to data gaps)
- **Missing Data**: <1% overall (mainly quarterly credit data interpolated to monthly)

### Performance Considerations
- OECD download via API can take 2-5 minutes depending on network
- MFCI notebook processes ~7,440 country-month observations
- ML notebook training time varies by model (RF: ~30s, XGBoost: ~2min)

### Platform Notes
- **macOS users**: XGBoost may require `brew install gcc` for compilation
- **Windows users**: Ensure proper activation of virtual environment (`Scripts\activate` not `bin/activate`)
- Consider using prebuilt wheels from PyPI when available

## Troubleshooting

**"No module named 'dbnomics'"**
```bash
pip install dbnomics
```

**OECD download fails**
- Check internet connection
- Verify DBnomics API is accessible: https://api.db.nomics.world/
- API may have rate limits; wait and retry

**Missing data errors in notebooks**
- Ensure all cleaning scripts ran successfully
- Check `data/cleaned data/` contains all required CSV files
- Run `python scripts/check.py` to diagnose coverage issues

**MFCI values seem incorrect**
- Verify date ranges match across datasets (1995-2023)
- Check for NaN values in key indicators
- Ensure PCA convergence (explained variance should be 20-30% for Factor 1)