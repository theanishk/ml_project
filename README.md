# MFCI / ML Project

Overview
- This repository contains data preparation scripts, notebooks and datasets used to build a Multi‑Factor Common Index (MFCI) and follow-up ML experiments.
- Primary content: cleaned time-series data in `data/cleaned data/`, intermediate MFCI outputs in `data/mfci/`, and analysis notebooks in `notebooks/`.

Repository structure (high level)
- `data/cleaned data/` — cleaned CSV datasets used by the notebooks and scripts.
- `data/mfci/` — MFCI results, loadings and summary tables.
- `notebooks/mfci.ipynb` — notebook that computes the MFCI (PCA, scaling, plots).
- `notebooks/ml_model.ipynb` — model-training experiments (Random Forest, XGBoost, SVC, etc.).
- `scripts/` — helper scripts for cleaning, processing and downloading data:
  - `data_clean.py` — cleaning raw CSVs into `data/cleaned data/`.
  - `oecd_mei_download.py` — pulls OECD MEI data via `dbnomics`.
  - `process_data.py` — processing and merging routines.
  - `check.py` — quick sanity checks on outputs.

Quick start
1. Create and activate a Python virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies from `requirements.txt`:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Reproduce data preprocessing (example):

   ```bash
   python scripts/data_clean.py
   python scripts/process_data.py
   ```

4. Open notebooks:

   ```bash
   jupyter notebook notebooks/mfci.ipynb
   jupyter notebook notebooks/ml_model.ipynb
   ```

Notes and recommendations
- The `requirements.txt` lists core packages used by scripts and notebooks. Pin exact versions if you need reproducible environments.
- If you will run the notebooks interactively, install `jupyterlab` or `notebook` and `ipykernel` and create a kernel from the virtualenv.
- Some notebooks use `xgboost` which may require system packages on macOS (brew-installed `gcc`) for compilation; consider using the prebuilt wheel available via pip on modern macOS.

Data sources
Raw time-series data were obtained from the following sources:
- **FRED (Federal Reserve Economic Data)**: 
  - [DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU) — Brent crude oil prices
  - [FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) — Federal funds effective rate
  - [INDPRO](https://fred.stlouisfed.org/series/INDPRO) — Industrial production index
  - [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y) — 10-Year Treasury constant maturity minus 2-Year Treasury (term spread)
  - [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS) — CBOE volatility index (VIX)
- **OECD**: [Main Economic Indicators (MEI)](https://data.oecd.org/mei.htm) — accessed via [DBnomics](https://db.nomics.world/) API (see `scripts/oecd_mei_download.py`)
- **BIS (Bank for International Settlements)**: 
  - [Credit statistics](https://www.bis.org/statistics/totcredit.htm) — total credit to non-financial sector
  - [Real Effective Exchange Rates (REER)](https://www.bis.org/statistics/eer.htm)

All cleaned CSVs are stored under `data/cleaned data/`.
