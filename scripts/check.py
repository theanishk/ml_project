import pandas as pd
import os


def check_coverage():
    print("--- DIAGNOSING DATA COVERAGE ---")

    # 1. Load the data
    file_path = os.path.join("..", "data", "raw data", "oecd_mei_final.csv")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Run the download script first.")
        return

    # 2. Pivot to see variables side-by-side
    # We want to know: For each Country, is the variable present?
    # We count the number of non-null months.
    coverage = df.pivot_table(
        index="Country",
        columns="Variable",
        values="Value",
        aggfunc="count",  # Counts valid months
    )

    # 3. Clean up the view
    # Fill NaN with 0 (meaning 0 months of data)
    coverage = coverage.fillna(0).astype(int)

    # 4. Add a "Completeness" Score
    # Total distinct variables found for that country (max 8)
    # We check if the column > 0 (meaning at least some data exists)
    coverage["Variables_Found"] = (coverage > 0).sum(axis=1)

    # Sort by best coverage first
    coverage = coverage.sort_values(by="Variables_Found", ascending=False)

    # 5. PRINT THE REPORT
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 1000)

    print(f"\nTotal Countries Scanned: {len(coverage)}")
    print("-" * 80)
    print("  DATA AVAILABILITY MATRIX (Months of Data Found)")
    print("-" * 80)
    print(coverage)
    print("-" * 80)

    # 6. RECOMENDATION ENGINE
    # We need countries with at least the 'Core 5' + maybe confidence
    good_countries = coverage[coverage["Variables_Found"] >= 7].index.tolist()

    print(
        f"\n✅ RECOMMENDATION: Keep these {len(good_countries)} countries (High Data Quality):"
    )
    print(good_countries)

    bad_countries = coverage[coverage["Variables_Found"] < 5].index.tolist()
    print(
        f"\n❌ WARNING: Drop these {len(bad_countries)} countries (Too much missing data):"
    )
    print(bad_countries)


if __name__ == "__main__":
    check_coverage()
