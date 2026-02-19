import os
import glob
import logging
import pandas as pd
from functools import reduce

# Config logging for data ETL
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Updated dir structure for better data hygiene
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

SUBFOLDERS = {
    "equities": "equity",
    "bonds": "bond",
    "commodities": "commodity",
    "precious_metals_etc": "pm",
}

def load_and_prepare_csv(path: str) -> pd.DataFrame:
    """
    Parses individual asset CSVs and standardizes the schema.
    Extracts 'Date' and 'Close' columns, mapping filenames to tickers.
    """
    df = pd.read_csv(path, header=0)
    
    # Standardise date column and ticker naming
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.rename(columns={df.columns[0]: "Date"})
    
    ticker = os.path.splitext(os.path.basename(path))[0]

    # Map specific index tickers to human-readable names
    if ticker == "^v30082.L":
        ticker = "UK_10Y_Yield"

    # We only care about the adjusted close for backtesting
    df_out = pd.DataFrame({
        "Date": df["Date"],
        ticker: df["Close"],
    })
    return df_out

def main():
    """
    Master ETL process: Scans raw subdirectories, standardizes price series,
    merges into a single matrix, and exports to the processed directory.
    """
    all_price_dfs = []
    
    for sub in SUBFOLDERS.keys():
        folder = os.path.join(RAW_DATA_DIR, sub)
        pattern = os.path.join(folder, "*.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            logger.warning(f"No source files found in {folder}")
            continue

        for f in csv_files:
            try:
                df = load_and_prepare_csv(f)
                all_price_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process {f}: {e}")

    if not all_price_dfs:
        raise RuntimeError("ETL Failed: No data found in data/raw structure.")

    # Outer join on Date ensures we don't drop days where some assets didn't trade
    logger.info("Merging asset classes into master price matrix...")
    merged = reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"),
        all_price_dfs,
    )

    # Standardise index and sort chronologically
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.set_index("Date").sort_index()
    
    # Production time-horizon filter
    start = pd.Timestamp("2015-01-01")
    end = pd.Timestamp("2026-01-01")
    merged = merged[(merged.index >= start) & (merged.index <= end)]
    
    # Drop rows with zero obs
    merged = merged.dropna(how="all")
    
    # Ensure processed dir exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DATA_DIR, "uk_multi_asset_prices_clean.csv")
    
    merged.to_csv(out_path, index=True)
    logger.info(f"ETL Complete: Matrix saved to {out_path} (Shape: {merged.shape})")

if __name__ == "__main__":
    main()