import os
import yfinance as yf
import pandas as pd
import logging

# Config for terminal feedback during data download
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Global horizons
START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"

# Ensure raw data is isolated from processed output
RAW_DATA_DIR = os.path.join("data", "raw")

# Multi-asset universe
ASSET_GROUPS = {
    "equities": [
        "HSBA.L", "LLOY.L", "BARC.L", "SHEL.L", "BP.L",
        "ULVR.L", "TSCO.L", "DGE.L", "AZN.L", "GSK.L",
        "RIO.L", "GLEN.L", "NG.L", "VOD.L", "RR.L",
    ],
    "bonds": [
        "IGLT.L",
        "^v30082.L",      
    ],
    "precious_metals_etc": [
        "SGLD.L",      
    ],
    "commodities": [
        "AIGC.L",      
    ],
}

def ensure_dir(path: str):
    """Helper for dir creation."""
    os.makedirs(path, exist_ok=True)

def download_one_ticker(ticker: str) -> pd.DataFrame:
    """Wrapper for yfinance download with standard parameters."""
    logger.info(f"Fetching {ticker} via yfinance API...")
    
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=False,   
        progress=False,
        actions=False,
    )
    
    if df.empty:
        logger.warning(f"No data returned for {ticker}. Check ticker symbol/delisting status.")
    return df

def main():
    """Batch download process for the asset universe."""
    for group, tickers in ASSET_GROUPS.items():
        # Group-based directory structure
        out_dir = os.path.join(RAW_DATA_DIR, group)
        ensure_dir(out_dir)

        for ticker in tickers:
            df = download_one_ticker(ticker)
            if df.empty:
                continue

            # Flatten index to ensure 'Date' is a column for the ETL step
            df = df.reset_index()
            
            # Sanitise filename 
            safe_ticker = ticker.replace("^", "")
            fname = f"{safe_ticker}.csv"
            out_path = os.path.join(out_dir, fname)
            
            df.to_csv(out_path, index=False)
            logger.info(f"Buffered {ticker} to {out_path}")

if __name__ == "__main__":
    main()