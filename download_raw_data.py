import os
import yfinance as yf
import pandas as pd

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"
INTERVAL = "1d"

BASE_DATA_DIR = "data"

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
    os.makedirs(path, exist_ok=True)


def download_one_ticker(ticker: str) -> pd.DataFrame:
    print(f"Downloading {ticker} from {START_DATE} to {END_DATE}...")
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
        print(f"WARNING: no data returned for {ticker}")
    return df


def main():
    for group, tickers in ASSET_GROUPS.items():
        out_dir = os.path.join(BASE_DATA_DIR, group)
        ensure_dir(out_dir)

        for ticker in tickers:
            df = download_one_ticker(ticker)
            if df.empty:
                continue

            df = df.reset_index()
            fname = f"{ticker}.csv"
            out_path = os.path.join(out_dir, fname)
            df.to_csv(out_path, index=False)
            print(f"Saved {ticker} to {out_path}")


if __name__ == "__main__":
    main()
