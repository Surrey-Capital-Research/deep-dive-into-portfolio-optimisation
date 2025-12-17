import os
import glob
import pandas as pd
from functools import reduce

DATA_DIR = "data"

SUBFOLDERS = {
    "equities": "equity",
    "bonds": "bond",
    "commodities": "commodity",
    "precious_metals_etc": "pm",
}


def load_and_prepare_csv(path: str) -> pd.DataFrame:
    """Read one CSV, extract Date and Close, rename Close to ticker."""
    print(f"DEBUG reading {path}")
    df = pd.read_csv(path, header=0)
    print("DEBUG columns:", list(df.columns))
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.rename(columns={df.columns[0]: "Date"})
    ticker = os.path.splitext(os.path.basename(path))[0]
    close_series = df["Close"]

    df_out = pd.DataFrame(
        {
            "Date": df["Date"],
            ticker: close_series,
        }
    )
    return df_out


def main():
    all_price_dfs = []
    for sub in SUBFOLDERS.keys():
        folder = os.path.join(DATA_DIR, sub)
        pattern = os.path.join(folder, "*.csv")
        csv_files = glob.glob(pattern)

        if not csv_files:
            print(f"No CSV files found in {folder}, skipping.")
            continue

        for f in csv_files:
            print(f"Loading {f}")
            df = load_and_prepare_csv(f)
            all_price_dfs.append(df)

    if not all_price_dfs:
        raise RuntimeError(
            "No CSV files loaded; check your data/raw structure and filenames."
        )

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"),
        all_price_dfs,
    )

    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.set_index("Date")
    merged = merged.sort_index()
    print("DEBUG merged index min/max:", merged.index.min(), merged.index.max())
    print("DEBUG merged shape before date filter:", merged.shape)
    start = pd.Timestamp("2015-01-01")
    end = pd.Timestamp("2025-01-01")
    merged = merged[(merged.index >= start) & (merged.index <= end)]
    print("DEBUG merged shape after date filter:", merged.shape)
    merged = merged.dropna(how="all")
    print("DEBUG merged shape after dropna(all):", merged.shape)
    out_path = os.path.join(DATA_DIR, "uk_multi_asset_prices_clean.csv")
    merged.to_csv(out_path, index=True)
    print(f"Saved clean price matrix to {out_path}")


if __name__ == "__main__":
    main()
