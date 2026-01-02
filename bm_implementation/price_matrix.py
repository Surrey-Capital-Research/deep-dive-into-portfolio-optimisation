import pandas as pd
import os

raw_data_path = "data"
output_path = "price_matrix.csv"
asset_files = {
    "IGLT.L": "data/bonds/IGLT.L.csv",
    "AIGC.L": "data/commodities/AIGC.L.csv",
    "AZN.L": "data/equities/AZN.L.csv",
    "BARC.L": "data/equities/BARC.L.csv",
    "BP.L": "data/equities/BP.L.csv",
    "DGE.L": "data/equities/DGE.L.csv",
    "GLEN.L": "data/equities/GLEN.L.csv",
    "GSK.L": "data/equities/GSK.L.csv",
    "HSBA.L": "data/equities/HSBA.L.csv",
    "LLOY.L": "data/equities/LLOY.L.csv",
    "NG.L": "data/equities/NG.L.csv",
    "RIO.L": "data/equities/RIO.L.csv",
    "RR.L": "data/equities/RR.L.csv",
    "SHEL.L": "data/equities/SHEL.L.csv",
    "TSCO.L": "data/equities/TSCO.L.csv",
    "ULVR.L": "data/equities/ULVR.L.csv",
    "VOD.L": "data/equities/VOD.L.csv",
    "SGLD.L": "data/precious_metals_etc/SGLD.L.csv"
}

price_df = pd.DataFrame()

for asset, filepath in asset_files.items():

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")

    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    price_df[asset] = df["Adj Close"]

price_df = price_df.sort_index()
price_df = price_df.iloc[:-1]
price_df.to_csv(output_path)

print("Price matrix saved to:", output_path)

