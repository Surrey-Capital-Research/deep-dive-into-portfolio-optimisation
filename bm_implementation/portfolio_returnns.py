import pandas as pd
import numpy as np

returns = pd.read_csv("bm_implementation/return_matrix.csv", index_col=0, parse_dates=True)
weights_df = pd.read_csv("bm_implementation/weights.csv", index_col=0)
weights = weights_df.iloc[:, 0].values

assert returns.shape[1] == len(weights)
assert np.isclose(weights.sum(), 1) 

monthly_groups = returns.groupby(returns.index.to_period("M"))

portfolio_returns = []
dates = [] 

for _, monthly_returns in monthly_groups:
    w = weights.copy()
    for date, daily_returns in monthly_returns.iterrows():
        port_ret = np.dot(daily_returns.values, w)
        portfolio_returns.append(port_ret)
        dates.append(date)
        w = w * (1 + daily_returns.values)
        w = w / w.sum()
      

portfolio_returns = pd.Series(
    portfolio_returns,
    index=pd.to_datetime(dates),
    name="portfolio_return"
)

portfolio_returns.to_csv("portfolio_returns_rebalanced.csv")

print("Monthly rebalanced portfolio returns saved")

