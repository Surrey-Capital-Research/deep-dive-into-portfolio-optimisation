import pandas as pd

return_matrix = pd.read_csv(
    "bm_implementation/return_matrix.csv",
    index_col=0,
    parse_dates=True
)

def equal_weights(assets):
    n = len(assets)
    return pd.Series(1 / n, index=assets)

weights = equal_weights(return_matrix.columns)

print(weights)
print("Sum of weights:", weights.sum())
weights.to_csv("bm_implementation/weights.csv", header=["weights"])
