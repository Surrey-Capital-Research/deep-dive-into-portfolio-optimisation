"""
BL Sensitivity Analysis
=======================
Varies tau in [0.01, 0.05, 0.1, 0.25] and risk_aversion in [1, 2, 3, 5],
runs a full backtest for each combination, and produces a single heatmap
of total return styled to match the report.

Output: reports/images/plots/results/fig_bl_sensitivity.pdf
"""

import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.backtesting.backtester import Backtester
from src.backtesting.strategies import BlackLittermanStrategy
from src.models.mvo import make_mvo_optimiser
from src.models.bl_views import momentum_view_builder

warnings.filterwarnings("ignore")

TAU_VALUES = [0.01, 0.05, 0.1, 0.25]
RA_VALUES  = [1, 2, 3, 5]

OUTPUT_PDF = "reports/images/plots/results/fig_bl_sensitivity.pdf"


def run_grid(prices: pd.DataFrame, rfr: pd.Series) -> pd.DataFrame:
    tickers = list(prices.columns)
    market_weights = pd.Series(1.0 / len(tickers), index=tickers)

    records = []
    total = len(TAU_VALUES) * len(RA_VALUES)
    for i, (tau, ra) in enumerate(itertools.product(TAU_VALUES, RA_VALUES), 1):
        print(f"  [{i:2d}/{total}]  tau={tau:.2f}  risk_aversion={ra}")
        optimiser = make_mvo_optimiser(risk_free_rate=rfr)
        strategy = BlackLittermanStrategy(
            market_weights=market_weights,
            risk_aversion=ra,
            tau=tau,
            view_builder=momentum_view_builder,
            optimiser=optimiser,
        )
        result = Backtester(
            prices=prices,
            strategy=strategy,
            risk_free_rate=rfr,
        ).run()
        m = result.metrics
        records.append({
            "tau": tau,
            "risk_aversion": ra,
            "Sharpe": m["Sharpe"],
            "total_return": m["total_return"] * 100,   # percent
            "volatility": m["volatility"] * 100,
            "max_drawdown": m["max_drawdown"] * 100,
            "avg_monthly_turnover": m["avg_monthly_turnover"] * 100,
        })

    return pd.DataFrame(records)


STYLE = {
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.color":        "#dddddd",
}


def plot_heatmap(df: pd.DataFrame) -> None:
    plt.rcParams.update(STYLE)

    pivot = df.pivot(index="risk_aversion", columns="tau", values="total_return")
    pivot = pivot.sort_index(ascending=False)   # highest risk_aversion at top

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    im = ax.imshow(
        pivot.values,
        cmap="Blues",
        aspect="auto",
        vmin=pivot.values.min() * 0.85,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(t) for t in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index], fontsize=9)
    ax.set_xlabel(r"Prior uncertainty scalar $\tau$", fontsize=9)
    ax.set_ylabel(r"Risk-aversion coefficient $\delta$", fontsize=9)

    # Remove all four spines on heatmap (imshow makes them meaningless)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Annotate cells
    vmin, vmax = pivot.values.min(), pivot.values.max()
    for (r, c), val in np.ndenumerate(pivot.values):
        brightness = (val - vmin) / (vmax - vmin + 1e-10)
        text_color = "white" if brightness > 0.6 else "#222222"
        ax.text(
            c, r,
            f"{val:.1f}%",
            ha="center", va="center",
            fontsize=8.5,
            color=text_color,
            fontfamily="serif",
        )

    # Highlight baseline cell (tau=0.05, delta=3) with a bold box
    tau_idx = list(pivot.columns).index(0.05)
    ra_idx  = list(pivot.index).index(3)
    rect = plt.Rectangle(
        (tau_idx - 0.5, ra_idx - 0.5), 1, 1,
        linewidth=2.0, edgecolor="black", facecolor="none",
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=300)
    print(f"\nSaved: {OUTPUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...")
    prices = pd.read_csv(
        "data/uk_multi_asset_prices_clean.csv",
        index_col=0,
        parse_dates=True,
    ).ffill()

    rfr = pd.read_csv(
        "data/risk_free_rate.csv",
        index_col=0,
        parse_dates=True,
    ).squeeze() / 100

    print(f"Running {len(TAU_VALUES) * len(RA_VALUES)} BL backtests...\n")
    results_df = run_grid(prices, rfr)

    print("\nResults grid:")
    print(
        results_df.pivot_table(
            index="risk_aversion", columns="tau", values="Sharpe"
        ).to_string(float_format="{:.3f}".format)
    )

    print("\nGenerating heatmap figure...")
    plot_heatmap(results_df)

    print("\nDone.")
