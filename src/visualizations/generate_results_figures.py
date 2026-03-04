"""
Generate all results figures for the portfolio optimisation report.
Runs all four backtests once and saves all figures to reports/images/plots/.

Run from project root:
    poetry run python src/visualizations/generate_results_figures.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.scripts.run_equal_weight_backtest import run as run_equal_weight
from src.scripts.run_risk_parity_backtest import run as run_risk_parity
from src.scripts.run_mvo_backtest import run as run_mvo
from src.scripts.run_bl_backtest import run as run_bl
from src.backtesting.backtester import BacktestResult


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/uk_multi_asset_prices_clean.csv"
RFR_PATH  = "data/risk_free_rate.csv"
SAVE_DIR  = "reports/images/plots/results"

COLOURS = {
    "Equal Weight":    "#104E8B",
    "MVO":             "#C0392B",
    "Black-Litterman": "#2E8B57",
    "Risk Parity":     "#D35400",
}

REGIMES = {
    "Brexit\n(Jun-Dec 2016)":     ("2016-06-23", "2016-12-31"),
    "COVID-19\n(Feb-Sep 2020)":   ("2020-02-19", "2020-09-30"),
    "Rate Hikes\n(Jan-Dec 2022)": ("2022-01-01", "2022-12-31"),
}

STYLE = {
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.color":        "#dddddd",
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   8.5,
}

pct = mticker.FuncFormatter(lambda v, _: f"{v:.0%}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).ffill()
    rfr    = pd.read_csv(RFR_PATH,  index_col=0, parse_dates=True).squeeze() / 100
    return prices, rfr  # type: ignore


def run_all(prices: pd.DataFrame, rfr: pd.Series) -> dict[str, BacktestResult]:
    print("Running backtests — this may take a moment...")
    results = {
        "Equal Weight":    run_equal_weight(prices),
        "MVO":             run_mvo(prices, rfr),
        "Black-Litterman": run_bl(prices, rfr),
        "Risk Parity":     run_risk_parity(prices),
    }
    print("Backtests complete.\n")
    return results


def compute_weights(result: BacktestResult, prices: pd.DataFrame) -> pd.DataFrame:
    """Convert position units to weight fractions using daily prices."""
    pos      = result.positions.astype(float)
    eq       = result.equity_curve.reindex(pos.index)
    p_aln    = prices.reindex(pos.index).ffill()
    holdings = pos.multiply(p_aln)
    weights  = holdings.divide(eq, axis=0).clip(0, 1)
    return weights


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def rolling_sharpe(equity: pd.Series, window: int = 252) -> pd.Series:
    rets = equity.pct_change().dropna()
    return (rets.rolling(window).mean() * 252) / (rets.rolling(window).std() * np.sqrt(252))




def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(SAVE_DIR, f"{name}.pdf")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)

# ── Figure 3: Cumulative Wealth ───────────────────────────────────────────────
def fig_cumulative_wealth(results: dict[str, BacktestResult]) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for name, res in results.items():
        ax.plot(res.equity_curve, color=COLOURS[name], lw=1.5, label=name)

    # Regime shading
    regime_style = [
        ("2016-06-23", "2016-12-31", "Brexit",     "#3082DA"),
        ("2020-02-19", "2020-09-30", "COVID-19",   "#3082DA"),
        ("2022-01-01", "2022-12-31", "Rate Hikes", "#3082DA"),
    ]
    y_bot = ax.get_ylim()[0]
    for start, end, label, colour in regime_style:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.10, color=colour, zorder=0)
        ax.text(pd.Timestamp(start), y_bot * 1.06, label,
                fontsize=7, color=colour, va="top", ha="left",
                fontstyle="italic")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"£{v/1_000:.0f}k")
    )
    ax.set_ylabel("Portfolio Value", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig3_cumulative_wealth")


# ── Figure 4: Drawdown ────────────────────────────────────────────────────────
def fig_drawdown(results: dict[str, BacktestResult]) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for name, res in results.items():
        dd = drawdown_series(res.equity_curve)
        ax.plot(dd, color=COLOURS[name], lw=1.2, label=name)
        ax.fill_between(dd.index, dd, 0, alpha=0.07, color=COLOURS[name])

    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Drawdown", fontsize=10)
    ax.set_ylim(top=0.01)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig4_drawdown")


# ── Figure 5: Risk-Return Scatter ─────────────────────────────────────────────
def fig_risk_return(results: dict[str, BacktestResult], rfr: pd.Series) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    avg_rf = float(rfr.mean())

    # Sharpe isocurves: μ_p = r_f + SR × σ  →  straight lines in (σ, μ_p) space
    vol_range = np.linspace(0.06, 0.20, 300)
    for sr in [0.1, 0.2, 0.3, 0.4]:
        y = avg_rf + sr * vol_range
        ax.plot(vol_range, y, color="#cccccc", lw=0.9, linestyle="--", zorder=1)
        ax.text(vol_range[-1] + 0.001, y[-1], f"SR = {sr:.1f}",
                fontsize=7, color="#aaaaaa", va="center")

    # Strategy dots (use annualised arithmetic mean return for consistency with isocurves)
    for name, res in results.items():
        rets     = res.equity_curve.pct_change().dropna()
        mean_ret = float(rets.mean() * 252)
        vol      = res.metrics["volatility"]
        ax.scatter(vol, mean_ret, color=COLOURS[name], s=80, zorder=4, linewidths=0)
        ax.annotate(name, (vol, mean_ret),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=COLOURS[name])

    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Annualised Volatility", fontsize=10)
    ax.set_ylabel("Annualised Return", fontsize=10)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    save(fig, "fig5_risk_return_scatter")


# ── Figure 6: Rolling Sharpe ──────────────────────────────────────────────────
def fig_rolling_sharpe(results: dict[str, BacktestResult]) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for name, res in results.items():
        rs = rolling_sharpe(res.equity_curve)
        ax.plot(rs, color=COLOURS[name], lw=1.2, label=name)

    ax.axhline(0, color="#555555", lw=0.8, linestyle="--")
    ax.set_ylabel("Rolling Sharpe (12-month)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig6_rolling_sharpe")


# ── Figure 7a–c: Regime Analysis (one figure per regime) ─────────────────────
def _regime_fig(results: dict[str, BacktestResult],
                title: str, start: str, end: str, fname: str) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for name, res in results.items():
        eq = res.equity_curve.loc[start:end]
        if eq.empty:
            continue
        norm = eq / eq.iloc[0] * 100
        ax.plot(norm, color=COLOURS[name], lw=1.4, label=name)

    ax.axhline(100, color="#aaaaaa", lw=0.7, linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Indexed Return (100 = start)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    if fname == "fig7c_regime_2022":
        ax.legend(fontsize=7.5, loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, fname)


def fig_regimes(results: dict[str, BacktestResult]) -> None:
    fnames = ["fig7a_regime_brexit", "fig7b_regime_covid", "fig7c_regime_2022"]
    titles = ["","",""]
    for fname, title, (_, (start, end)) in zip(fnames, titles, REGIMES.items()):
        _regime_fig(results, title, start, end, fname)


def print_regime_returns(results: dict[str, BacktestResult]) -> None:
    col_w = 24
    header = f"{'Strategy':<20}" + "".join(f"{t:>{col_w}}" for t in [
        "Brexit", "COVID-19", "Rate Hikes 2022"
    ])
    print("\nCumulative Returns by Regime:")
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        row = f"{name:<20}"
        for _, (start, end) in REGIMES.items():
            eq = res.equity_curve.loc[start:end]
            if eq.empty or len(eq) < 2:
                row += f"{'N/A':>{col_w}}"
            else:
                cum_ret = eq.iloc[-1] / eq.iloc[0] - 1
                row += f"{cum_ret:>{col_w}.2%}"
        print(row)
    print()


# ── Figure 8: Allocation Heatmaps ─────────────────────────────────────────────
def fig_heatmaps(results: dict[str, BacktestResult], prices: pd.DataFrame) -> None:
    plt.rcParams.update(STYLE)
    to_plot = ["MVO", "Black-Litterman", "Risk Parity"]

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))

    for ax, name in zip(axes, to_plot):
        weights = compute_weights(results[name], prices)
        monthly = weights.resample("ME").last().T      # assets × months

        im = ax.imshow(
            monthly.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=0.6,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(monthly.index)))
        ax.set_yticklabels(monthly.index, fontsize=6.5)
        ax.set_title(name, fontsize=10, pad=4)

        # Year ticks on x-axis
        dates     = list(monthly.columns)
        year_pos  = [i for i, d in enumerate(dates) if d.month == 1]
        year_lbl  = [dates[i].year for i in year_pos]
        ax.set_xticks(year_pos)
        ax.set_xticklabels(year_lbl, fontsize=8)

        cbar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
        cbar.ax.yaxis.set_major_formatter(pct)
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "fig8_allocation_heatmaps")


# ── Figure 9: Rolling Monthly Turnover ────────────────────────────────────────
def fig_turnover(results: dict[str, BacktestResult], prices: pd.DataFrame) -> None:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    for name, res in results.items():
        weights  = compute_weights(res, prices)
        monthly  = weights.resample("ME").last()
        turnover = monthly.diff().abs().sum(axis=1).dropna()
        ax.plot(turnover, color=COLOURS[name], lw=1.2, label=name, alpha=0.9)

    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Monthly Turnover", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig9_turnover")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    prices, rfr = load_data()
    results     = run_all(prices, rfr)

    print_regime_returns(results)

    print("Generating figures...")
    fig_cumulative_wealth(results)
    fig_drawdown(results)
    fig_risk_return(results, rfr)
    fig_rolling_sharpe(results)
    fig_regimes(results)
    fig_turnover(results, prices)
    print("\nAll figures saved to", SAVE_DIR)


if __name__ == "__main__":
    main()
