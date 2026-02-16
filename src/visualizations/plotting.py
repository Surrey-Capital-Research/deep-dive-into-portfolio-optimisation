import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def format_pct_axis(x, pos):
    """Help func to turn 0.1 into 10%"""
    return f'{x:.0%}'

def create_professional_tearsheet(results: dict, benchmark_prices: pd.Series = None, title: str = "Strategy Performance Audit"):
    """
    Generates the main dashboard (Tearsheet) for the report.
    It plots the Equity Curve, Drawdowns, Monthly Heatmap, and Risk metrics all in one go.
    """
    
    # 1. Data wrangling
    # Grab equity curve from the results dict
    equity = results['equity_curve']
    equity.index = pd.to_datetime(equity.index)
    
    # Need daily returns for the vol and distr plots
    returns = equity.pct_change().fillna(0)
    
    # Calc Drawdown series
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    
    # Compare dates to benchmark
    bench_equity = None
    if benchmark_prices is not None:
        benchmark_prices.index = pd.to_datetime(benchmark_prices.index)
        
        # Forward fill benchmark to match our strats timeline
        bench_aligned = benchmark_prices.reindex(equity.index, method='ffill')
        bench_ret = bench_aligned.pct_change().fillna(0)
        
        # Normalize benchmark to start at the same $$ amount as our strategy
        bench_equity = (1 + bench_ret).cumprod() * equity.iloc[0]

    # Plot styling
    # Try to use the seaborn style, fallback if libr is old
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('seaborn-whitegrid') 
    
    # Setting up the grid layout: 3 rows, 2 columns
    # Row 1 is the big main chart (Equity Curve)
    # Row 2 is Drawdown & Volatility
    # Row 3 is the Heatmap & Histogram
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.95, color='#333333')
    
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 2, 2], hspace=0.3, wspace=0.2)

    # Main Chart: Equity Curve
    ax_equity = fig.add_subplot(gs[0, :]) # Spans both columns
    ax_equity.plot(equity.index, equity, color='#104E8B', linewidth=2.5, label='Strategy')
    
    # Add benchmark line if available
    if bench_equity is not None:
        ax_equity.plot(bench_equity.index, bench_equity, color='#888888', linestyle='--', linewidth=1.5, alpha=0.8, label='Benchmark')

    ax_equity.set_title("Cumulative Wealth", fontsize=14, fontweight='bold', loc='left')
    ax_equity.set_ylabel("Portfolio Value (£)", fontsize=12)
    ax_equity.legend(loc='upper left', frameon=True)
    ax_equity.grid(True, alpha=0.3)
    
    # Nice little shaded area under the curve to make it look modern
    ax_equity.fill_between(equity.index, equity, equity.iloc[0], color='#104E8B', alpha=0.05)

    # Risk Chart: Drawdown
    ax_dd = fig.add_subplot(gs[1, 0])
    ax_dd.plot(drawdown.index, drawdown, color='#CD3333', linewidth=1.5)
    # Fill the 'underwater' area red
    ax_dd.fill_between(drawdown.index, drawdown, 0, color='#CD3333', alpha=0.2)
    
    ax_dd.set_title("Drawdown (Risk)", fontsize=14, fontweight='bold', loc='left')
    ax_dd.set_ylabel("% from Peak", fontsize=12)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))
    ax_dd.grid(True, alpha=0.3)

    # Volatility chart
    ax_vol = fig.add_subplot(gs[1, 1])
    # Calculate 30-day rolling vol, annualised
    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    
    ax_vol.plot(rolling_vol.index, rolling_vol, color='#FF8C00', linewidth=1.5)
    ax_vol.set_title("30-Day Rolling Volatility (Annualized)", fontsize=14, fontweight='bold', loc='left')
    ax_vol.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))
    ax_vol.grid(True, alpha=0.3)

    # Heatmap
    ax_heat = fig.add_subplot(gs[2, 0])
    
    # Resample to monthly returns for the heatmap
    monthly_rets = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Pivot table so Years are rows and Months are columns
    monthly_rets_df = pd.DataFrame({
        'Year': monthly_rets.index.year,
        'Month': monthly_rets.index.month,
        'Return': monthly_rets.values
    })
    heatmap_data = monthly_rets_df.pivot(index='Year', columns='Month', values='Return')
    
    # Plot it
    sns.heatmap(heatmap_data, annot=True, fmt='.1%', center=0, cmap='RdYlGn', 
                cbar=False, ax=ax_heat, linewidths=0.5, linecolor='white')
    ax_heat.set_title("Monthly Returns %", fontsize=14, fontweight='bold', loc='left')
    ax_heat.set_ylabel("")

    # Dist Histogram
    ax_dist = fig.add_subplot(gs[2, 1])
    sns.histplot(returns, bins=50, kde=True, color='#2E8B57', ax=ax_dist, stat='density')
    
    # Dashed line for the 95% VaR 
    var_95 = returns.quantile(0.05)
    ax_dist.axvline(var_95, color='red', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2%}')
    
    ax_dist.set_title("Daily Return Distribution", fontsize=14, fontweight='bold', loc='left')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.2)

    # Stats Box
    # Recalc these here just to print them on the chart
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / ann_vol if ann_vol > 0 else 0 # simple 2% risk-free assumption
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    stats_text = (
        f"Total Return: {total_ret:.2%}\n"
        f"CAGR:         {cagr:.2%}\n"
        f"Sharpe Ratio: {sharpe:.2f}\n"
        f"Annual Vol:   {ann_vol:.2%}\n"
        f"Max Drawdown: {max_dd:.2%}\n"
        f"Calmar Ratio: {calmar:.2f}\n"
    )
    
    # Text box in the top right corner of the main plot
    ax_equity.text(0.98, 0.05, stats_text, transform=ax_equity.transAxes, 
                   fontsize=11, fontfamily='monospace', verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

    # Save it
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Made room for the main title
    
    filename = "results/tearsheet.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Tearsheet generated and saved to {filename}")