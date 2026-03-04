"""
Illustrative efficient frontier for the theory section.
Uses synthetic assets so the curve is clean and textbook-shaped.
Run from project root: poetry run python src/visualizations/plot_efficient_frontier.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Synthetic assets ─────────────────────────────────────────────────────────
# σ ≈ 7%, 12%, 18%, 28% — same correlation structure, scaled down and shifted
MU = np.array([0.02, 0.08, 0.11, 0.10])

COV = np.array([
    [0.0049, 0.0002, 0.0001, 0.0001],
    [0.0002, 0.0744, 0.0008, 0.0002],
    [0.0001, 0.0008, 0.0224, 0.0008],
    [0.0001, 0.0002, 0.0008, 0.0400],
])

RF   = 0.01
SAVE = "reports/images/plots/theory/efficient_frontier"


def _frontier(mu, cov, n_points=400):
    """Analytical unconstrained mean-variance frontier."""
    cov_inv = np.linalg.inv(cov)
    ones    = np.ones(len(mu))

    a = ones @ cov_inv @ ones
    b = ones @ cov_inv @ mu
    c = mu   @ cov_inv @ mu
    D = a * c - b ** 2

    mu_min  = b / a
    mu_grid = np.linspace(mu_min, mu.max() * 1.10, n_points)

    sigmas = []
    for mp in mu_grid:
        lam = (a * mp - b) / D
        gam = (c - b * mp) / D
        w   = lam * (cov_inv @ mu) + gam * (cov_inv @ ones)
        sigmas.append(np.sqrt(max(float(w @ cov @ w), 0.0)))

    return np.array(sigmas), mu_grid


def _gmv(cov):
    cov_inv = np.linalg.inv(cov)
    ones    = np.ones(cov.shape[0])
    w       = (cov_inv @ ones) / (ones @ cov_inv @ ones)
    return w


def _max_sharpe(mu, cov, rf):
    cov_inv = np.linalg.inv(cov)
    z       = cov_inv @ (mu - rf)
    return z / z.sum()


def plot(save_path=SAVE):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sigma_f, mu_f = _frontier(MU, COV)

    w_gmv     = _gmv(COV)
    mu_gmv    = w_gmv @ MU
    sigma_gmv = np.sqrt(w_gmv @ COV @ w_gmv)

    w_msr     = _max_sharpe(MU, COV, RF)
    mu_msr    = w_msr @ MU
    sigma_msr = np.sqrt(w_msr @ COV @ w_msr)

    # Random feasible portfolios (long-only)
    rng      = np.random.default_rng(0)
    rand_w   = rng.dirichlet(np.ones(len(MU)), size=4000)
    r_sig    = np.sqrt(np.einsum('ij,jk,ik->i', rand_w, COV, rand_w))
    r_mu     = rand_w @ MU
    r_sharpe = (r_mu - RF) / r_sig          # colour by Sharpe ratio

    # ── Style ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'serif',
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'axes.linewidth':   0.8,
        'grid.linewidth':   0.5,
        'grid.color':       '#dddddd',
    })

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Feasible set — coloured by Sharpe ratio
    sc = ax.scatter(r_sig, r_mu, c=r_sharpe, cmap='YlGnBu',
                    s=8, alpha=0.55, linewidths=0,
                    label='Feasible portfolios', zorder=1)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('Sharpe Ratio', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Efficient frontier
    ax.plot(sigma_f, mu_f, color='#104E8B', linewidth=2,
            label='Efficient frontier', zorder=3)

    # GMV
    ax.scatter(sigma_gmv, mu_gmv, s=80, color='#2E8B57',
               zorder=5, label='Min-variance portfolio', edgecolors='white', linewidths=0.8)

    # Max Sharpe
    ax.scatter(sigma_msr, mu_msr, s=80, color='#C0392B',
               zorder=5, label='Max Sharpe portfolio', edgecolors='white', linewidths=0.8)

    # Capital Market Line — from RF intercept to x=0.20
    cml_x = np.array([0, 0.15])
    cml_y = RF + (mu_msr - RF) / sigma_msr * cml_x
    ax.plot(cml_x, cml_y, color='#C0392B', linewidth=1, linestyle='--',
            alpha=0.6, label='Capital market line', zorder=2)

    # Risk-free rate — dot on y-axis with label
    ax.scatter(0, RF, s=50, color='#C0392B', zorder=6, clip_on=False)
    ax.text(0.002, RF - 0.0075, '$r_f$', fontsize=10, color='#C0392B', va='bottom')

    pct = mticker.FuncFormatter(lambda x, _: f'{x:.1%}')
    ax.set_yticks([0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15])
    ax.set_xticks([0.00, 0.05, 0.10, 0.15, 0.20])

    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel('Volatility $\\sigma_p$', fontsize=11)
    ax.set_ylabel('Expected Return $\\mu_p$', fontsize=11)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9,
              edgecolor='#cccccc', loc='upper left')
    ax.grid(True, axis='y')

    ax.set_xlim(0.0, 0.20)
    ax.set_ylim(0.0, 0.16)
    fig.tight_layout()
    fig.savefig(f'{save_path}.pdf', dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}.{{pdf}}")
    plt.show()


if __name__ == '__main__':
    plot()
