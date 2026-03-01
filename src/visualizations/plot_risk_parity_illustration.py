"""
Illustrative risk parity scatter plot for the theory section.
X-axis: annualised variance. Y-axis: risk parity weight.
Shows the inverse relationship between asset variance and RP weight.
Run from project root: poetry run python src/visualizations/plot_risk_parity_illustration.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Synthetic assets, ordered by decreasing volatility ────────────────────────
LABELS = ["Asset A", "Asset B", "Asset C", "Asset D"]
VOLS   = np.array([0.30, 0.24, 0.18, 0.14])
N      = len(VOLS)

VAR    = VOLS**2 * 100
inv_vol = 1 / VOLS
W_RP   = inv_vol / inv_vol.sum() * 100

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.color":        "#e0e0e0",
})

BLUE  = "#104E8B"

fig, ax = plt.subplots(figsize=(6, 4.5))

# Smooth inverse-vol reference curve
vol_range  = np.linspace(0.05, 0.40, 200)
var_range  = vol_range**2 * 100
w_curve    = (1 / vol_range) / inv_vol.sum() * 100

ax.plot(var_range, w_curve, color="#4b4848", linewidth=1.2,
        linestyle="--", zorder=2, label="Inverse-vol curve")

for i, label in enumerate(LABELS):
    ax.scatter(VAR[i], W_RP[i], s=80, color=BLUE,
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.text(VAR[i] + 0.4, W_RP[i] + 1, label, fontsize=8.5,
            va="center", color="#333333")

ax.set_xlabel("Annualised Variance (%)", fontsize=10)
ax.set_ylabel("Risk Parity Weight (%)", fontsize=10)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.legend(fontsize=8.5, frameon=False)

# ── Save ──────────────────────────────────────────────────────────────────────
SAVE = "reports/images/plots/theory/risk_parity_illustration"
os.makedirs(os.path.dirname(SAVE), exist_ok=True)

fig.savefig(f"{SAVE}.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{SAVE}.png", bbox_inches="tight", dpi=150)
print(f"Saved: {SAVE}.pdf / .png")

plt.close(fig)
