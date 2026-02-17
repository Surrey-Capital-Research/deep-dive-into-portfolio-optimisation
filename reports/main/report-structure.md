# Report Structure: A Deep Dive into Portfolio Optimisation

This document outlines the structure for the final research report.

**Estimated Length:** 12–16 pages (two-column format)

---

## 1. Introduction

### 1.1 The Portfolio Allocation Problem

- Brief context: investors must allocate across assets
- The promise of optimisation (Markowitz) vs the reality (estimation error)
- One paragraph, sets up the tension

### 1.2 Research Questions

1. Do optimisation models outperform naive 1/N out-of-sample?
2. Which model offers the best risk-adjusted performance?
3. How do models perform during market stress?

### 1.3 Summary of Findings

State the main result upfront. Example:

> "We find that no optimisation model consistently outperforms equal weighting. Risk Parity achieves lower volatility but sacrifices returns, particularly during the 2022 bond crisis when its structural overweight to fixed income proved costly."

### 1.4 Structure of Report

One paragraph roadmap.

---

## 2. Literature and Theoretical Background

### 2.1 Notation

Establish once, use throughout:

| Symbol                                 | Meaning              |
| -------------------------------------- | -------------------- |
| $w \in \mathbb{R}^n$                 | Portfolio weights    |
| $\mu \in \mathbb{R}^n$               | Expected returns     |
| $\Sigma \in \mathbb{R}^{n \times n}$ | Covariance matrix    |
| $r_f$                                | Risk-free rate       |
| $\sigma_p = \sqrt{w^\top \Sigma w}$  | Portfolio volatility |
| $\mathbf{1}$                         | Identity             |

---

### 2.2 The 1/N Puzzle

DeMiguel, Garlappi, and Uppal (2009) tested 14 optimisation models across 7 datasets. None consistently outperformed:

$$
w_i^{EW} = \frac{1}{n}
$$

Why? Estimation error. Sample estimates satisfy:

$$
\hat{\mu} - \mu = O\left(\frac{1}{\sqrt{T}}\right)
$$

With monthly data and realistic return distributions, ~500 years are required for statistically reliable estimates. Optimisation amplifies these errors.

---

### 2.3 Mean-Variance Optimisation

**Origins:** Markowitz (1952) formalised the risk-return trade-off, establishing Modern Portfolio Theory.

**The Problem:**

$$
\max_{w \in \mathcal{W}} \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
$$

**Derivation:** Ignoring the non-negativity constraint, form the Lagrangian for the equivalent problem of minimising variance for a target return $\mu_p$:

$$
\mathcal{L}(w, \lambda, \gamma) = \frac{1}{2} w^\top \Sigma w - \lambda (w^\top \mu - \mu_p) - \gamma (\mathbf{1}^\top w - 1)
$$

First-order conditions:

$$
\frac{\partial \mathcal{L}}{\partial w} = \Sigma w - \lambda \mu - \gamma \mathbf{1} = 0
$$

Solving:

$$
w^* = \lambda \Sigma^{-1} \mu + \gamma \Sigma^{-1} \mathbf{1}
$$

Applying constraints yields the **two-fund theorem**: all efficient portfolios are combinations of:

$$
w_A = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}, \quad w_B = \frac{\Sigma^{-1} \mu}{\mathbf{1}^\top \Sigma^{-1} \mu}
$$

The **maximum Sharpe ratio portfolio** is:

$$
w^{MSR} = \frac{\Sigma^{-1}(\mu - r_f \mathbf{1})}{\mathbf{1}^\top \Sigma^{-1}(\mu - r_f \mathbf{1})}
$$

**The Problem in Practice:** Michaud (1989) showed MVO is an "error maximiser" — it overweights assets with overestimated returns and underweights those with underestimated returns.

---

### 2.4 Black-Litterman Model

**Origins:** Developed at Goldman Sachs by Black and Litterman (1992) to address MVO instability.

**Key Insight:** Anchor expected returns to market equilibrium, then adjust with investor views using Bayesian inference.

**Step 1: Equilibrium Returns**

Assume CAPM holds. Implied equilibrium returns are:

$$
\pi = \delta \Sigma w_m
$$

where $w_m$ is the market-cap weight vector and $\delta$ is risk aversion:

$$
\delta = \frac{\mathbb{E}[r_m] - r_f}{\sigma_m^2}
$$

**Step 2: Express Views**

Views take the form:

$$
P \mu = Q + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Omega)
$$

where:

- $P \in \mathbb{R}^{k \times n}$: pick matrix (which assets)
- $Q \in \mathbb{R}^k$: expected view returns
- $\Omega \in \mathbb{R}^{k \times k}$: view uncertainty (diagonal)

**Step 3: Bayesian Update**

Prior:

$$
\mu \sim \mathcal{N}(\pi, \tau \Sigma)
$$

Posterior:

$$
\mu \mid Q \sim \mathcal{N}(\mu_{BL}, \Sigma_{BL})
$$

where:

$$
\mu_{BL} = \left[ (\tau \Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \pi + P^\top \Omega^{-1} Q \right]
$$

**Step 4: Optimise**

Apply MVO using $\mu_{BL}$:

$$
w^{BL} = \arg\max_{w \in \mathcal{W}} \; \frac{w^\top \mu_{BL} - r_f}{\sqrt{w^\top \Sigma w}}
$$

---

### 2.5 Risk Parity

**Origins:** Popularised by Bridgewater's "All Weather" fund post-2008. Formalised by Maillard, Roncalli, and Teiletche (2010).

**Key Insight:** Diversify risk, not capital. Each asset should contribute equally to total portfolio risk.

**Risk Decomposition:**

By Euler's theorem for homogeneous functions, portfolio volatility decomposes as:

$$
\sigma_p = \sum_{i=1}^n w_i \frac{\partial \sigma_p}{\partial w_i}
$$

The **marginal risk contribution** is:

$$
\text{MRC}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}
$$

The **risk contribution** of asset $i$ is:

$$
\text{RC}_i = w_i \cdot \text{MRC}_i = \frac{w_i (\Sigma w)_i}{\sigma_p}
$$

Note: $\sum_{i=1}^n \text{RC}_i = \sigma_p$.

**The Risk Parity Condition:**

$$
\text{RC}_i = \frac{\sigma_p}{n} \quad \forall \, i
$$

Equivalently (ignoring the $\sigma_p$ scaling):

$$
w_i (\Sigma w)_i = c \quad \forall \, i
$$

**Solution via Newton-Raphson:**

Define $x = [w^\top, c]^\top$ and:

$$
f(x) = \begin{bmatrix} w_1 (\Sigma w)_1 - c \\ \vdots \\ w_n (\Sigma w)_n - c \\ \mathbf{1}^\top w - 1 \end{bmatrix} = \mathbf{0}
$$

The Jacobian is:

$$
J(x) = \begin{bmatrix} \text{diag}(\Sigma w) + \text{diag}(w) \Sigma & -\mathbf{1} \\ \mathbf{1}^\top & 0 \end{bmatrix}
$$

Newton iteration:

$$
x^{(k+1)} = x^{(k)} - J(x^{(k)})^{-1} f(x^{(k)})
$$

Initialise with inverse-volatility weights. Converges in ~5-10 iterations.

**Special Case:** If assets are uncorrelated ($\Sigma$ diagonal):

$$
w_i^{RP} = \frac{1/\sigma_i}{\sum_{j=1}^n 1/\sigma_j}
$$

---

### 2.6 Model Summary

| Model           | Requires$\mu$? | Requires$\Sigma$? | Key Weakness                 |
| --------------- | ---------------- | ------------------- | ---------------------------- |
| Equal Weight    | No               | No                  | Ignores all information      |
| MVO             | Yes              | Yes                 | Estimation error in $\mu$ |
| Black-Litterman | Partial (views)  | Yes                 | Depends on view quality      |
| Risk Parity     | No               | Yes                 | Overweights low-vol assets   |

---

## 3. Methodology

### 3.1 Data

**Table: Asset Universe**

| Asset Class     | Tickers                               | Count        |
| --------------- | ------------------------------------- | ------------ |
| UK Equities     | HSBC, Lloyds, Barclays, Shell, etc... | 15           |
| UK Govt Bonds   | IGLT.L                                | 1            |
| Precious Metals | SGLD.L                                | 1            |
| Commodities     | WCOG.L                                | 1            |
| **Total** |                                       | **18** |

- Data period: 2015-01-01 to 2025-12-31
- Frequency: Daily adjusted close prices - why did we choose adj close?
- Source: Yahoo Finance

**Figure 1:** Normalised price chart with regime annotations (Brexit, COVID, 2022)

### 3.2 Model Implementation

| Model           | Estimation Window | Constraints   | Solver         |
| --------------- | ----------------- | ------------- | -------------- |
| Equal Weight    | None              | $w_i = 1/n$ | Closed-form    |
| MVO             | 252 days          | Long-only     | SLSQP          |
| Black-Litterman | 252 days          | Long-only     | SLSQP          |
| Risk Parity     | 252 days          | Long-only     | Newton-Raphson |

For Black-Litterman: describe view generation method.

### 3.3 Backtesting Framework

- Monthly rebalancing (month-end)
- Rolling out-of-sample (no lookahead)
- Initial capital: £100,000

**Figure 2:** Backtester flowchart

### 3.4 Performance Metrics

**Sharpe Ratio:**

$$
\text{Sharpe} = \frac{\bar{r}_p - r_f}{\sigma_p}
$$

**Sortino Ratio:**

$$
\text{Sortino} = \frac{\bar{r}_p - r_f}{\sigma_d}, \quad \sigma_d = \sqrt{\frac{1}{T} \sum_{t: r_t < 0} r_t^2}
$$

**Maximum Drawdown:**

$$
\text{MDD} = \max_{t} \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s}
$$

**Turnover:**

$$
\text{Turnover}_t = \sum_{i=1}^n |w_{i,t} - w_{i,t-1}|
$$

Possibly more may be added
--------------------------

## 4. Results (2.5-3 pages)

### 4.1 Main Result: Performance Comparison

**Table 1:** Full-period performance (all 4 strategies, all metrics)

| Metric       | Equal Weight | MVO | Black-Litterman | Risk Parity |
| ------------ | ------------ | --- | --------------- | ----------- |
| Total Return | X%           | X%  | X%              | X%          |
| CAGR         | X%           | X%  | X%              | X%          |
| Volatility   | X%           | X%  | X%              | X%          |
| Sharpe Ratio | X            | X   | X               | X           |
| Max Drawdown | X%           | X%  | X%              | X%          |
| Avg Turnover | X%           | X%  | X%              | X%          |

**Figure 3:** Cumulative wealth curves (all strategies on same axes)

Key observations:

- Which won? By how much?
- What's surprising?

### 4.2 Risk-Return Trade-off

**Figure 4:** Risk-return scatter or Sharpe ratio comparison

Discussion:

- Risk Parity: lowest vol, lowest return
- MVO: [describe behaviour]
- The efficiency frontier in practice

### 4.3 Regime Dependence

**Table 2:** Performance by regime

| Period        | Event      | EW | MVO | BL | RP |
| ------------- | ---------- | -- | --- | -- | -- |
| Jun–Dec 2016 | Brexit     | X% | X%  | X% | X% |
| Feb–Mar 2020 | COVID      | X% | X%  | X% | X% |
| Jan–Dec 2022 | Rate Hikes | X% | X%  | X% | X% |

**Figure 5:** 2022 Crisis Deep Dive

Key insight: Risk Parity's bond overweight was a liability when bonds and equities fell together (correlation breakdown).

### 4.4 Portfolio Characteristics

**Figure 6:** Average allocation by asset class (stacked bar chart)

**Figure 7:** Risk Parity allocation over time (heatmap)

Discussion:

- How stable are allocations?
- Turnover comparison

---

## 5. Discussion

### 5.1 Why Did 1/N Perform Well?

- Estimation error argument (DeMiguel et al.)
- UK market specifics (concentrated sectors)

### 5.2 The Risk Parity Puzzle

- Lower risk, lower Sharpe — why?
- 2022 as a regime break (equity-bond correlation)
- Is Risk Parity "broken" or was this exceptional?

### 5.3 Practical Implications

- When to use each model
- Role of constraints and robustness
- Cost of complexity

### 5.4 Limitations

- UK-only universe (home bias)
- 10-year period (single market cycle)
- No transaction costs (or simplified)
- Parameter choices (estimation window length)

---

## 6. Conclusion

- Restate main finding
- Answer research questions directly
- One sentence on future work

---

## Declarations

- **Who did what?** May be removed, not sure yet lmk what you think

---

## References

Key references to include:

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*
- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio Management*
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*
- Michaud, R. (1989). The Markowitz Optimization Enigma. *Financial Analysts Journal*
- Chaves, D., Hsu, J., Li, F., & Shakernia, O. (2012). Efficient Algorithms for Computing Risk Parity Portfolio Weights. *Journal of Investing*
- Add more if needed

---

## Figures and Tables Checklist

### Figures

| # | Description                             | Section |
| - | --------------------------------------- | ------- |
| 1 | Normalised prices with regime markers   | 3.1     |
| 2 | Backtester flowchart                    | 3.3     |
| 3 | Cumulative wealth curves                | 4.1     |
| 4 | Risk-return scatter / Sharpe comparison | 4.2     |
| 5 | 2022 crisis deep dive                   | 4.3     |
| 6 | Average allocation by asset class       | 4.4     |
| 7 | Allocation heatmap (Risk Parity)        | 4.4     |
| 8 | Add more if needed please               | x       |

### Tables

| # | Description                     | Section |
| - | ------------------------------- | ------- |
| 1 | Full-period performance summary | 4.1     |
| 2 | Performance by regime           | 4.3     |
| 3 | Add more if needed please       | x       |

---

## Equation Checklist

The report should include:

- [ ] Equal weight formula
- [ ] MVO Lagrangian and first-order conditions
- [ ] Two-fund theorem
- [ ] Maximum Sharpe portfolio formula
- [ ] Black-Litterman equilibrium returns
- [ ] Black-Litterman view formulation
- [ ] Black-Litterman posterior formula
- [ ] Risk contribution decomposition (Euler's theorem)
- [ ] Risk Parity condition
- [ ] Newton-Raphson system and Jacobian
- [ ] Inverse-volatility special case
- [ ] Sharpe ratio formula
- [ ] Sortino ratio formula
- [ ] Maximum drawdown formula
- [ ] Maybe others?
