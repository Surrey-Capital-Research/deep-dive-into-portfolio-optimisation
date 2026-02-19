import os
import sys
import numpy as np
import pandas as pd

# Allow imports from src/ when running from the research/ dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pypfopt import EfficientFrontier, objective_functions
from src.backtesting.strategies import BlackLittermanStrategy
from src.models.predictor import ProductionStockRegressor 

# View builder (FFT Signals + Regression)
def fft_view_builder(past_prices, decision_date):
    """
    Generates views (P, Q, Omega) for the Black-Litterman model using FFT cycle analysis.
    """
    tickers = list(past_prices.columns)
    P_list, Q_list, Omega_diag = [], [], []

    print(f"Generating FFT views for {len(tickers)} assets...")

    for i, ticker in enumerate(tickers):
        # init regression model
        reg = ProductionStockRegressor(ticker=ticker, predict_days=30)
        reg.load_data(past_prices[ticker])
        reg.train_hybrid_model()
        
        # extract expected return forecast
        view_return = reg.get_bl_view()
        
        # construct asset picking matrix P
        row = np.zeros(len(tickers))
        row[i] = 1.0
        P_list.append(row)
        
        Q_list.append(view_return)
        
        # set high confidence (low variance) for the signal
        Omega_diag.append(0.0001)

    return np.array(P_list), np.array(Q_list), np.diag(Omega_diag)

# MVO
def mvo_optimizer(mu, cov):
    """
    Translates Black-Litterman posterior returns into optimal target weights.
    Caps positions at 40% to manage concentration risk.
    """
    print("\n[INFO] Top 3 BL Posterior Expected Returns:")
    print(mu.sort_values(ascending=False).head(3))
    
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(0.0, 0.40))
        ef.add_objective(objective_functions.L2_reg, gamma=0.05)
        weights_dict = ef.max_sharpe(risk_free_rate=0.0) 
        return pd.Series(dict(ef.clean_weights()))
    except Exception as e:
        print(f"\n[WARNING] Optimizer collision: {e}. Defaulting to Equal Weight.")
        return pd.Series(1.0 / len(mu), index=mu.index)

# Main loop
def main():
    # Load processed data
    data_path = "data/processed/uk_multi_asset_prices_clean.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")
        
    prices = pd.read_csv(data_path, index_col=0, parse_dates=True)
    market_weights = pd.Series(1.0 / prices.shape[1], index=prices.columns)

    decision_date = prices.index[500]
    past_prices = prices.loc[:decision_date]

    print(f"Calculating FFT signals for {decision_date.date()}...")
    P, Q, Omega = fft_view_builder(past_prices, decision_date)

    def instant_view_builder(prices, date): 
        return P, Q, Omega

    strat = BlackLittermanStrategy(
        market_weights=market_weights,
        risk_aversion=3.0,
        tau=0.05,
        view_builder=instant_view_builder,
        optimizer=mvo_optimizer,
    )

    weights = strat.get_target_weights(
        decision_date=decision_date,
        past_prices=past_prices,
        current_positions=pd.Series(0.0, index=prices.columns),
        cash=100_000.0,
    )

    # Audit Log
    audit_dir = "results/audits"
    os.makedirs(audit_dir, exist_ok=True)

    audit_df = pd.DataFrame({
        'Ticker': prices.columns,
        'Forecasted_Return': Q,
        'Direction': ["BULLISH" if val > 0 else "BEARISH" for val in Q]
    })
    
    file_path = f"{audit_dir}/forecast_audit_{decision_date.date()}.csv"
    audit_df.to_csv(file_path, index=False)
    print(f"\n[INFO] Audit log saved to: {file_path}")

    # Dashboard on terminal
    print(f"\n--- FFT SIGNAL DASHBOARD: {decision_date.date()} ---")
    audit_df = audit_df.sort_values(by='Forecasted_Return', ascending=False)
    
    for _, row in audit_df.iterrows():
        # Clean terminal formatting without emojis
        print(f"{row['Ticker']:<8} | {row['Forecasted_Return']:>8.2%} | {row['Direction']}")

    print(f"\n--- Strategy Report (Dummy Weights) ---")
    print(weights.head())
    
if __name__ == "__main__":
    main()