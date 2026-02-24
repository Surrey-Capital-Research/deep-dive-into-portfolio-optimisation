import numpy as np
import pandas as pd
import os
from src.backtesting.strategies import BlackLittermanStrategy
from src.models.predictor import ProductionStockRegressor 

# --- 1. THE VIEW BUILDER (FFT OPINIONS) ---
def fft_view_builder(past_prices, decision_date):
    """
    This is our 'Opinion Generator'. It scans every stock in the list,
    runs a cycle analysis (FFT), and tells the model what it thinks 
    is going to happen over the next 30 days.
    """
    tickers = list(past_prices.columns)
    P_list = []
    Q_list = []
    Omega_diag = []

    # Let's let the user know we're actually doing work here
    print(f"Running FFT cycles for {len(tickers)} assets... hang tight.")

    for i, ticker in enumerate(tickers):
        # We pull up the 'Best Version Ever' of our brain for this specific ticker
        brain = ProductionStockRegressor(ticker=ticker, predict_days=30)
        
        # Shoving the raw price data into the brain
        brain.load_data(past_prices[ticker])
        
        # Run the math to find the trend and the cycles
        brain.train_hybrid_model()
        
        # This is the 'Aha!' moment—the expected % return according to the math
        view_return = brain.get_bl_view()
        
        # We build a 'Picker' matrix (P) so the model knows which stock we're talking about
        row = np.zeros(len(tickers))
        row[i] = 1
        P_list.append(row)
        
        # Add the forecast (Q) to our list
        Q_list.append(view_return)
        
        # We set a tiny 'uncertainty' value. 0.0001 means we trust this signal quite a bit.
        Omega_diag.append(0.0001)

    return np.array(P_list), np.array(Q_list), np.diag(Omega_diag)


# --- 2. THE PLACEHOLDER OPTIMISER ---
def dummy_optimiser(mu, cov):
    """
    TECK DEBT NOTE: My teammate is still finishing the MVO 'Chef'.
    For now, I'm just outputting equal weights so we can verify the 
    Black-Litterman returns (mu) are actually being updated by the FFT.
    """
    # Quick sanity check: show us the top 3 stocks the FFT is bullish on
    print("\nFFT is most Bullish on:")
    print(mu.sort_values(ascending=False).head(3))
    
    # Just split the money evenly for now to keep the code running
    n = len(mu)
    return pd.Series(1.0 / n, index=mu.index)


# --- 3. THE MAIN LOOP ---
def main():
    # 1. Load Data
    prices = pd.read_csv("data/uk_multi_asset_prices_clean.csv", index_col=0, parse_dates=True)
    market_weights = pd.Series(1.0 / prices.shape[1], index=prices.columns)

    # 2. Setup the "Heavy Lifting"
    decision_date = prices.index[500]
    past_prices = prices.loc[:decision_date]

    print(f"Pre-calculating FFT signals for {decision_date.date()}...")
    P, Q, Omega = fft_view_builder(past_prices, decision_date)

    # 3. Setup Strategy with Dummy Optimizer
    def instant_view_builder(prices, date): return P, Q, Omega

    strat = BlackLittermanStrategy(
        market_weights=market_weights,
        risk_aversion=3.0,
        tau=0.05,
        view_builder=instant_view_builder,
        optimiser=dummy_optimiser,
    )

    # 4. Get the (Dummy) Weights
    weights = strat.get_target_weights(
        decision_date=decision_date,
        past_prices=past_prices,
        current_positions=pd.Series(0.0, index=prices.columns),
        cash=100_000.0,
    )

    # ... (Keep everything above step 5 the same) ...

    # --- 5. THE AUDIT LOG (TRACK RECORD) ---
    if not os.path.exists("results"):
        os.makedirs("results")

    audit_df = pd.DataFrame({
        'Ticker': prices.columns,
        'Forecasted_Return': Q,
        'Direction': ["BULLISH" if val > 0 else "BEARISH" for val in Q]
    })
    
    file_path = f"results/forecast_audit_{decision_date.date()}.csv"
    audit_df.to_csv(file_path, index=False)
    print(f"\n✅ Full Audit Log saved to: {file_path}")

    # --- 6. THE TERMINAL DASHBOARD ---
    print(f"\n--- FFT CONVICTION DASHBOARD: {decision_date.date()} ---")
    
    # Sort them so you see the strongest signals first
    audit_df = audit_df.sort_values(by='Forecasted_Return', ascending=False)
    
    for _, row in audit_df.iterrows():
        emoji = "🚀 BULLISH" if row['Forecasted_Return'] > 0 else "📉 BEARISH"
        # Formatting the print so it's easy to read in the terminal
        print(f"{row['Ticker']:<8} | {row['Forecasted_Return']:>8.2%} | {emoji}")

    print(f"\n--- Strategy Report (Dummy Weights) ---")
    print(weights.head())
    
if __name__ == "__main__":
    main()