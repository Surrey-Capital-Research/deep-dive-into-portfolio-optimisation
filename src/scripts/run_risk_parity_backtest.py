import pandas as pd                                                                                                              
from src.backtesting.backtester import Backtester                                                                                
from src.backtesting.strategies import RiskParityStrategy                                                                        
                                                                                                                                
# Load price data                                                                                                           
prices = pd.read_csv(                                                                                                            
    "data/uk_multi_asset_prices_clean.csv",                                                                                      
    index_col=0,                                                                                                                 
    parse_dates=True                                                                                                             
)                                                                                                                                
                                                                                                                                
# Create strategy                                                                                                                
strategy = RiskParityStrategy(                                                                                                   
    tickers=prices.columns.tolist(),                                                                                             
    cov_window=252  # 1 year of daily data                                                                                       
)                                                                                                                                
                                                                                                                                
# Run backtest                                                                                                                   
backtester = Backtester(prices=prices, strategy=strategy)                                                                        
result = backtester.run()                                                                                                        
                                                                                                                                
# View results                                                                                                                   
print(result.metrics)
print(result.equity_curve)