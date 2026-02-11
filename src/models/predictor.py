import numpy as np
import pandas as pd
import datetime as dt
from typing import List, Dict, Optional
import os
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Need to check first if we have GPU power for the deep learning stuff
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

def _calculate_fft_waveform(residuals: np.ndarray, num_harmonics: int, total_samples: int) -> np.ndarray:
    """
    DEEP DIVE: We use FFT to find 'Market Cycles'.
    Think of it like tuning a radio to find the clearest signal amidst the static.
    """
    n = len(residuals)
    if n == 0: return np.zeros(total_samples)
    
    # Transform to freq domain
    fft_coeffs = np.fft.fft(residuals)
    frequencies = np.fft.fftfreq(n)
    
    # Sort since we only care for the dominant harmonics
    sorted_indices = np.argsort(np.abs(fft_coeffs[1:n//2]))[::-1] + 1
    top_indices = [0] + list(sorted_indices[:num_harmonics])
    
    t = np.arange(total_samples)
    fft_reconstructed = np.zeros(total_samples)
    
    for i in top_indices:
        # Reconstruct the sine wave for this freq
        amplitude = np.abs(fft_coeffs[i]) / n
        phase = np.angle(fft_coeffs[i])
        fft_reconstructed += amplitude * np.cos(2 * np.pi * frequencies[i] * t + phase)
        
        # Need to mirror for the neg freq
        if i != 0 and i != n//2:
            neg_i = n - i
            fft_reconstructed += (np.abs(fft_coeffs[neg_i]) / n) * \
                                np.cos(2 * np.pi * frequencies[neg_i] * t + np.angle(fft_coeffs[neg_i]))
    return fft_reconstructed

class ProductionStockRegressor:
    """
    THE ALL-IN-ONE BRAIN.
    Combines Ridge, FFT, and LSTM.
    """
    def __init__(self, ticker, predict_days=30):
        self.ticker = ticker
        self.predict_days = predict_days
        self.predictions: Dict[str, pd.Series] = {}
        self.data: Optional[pd.DataFrame] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, price_series: pd.Series):
        """
        HUMAN NOTE: We use 'Adj Close' because it accounts for dividends. 
        If we used raw 'Close', our math would break every time a stock splits.
        """
        df = price_series.to_frame(name='Adj Close')
        self.data = df.ffill().bfill() # Gap filling so doesn't explode
        
        # Timeline setup
        self.train_size = len(self.data)
        self.total_size = self.train_size + self.predict_days
        self.X_train = np.arange(self.train_size).reshape(-1, 1)
        self.y_train = self.data['Adj Close'].values
        
        # Future timeline setup for forecast
        self.full_timeline = np.arange(self.total_size).reshape(-1, 1)
        self.future_index = pd.date_range(
            start=self.data.index[0], 
            periods=self.total_size, 
            freq='B' # Skips weekends so 'B' for business days
        )

    def train_hybrid_model(self):
        """
        THE GOLD STANDARD:
        1. Ridge handles the drift.
        2. FFT handles the seasonality.
        """
        # --- 1. RIDGE REGRESSION ---
        # Use alpha=50 to penalise model if it gets too twitchy. 
        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(self.X_train)
        trend_model = Ridge(alpha=100.0).fit(X_poly, self.y_train)
        
        full_poly = poly.transform(self.full_timeline)
        trend_forecast = trend_model.predict(full_poly)
        
        # --- 2. FFT WAVEFORM ---
        # Let's look for the 6 strongest cycles in the data.
        residuals = self.y_train - trend_forecast[:self.train_size]
        cycle_forecast = _calculate_fft_waveform(residuals, num_harmonics=6, total_samples=self.total_size)
        
        # --- 3. COMBINE ---
        self.predictions['Hybrid_FFT'] = pd.Series(
            trend_forecast + cycle_forecast, 
            index=self.future_index
        )

    def train_lstm_logic(self):
        """
        THE DEEP LEARNING BIT: 
        LSTMs have 'memory'. They remember if a stock usually bounces 
        after a 3-day drop. Great for spotting patterns humans miss.
        """
        if not KERAS_AVAILABLE:
            print("Keras missing. Skipping LSTM...")
            return

        # Prepare Data
        scaled_data = self.scaler.fit_transform(self.y_train.reshape(-1, 1))
        
        # Look at prev 60 days to predict the next 1
        X_lstm, y_lstm = [], []
        win = 60
        for i in range(win, len(scaled_data)):
            X_lstm.append(scaled_data[i-win:i, 0])
            y_lstm.append(scaled_data[i, 0])
        
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

        # Build the Stack
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(win, 1)),
            Dropout(0.2), # Shut off 20% to prevent memorisation
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        # EarlyStopping: Stop training once the model stops improving. 
        # Saves time
        callback = EarlyStopping(monitor='loss', patience=3)
        model.fit(X_lstm, y_lstm, batch_size=32, epochs=10, callbacks=[callback], verbose=0)

        # Forward forecast
        last_60_days = scaled_data[-win:]
        current_batch = last_60_days.reshape((1, win, 1))
        lstm_preds = []

        for _ in range(self.predict_days):
            pred = model.predict(current_batch, verbose=0)[0]
            lstm_preds.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

        # Inverse scaling to get back to dollar/pound amounts
        final_preds = self.scaler.inverse_transform(np.array(lstm_preds).reshape(-1, 1))
        
        # Stick it onto the end of our actual prices
        full_series = np.append(self.y_train, final_preds)
        self.predictions['LSTM_Model'] = pd.Series(full_series, index=self.future_index)

    def get_bl_view(self) -> float:
        """
        THE INTERFACE:
        This is what Black-Litterman calls. It asks: 'Give me your best guess'.
        We average the FFT and LSTM to be safe (Ensemble approach).
        """
        preds = []
        if 'Hybrid_FFT' in self.predictions: preds.append(self.predictions['Hybrid_FFT'].iloc[-1])
        if 'LSTM_Model' in self.predictions: preds.append(self.predictions['LSTM_Model'].iloc[-1])
        
        if not preds: return 0.0
        
        target_price = np.mean(preds)
        current_price = self.y_train[-1]
        
        # Return expected % return
        return (target_price - current_price) / current_price