"""
GoldBot Indicators
- Range breakout detection
- EMA trend filter
- RSI momentum
- ATR volatility
"""

import pandas as pd
import numpy as np


class Indicators:

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta    = prices.diff()
        gain     = delta.where(delta > 0, 0.0)
        loss     = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def atr(high: pd.Series, low: pd.Series,
            close: pd.Series, period: int = 14) -> pd.Series:
        prev  = close.shift(1)
        tr    = pd.concat([
            high - low,
            (high - prev).abs(),
            (low  - prev).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def range_breakout(high: pd.Series, low: pd.Series,
                       period: int, buffer: float):
        """
        Hitung range high/low dari N candle sebelumnya.
        Return upper band dan lower band (sudah include buffer).

        BUY signal  : close > upper_band
        SELL signal : close < lower_band
        """
        # Shift 1 supaya tidak include candle saat ini
        range_high = high.shift(1).rolling(period).max()
        range_low  = low.shift(1).rolling(period).min()
        upper_band = range_high + buffer * 0.01  # buffer dalam points (digits=2)
        lower_band = range_low  - buffer * 0.01
        return upper_band, range_high, range_low, lower_band
