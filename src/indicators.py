"""
Chỉ báo kỹ thuật: RSI (Wilder), EMA, WMA, ATR.

Tất cả hàm nhận/ trả về pandas Series cùng index, không làm thay đổi input.
RSI dùng Wilder smoothing (giống mặc định TradingView).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI theo Wilder (RMA smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing = EMA với alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Khi avg_loss = 0 (toàn tăng) -> RSI = 100
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    # Khi avg_gain = 0 (toàn giảm) -> RSI = 0
    rsi = rsi.where(avg_gain != 0.0, 0.0)
    return rsi.rename("rsi")


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average tuyến tính (trọng số 1..period)."""
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()

    def _wma(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / wsum)

    return series.rolling(window=period, min_periods=period).apply(_wma, raw=True)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Average True Range theo Wilder."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename("atr")


def compute_rsi_stack(df: pd.DataFrame, rsi_period: int, ema_period: int,
                      wma_period: int, atr_period: int,
                      swing_window: int = 10) -> pd.DataFrame:
    """
    Tính RSI và 2 đường MA của RSI + ATR + swing high/low, gắn vào bản sao của df.

    Trả về df mới với các cột: rsi, rsi_ema, rsi_wma, atr, swing_low, swing_high.
    """
    out = df.copy()
    out["rsi"] = rsi_wilder(out["close"], rsi_period)
    out["rsi_ema"] = ema(out["rsi"], ema_period)
    out["rsi_wma"] = wma(out["rsi"], wma_period)
    out["atr"] = atr_wilder(out["high"], out["low"], out["close"], atr_period)
    out["swing_low"] = out["low"].rolling(swing_window, min_periods=1).min()
    out["swing_high"] = out["high"].rolling(swing_window, min_periods=1).max()
    return out
