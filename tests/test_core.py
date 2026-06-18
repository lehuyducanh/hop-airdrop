"""
Sanity tests: chỉ báo, không look-ahead, và engine chạy được.
Chạy: python -m pytest tests/  (hoặc) python tests/test_core.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators import rsi_wilder, ema, wma, compute_rsi_stack
from src import data as datamod
from src.signals import (build_timeframe_signals, align_higher_tf_states,
                         add_confluence, prepare_signal_frame)
import config as C


def test_rsi_bounds():
    close = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 500)) + 100)
    r = rsi_wilder(close, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all(), "RSI phải nằm trong [0,100]"


def test_rsi_all_up_is_100():
    close = pd.Series(np.arange(1, 100, dtype=float))
    r = rsi_wilder(close, 14).dropna()
    assert r.iloc[-1] > 99.9, "Chuỗi toàn tăng -> RSI ~ 100"


def test_wma_weights_recent_more():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    w = wma(s, 3).iloc[-1]
    # WMA(3) của [3,4,5] = (3*1+4*2+5*3)/6 = 26/6
    assert abs(w - 26 / 6) < 1e-9


def test_no_lookahead_alignment():
    """State khung cao tại nến exec không được dùng thông tin tương lai."""
    base_1h = datamod.synthetic_ohlcv("BTCUSDT", "1h", periods=3000)
    ohlcv = {"4h": datamod.resample_ohlcv(base_1h, "4h"),
             "1d": datamod.resample_ohlcv(base_1h, "1d")}
    ind = C.IndicatorParams()
    sig = build_timeframe_signals(ohlcv, ind)
    aligned = align_higher_tf_states(sig["4h"], {"1d": sig["1d"]})

    # Với mỗi nến 4h, close_time của nến 1d được gán phải <= open_time nến 4h
    daily = sig["1d"].reset_index()[["close_time", "state"]].sort_values("close_time")
    for t in aligned.index[::50]:
        st = aligned.loc[t, "state_1d"]
        if pd.isna(st):
            continue
        valid = daily[daily["close_time"] <= t]
        assert not valid.empty
        assert valid.iloc[-1]["state"] == st, f"Look-ahead tại {t}!"


def test_engine_runs():
    cfg = C.BacktestConfig()
    base_1h = datamod.synthetic_ohlcv("BTCUSDT", "1h", periods=8000)
    ohlcv = {tf: (base_1h if tf == "1h" else datamod.resample_ohlcv(base_1h, tf))
             for tf in [cfg.execution_tf] + cfg.confluence_tfs}
    sig = prepare_signal_frame(ohlcv, cfg).dropna(subset=["atr", "rsi_wma"])
    from src.engine import run_backtest
    res = run_backtest(sig, "BTCUSDT", cfg)
    assert len(res.equity_curve) == len(sig)
    assert res.equity_curve.iloc[-1] > 0, "Equity không được âm tuyệt đối"
    print(f"OK engine: {len(res.trades)} lệnh, equity cuối = {res.equity_curve.iloc[-1]:.2f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nTất cả test PASS.")
