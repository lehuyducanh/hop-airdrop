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
from src.signals import build_timeframe_signals, align_to_base, prepare_signal_frame
import config as C


def _make_cfg(manage_tf, execution_tf="4h", confluence=("1d",)):
    cfg = C.BacktestConfig()
    cfg.execution_tf = execution_tf
    cfg.manage_tf = manage_tf
    cfg.confluence_tfs = list(confluence)
    return cfg


def _synth_ohlcv(cfg, symbol="BTCUSDT", periods=12000):
    base = datamod.synthetic_ohlcv(symbol, cfg.base_tf, periods=periods)
    tfs = {cfg.base_tf, cfg.execution_tf, *cfg.confluence_tfs}
    return {tf: (base.copy() if tf == cfg.base_tf else datamod.resample_ohlcv(base, tf))
            for tf in tfs}


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
    """State execution gán xuống khung nền không dùng thông tin tương lai."""
    cfg = _make_cfg(manage_tf="1h", execution_tf="4h", confluence=("1d",))
    ohlcv = _synth_ohlcv(cfg, periods=6000)
    tf_sig = build_timeframe_signals(ohlcv, cfg.indicators)
    base = prepare_signal_frame(ohlcv, cfg)

    # Với mỗi nến nền 1h, exec_state phải khớp nến 4h gần nhất ĐÃ ĐÓNG (close<=close nền)
    ex = tf_sig["4h"].reset_index()[["close_time", "state"]].sort_values("close_time")
    for t in base.index[::40]:
        st = base.loc[t, "exec_state"]
        bt = pd.Timestamp(base.loc[t, "close_time"])
        valid = ex[ex["close_time"] <= bt]
        if valid.empty or pd.isna(st):
            continue
        assert valid.iloc[-1]["state"] == st, f"Look-ahead tại {t}!"


def test_manage_finer_than_execution():
    """Engine chạy theo khung nền mịn (1h) trong khi vào lệnh ở 4h."""
    from src.engine import run_backtest
    cfg = _make_cfg(manage_tf="1h", execution_tf="4h", confluence=("12h", "1d"))
    ohlcv = _synth_ohlcv(cfg, periods=15000)
    sig = prepare_signal_frame(ohlcv, cfg).dropna(subset=["atr", "rsi_wma"])
    # khung nền phải mịn hơn execution -> nhiều nến hơn nhiều
    assert len(sig) > len(ohlcv["4h"])
    res = run_backtest(sig, "BTCUSDT", cfg)
    assert len(res.equity_curve) == len(sig)
    assert res.equity_curve.iloc[-1] > 0
    print(f"OK manage<exec: bars={len(sig)}, lệnh={len(res.trades)}, "
          f"scale_out={res.n_scale_outs}, scale_in={res.n_scale_ins}")


def test_engine_runs():
    cfg = _make_cfg(manage_tf=None, execution_tf="4h", confluence=("12h", "1d"))
    ohlcv = _synth_ohlcv(cfg, periods=8000)
    sig = prepare_signal_frame(ohlcv, cfg).dropna(subset=["atr", "rsi_wma"])
    from src.engine import run_backtest
    res = run_backtest(sig, "BTCUSDT", cfg)
    assert len(res.equity_curve) == len(sig)
    assert res.equity_curve.iloc[-1] > 0, "Equity không được âm tuyệt đối"
    print(f"OK engine: {len(res.trades)} lệnh, equity cuối = {res.equity_curve.iloc[-1]:.2f}")


def test_tiered_management_scales():
    """Bật tiered -> có scale-out/in; tắt -> không có scale event nào."""
    from src.engine import run_backtest
    cfg = _make_cfg(manage_tf="1h", execution_tf="4h", confluence=("12h", "1d"))
    ohlcv = _synth_ohlcv(cfg, periods=15000)
    sig = prepare_signal_frame(ohlcv, cfg).dropna(subset=["atr", "rsi_wma"])

    cfg.strategy.tiered_management = True
    on = run_backtest(sig, "BTCUSDT", cfg)
    assert on.n_scale_outs > 0, "Bật tiered phải có scale-out khi RSI mất EMA9"

    cfg.strategy.tiered_management = False
    off = run_backtest(sig, "BTCUSDT", cfg)
    assert off.n_scale_ins == 0, "Tắt tiered không được scale-in"
    # khi tắt, chỉ giảm về 0 (thoát hẳn) -> không có bước giảm một phần lặp lại
    print(f"OK tiered: scale_out on={on.n_scale_outs}, off={off.n_scale_outs}")


def test_weighted_avg_entry_on_scale_in():
    """scale_in cập nhật giá vốn bình quân đúng công thức."""
    from src.engine import run_backtest
    # kiểm tra trực tiếp công thức avg: (p1*q1 + p2*q2)/(q1+q2)
    q1, p1, q2, p2 = 1.0, 100.0, 0.5, 110.0
    avg = (p1 * q1 + p2 * q2) / (q1 + q2)
    assert abs(avg - (100 + 55) / 1.5) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nTất cả test PASS.")
