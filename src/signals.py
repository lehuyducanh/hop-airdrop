"""
Sinh tín hiệu: trạng thái/trigger từng khung + ghép đa khung KHÔNG look-ahead.

Quy tắc (xem config.py):
- state = +1 nếu RSI > WMA45 (bullish), -1 nếu RSI < WMA45 (bearish), 0 nếu chưa đủ dữ liệu.
- trigger_long  : RSI[t] > EMA9 & > WMA45 và RSI[t-1] <= WMA45[t-1]  (cắt lên từ dưới).
- trigger_short : RSI[t] < EMA9 & < WMA45 và RSI[t-1] >= WMA45[t-1]  (cắt xuống từ trên).

Chống look-ahead: state của khung CAO chỉ được dùng SAU khi nến khung đó ĐÓNG
(close_time). Ghép bằng merge_asof trên close_time -> mỗi nến execution chỉ thấy
thông tin khung cao đã thực sự chốt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import compute_rsi_stack


def add_state_and_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột state, trigger_long, trigger_short. Yêu cầu df đã có rsi/rsi_ema/rsi_wma."""
    out = df.copy()
    rsi, ema_, wma_ = out["rsi"], out["rsi_ema"], out["rsi_wma"]

    state = np.where(rsi > wma_, 1, np.where(rsi < wma_, -1, 0))
    out["state"] = state

    prev_rsi = rsi.shift(1)
    prev_wma = wma_.shift(1)

    out["trigger_long"] = (
        (rsi > ema_) & (rsi > wma_) & (prev_rsi <= prev_wma)
    ).fillna(False)
    out["trigger_short"] = (
        (rsi < ema_) & (rsi < wma_) & (prev_rsi >= prev_wma)
    ).fillna(False)
    return out


def build_timeframe_signals(ohlcv: dict[str, pd.DataFrame], indicators) -> dict[str, pd.DataFrame]:
    """Tính chỉ báo + state/trigger cho từng khung. ohlcv: {tf: DataFrame}."""
    result = {}
    for tf, df in ohlcv.items():
        stacked = compute_rsi_stack(
            df, indicators.rsi_period, indicators.ema_period,
            indicators.wma_period, indicators.atr_period,
        )
        result[tf] = add_state_and_triggers(stacked)
    return result


def align_higher_tf_states(exec_df: pd.DataFrame,
                           higher: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Gán state các khung cao xuống khung thực thi theo close_time (no look-ahead).

    Với mỗi nến execution có open_time t, ta lấy state của khung cao từ nến gần
    nhất ĐÃ ĐÓNG trước hoặc đúng t (merge_asof backward trên close_time).
    """
    base = exec_df.copy()
    # mốc ra quyết định = open của nến exec; chuẩn hóa độ phân giải datetime
    base["_exec_open"] = pd.to_datetime(base.index).as_unit("ns")

    for tf, hdf in higher.items():
        h = hdf.reset_index()[["close_time", "state"]].rename(
            columns={"state": f"state_{tf}"}
        ).sort_values("close_time")
        h["close_time"] = pd.to_datetime(h["close_time"]).dt.as_unit("ns")
        merged = pd.merge_asof(
            base.sort_values("_exec_open"),
            h,
            left_on="_exec_open",
            right_on="close_time",
            direction="backward",
        )
        base[f"state_{tf}"] = merged[f"state_{tf}"].to_numpy()

    base = base.drop(columns=["_exec_open"])
    return base


def add_confluence(exec_df: pd.DataFrame, confluence_tfs: list[str]) -> pd.DataFrame:
    """Đếm số khung cao bullish / bearish tại mỗi nến execution."""
    out = exec_df.copy()
    cols = [f"state_{tf}" for tf in confluence_tfs]
    states = out[cols]
    out["confl_long"] = (states == 1).sum(axis=1)
    out["confl_short"] = (states == -1).sum(axis=1)
    return out


def prepare_signal_frame(symbol_ohlcv: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """
    Pipeline đầy đủ cho 1 symbol:
      OHLCV mọi khung -> chỉ báo+state/trigger -> ghép khung cao -> đếm confluence.
    Trả về DataFrame khung thực thi đã đủ cột để chạy engine.
    """
    tf_sig = build_timeframe_signals(symbol_ohlcv, cfg.indicators)
    exec_df = tf_sig[cfg.execution_tf]
    higher = {tf: tf_sig[tf] for tf in cfg.confluence_tfs}
    aligned = align_higher_tf_states(exec_df, higher)
    final = add_confluence(aligned, cfg.confluence_tfs)
    return final
