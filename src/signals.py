"""
Sinh tín hiệu đa khung, ghép về KHUNG NỀN (manage_tf) KHÔNG look-ahead.

Phân tách trách nhiệm:
- execution_tf : sinh trigger vào lệnh + state để thoát hẳn (mất phe).
- confluence_tfs: state khung cao để đếm đồng thuận.
- manage_tf (nền): khung mịn hơn execution; engine chạy theo khung này và dùng
  RSI/EMA9 của chính nó để tăng/giảm volume (scale-out/in).

Chống look-ahead: với mỗi nến nền đóng tại close_time t, chỉ dùng dữ liệu của
khung cao/execution đã ĐÓNG trước hoặc đúng t (merge_asof backward theo close_time).
Quyết định ở close nến nền -> khớp ở open nến nền kế tiếp (xử lý trong engine).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import compute_rsi_stack


def add_state_and_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm state, trigger_long, trigger_short. Yêu cầu đã có rsi/rsi_ema/rsi_wma."""
    out = df.copy()
    rsi, ema_, wma_ = out["rsi"], out["rsi_ema"], out["rsi_wma"]

    out["state"] = np.where(rsi > wma_, 1, np.where(rsi < wma_, -1, 0))

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
    """Tính chỉ báo + state/trigger cho từng khung."""
    result = {}
    for tf, df in ohlcv.items():
        stacked = compute_rsi_stack(
            df, indicators.rsi_period, indicators.ema_period,
            indicators.wma_period, indicators.atr_period,
        )
        result[tf] = add_state_and_triggers(stacked)
    return result


def align_to_base(base_df: pd.DataFrame, source_df: pd.DataFrame,
                  value_map: dict[str, str],
                  also_close_time_as: str | None = None) -> pd.DataFrame:
    """
    Gán các cột của source_df xuống base_df theo close_time (backward, no look-ahead).

    value_map: {tên_cột_nguồn: tên_cột_đích}. Với mỗi nến nền, lấy giá trị của
    nến nguồn gần nhất đã ĐÓNG (close_time <= close_time nền).
    """
    base = base_df.sort_index().copy()
    bkey = pd.to_datetime(base["close_time"]).dt.as_unit("ns").to_numpy()

    src_cols = list(value_map.keys())
    src = source_df.sort_index().reset_index()
    src_ct = pd.to_datetime(src["close_time"]).dt.as_unit("ns").to_numpy()

    left = pd.DataFrame({"_k": bkey})
    right = pd.DataFrame({"_k": src_ct})
    for c in src_cols:
        right[value_map[c]] = src[c].to_numpy()
    if also_close_time_as:
        right[also_close_time_as] = src_ct
    right = right.sort_values("_k")

    merged = pd.merge_asof(left, right, on="_k", direction="backward")

    out_cols = [value_map[c] for c in src_cols]
    if also_close_time_as:
        out_cols.append(also_close_time_as)
    for col in out_cols:
        base[col] = merged[col].to_numpy()
    return base


def prepare_signal_frame(symbol_ohlcv: dict[str, pd.DataFrame], cfg) -> pd.DataFrame:
    """
    Pipeline đầy đủ cho 1 symbol:
      OHLCV mọi khung -> chỉ báo+state/trigger -> ghép execution+confluence xuống
      khung nền (manage_tf) -> đếm confluence. Trả về DataFrame khung nền sẵn sàng
      cho engine.

    Cột khung nền giữ nguyên rsi/rsi_ema/rsi_wma/atr/ohlc của chính nó (dùng để
    quản lý volume + stop). Thêm:
      exec_trigger_long/short, exec_state, exec_ct  (từ execution_tf)
      state_<tf> + confl_long/short                (từ confluence_tfs)
    """
    tf_sig = build_timeframe_signals(symbol_ohlcv, cfg.indicators)
    base = tf_sig[cfg.base_tf]
    exec_sig = tf_sig[cfg.execution_tf]

    # Tín hiệu execution -> nền (kèm exec_ct để phát hiện nến execution mới)
    base = align_to_base(
        base, exec_sig,
        {"trigger_long": "exec_trigger_long",
         "trigger_short": "exec_trigger_short",
         "state": "exec_state"},
        also_close_time_as="exec_ct",
    )

    # Đồng thuận khung cao -> nền
    for tf in cfg.confluence_tfs:
        base = align_to_base(base, tf_sig[tf], {"state": f"state_{tf}"})

    # Chuẩn hóa kiểu dữ liệu
    base["exec_trigger_long"] = base["exec_trigger_long"].fillna(False).astype(bool)
    base["exec_trigger_short"] = base["exec_trigger_short"].fillna(False).astype(bool)
    base["exec_state"] = base["exec_state"].fillna(0)

    cols = [f"state_{tf}" for tf in cfg.confluence_tfs]
    base["confl_long"] = (base[cols] == 1).sum(axis=1)
    base["confl_short"] = (base[cols] == -1).sum(axis=1)
    return base
