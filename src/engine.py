"""
Event-driven backtest engine cho Binance USDT-M Futures (long + short)
với QUẢN LÝ KHỐI LƯỢNG ĐỘNG (scale-out / scale-in).

Mô hình "target-weight reconciliation":
- Mỗi khi nến khung thực thi đóng, hệ thống xác định TRỌNG SỐ MỤC TIÊU của vị thế
  dựa trên rủi ro đảo chiều khung nhỏ:
    STRONG  (RSI > EMA9)              -> weight 1.0   (full)
    CAUTION (RSI <= EMA9, > WMA45)    -> weight reduced_weight  (giảm volume)
    BROKEN  (RSI < WMA45 / mất phe)   -> weight 0     (thoát hẳn)
  Khi khung nhỏ vào lại chu kỳ (RSI lấy lại EMA9) -> tăng volume trở lại full.
- Lệnh (mở/đóng/scale) khớp ở OPEN nến kế tiếp (chống look-ahead).
- Trong nến: ưu tiên kiểm tra STOP trước.

Hạch toán: dùng giá vốn bình quân gia quyền cho phần scale-in; PnL của phần
scale-out được hiện thực hóa và cộng dồn; mỗi vị thế (round-trip) phát sinh đúng
1 bản ghi Trade khi đóng hoàn toàn -> n_trades = số vòng lệnh, dễ đọc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Trade:
    symbol: str
    side: str            # 'long' | 'short'
    entry_time: pd.Timestamp
    entry_price: float   # giá vốn bình quân ban đầu
    qty: float           # khối lượng đỉnh (base_qty)
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0          # lãi/lỗ ròng cả vòng (đã trừ phí + funding)
    fees: float = 0.0
    funding: float = 0.0
    r_multiple: float = 0.0
    reason: str = ""
    scale_outs: int = 0
    scale_ins: int = 0


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: pd.Series
    trades: list = field(default_factory=list)
    config_label: str = ""
    n_scale_outs: int = 0
    n_scale_ins: int = 0


def _funding_due(open_time: pd.Timestamp) -> bool:
    """Funding Binance ~ 00:00 / 08:00 / 16:00 UTC."""
    return open_time.hour in (0, 8, 16)


def _manage_weight(side: str, rsi, ema, strat) -> float:
    """
    Trọng số volume theo rủi ro đảo chiều KHUNG NỀN (manage_tf).
    Việc thoát hẳn do execution_tf/stop quyết định, không nằm ở đây.
    - long : RSI còn trên EMA9 -> full (1.0); mất EMA9 -> reduced_weight.
    - short: RSI còn dưới EMA9 -> full (1.0); mất EMA9 -> reduced_weight.
    """
    if not strat.tiered_management:
        return 1.0
    if np.isnan(rsi) or np.isnan(ema):
        return 1.0
    if side == "long":
        return 1.0 if rsi > ema else strat.reduced_weight
    return 1.0 if rsi < ema else strat.reduced_weight


def run_backtest(df: pd.DataFrame, symbol: str, cfg, label: str = "") -> BacktestResult:
    """
    df: khung thực thi đã có open/high/low/close, atr, rsi, rsi_ema, rsi_wma,
        state, trigger_long/short, confl_long/short.
    """
    risk = cfg.risk
    strat = cfg.strategy

    equity = risk.initial_equity      # số dư đã hiện thực hóa (cash, futures PnL-based)
    equity_curve = []
    trades: list[Trade] = []
    n_scale_outs = n_scale_ins = 0

    pos = None
    n = len(df)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    rsi = df["rsi"].to_numpy()              # RSI khung nền (manage) -> quản lý volume
    rsi_ema = df["rsi_ema"].to_numpy()
    # Tín hiệu execution (đã align xuống khung nền)
    tlong = df["exec_trigger_long"].to_numpy()
    tshort = df["exec_trigger_short"].to_numpy()
    exec_state = df["exec_state"].to_numpy()
    exec_ct = df["exec_ct"].to_numpy()
    clong = df["confl_long"].to_numpy()
    cshort = df["confl_short"].to_numpy()
    exec_prev_squeeze = df["exec_prev_squeeze"].to_numpy()
    exec_dbl_bottom = df["exec_dbl_bottom"].to_numpy()
    exec_dbl_top = df["exec_dbl_top"].to_numpy()
    exec_swing_low = df["exec_swing_low"].to_numpy()
    exec_swing_high = df["exec_swing_high"].to_numpy()
    times = df.index

    pending = None       # (side, weight, reason, sw_low, sw_high): trạng thái mục tiêu
    last_exec_ct = None  # close_time của nến execution gần nhất đã xử lý (edge detect)

    # ---- thao tác vị thế ----
    def open_position(side, price, time_i, atr_val, weight, sw_low=np.nan, sw_high=np.nan):
        nonlocal equity
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            return None
        fill = price * (1 + risk.slippage) if side == "long" else price * (1 - risk.slippage)

        # Stop distance: swing high/low hoặc ATR fallback
        if risk.use_swing_stop and not np.isnan(sw_low) and not np.isnan(sw_high):
            if side == "long":
                sw_dist = max(fill - sw_low, 0.0)
            else:
                sw_dist = max(sw_high - fill, 0.0)
            floor_d = risk.min_swing_atr_mult * atr_val
            cap_d = risk.max_swing_atr_mult * atr_val
            stop_dist = float(np.clip(sw_dist, floor_d, cap_d))
        else:
            stop_dist = risk.atr_stop_mult * atr_val

        if stop_dist <= 0:
            return None
        risk_cash = equity * risk.risk_per_trade
        base_qty = risk_cash / stop_dist
        max_notional = risk.max_leverage * equity
        if base_qty * price > max_notional:
            base_qty = max_notional / price
        qty = base_qty * weight
        if qty <= 0:
            return None
        fee = fill * qty * risk.taker_fee
        equity -= fee
        stop = fill - stop_dist if side == "long" else fill + stop_dist
        return {
            "side": side, "entry_price": fill, "qty": qty, "base_qty": base_qty,
            "stop": stop, "entry_time": time_i, "atr": atr_val, "fees": fee,
            "funding": 0.0, "realized": 0.0, "extreme": fill,
            "init_stop_dist": stop_dist, "scale_outs": 0, "scale_ins": 0,
        }

    def scale_out(dq, price, time_i):
        """Giảm khối lượng dq tại price, hiện thực hóa PnL phần giảm."""
        nonlocal equity, n_scale_outs
        side = pos["side"]
        fill = price * (1 - risk.slippage) if side == "long" else price * (1 + risk.slippage)
        fee = fill * dq * risk.taker_fee
        gross = (fill - pos["entry_price"]) * dq if side == "long" else (pos["entry_price"] - fill) * dq
        equity += gross - fee
        pos["realized"] += gross
        pos["fees"] += fee
        pos["qty"] -= dq
        pos["scale_outs"] += 1
        n_scale_outs += 1

    def scale_in(dq, price, time_i):
        """Tăng khối lượng dq tại price, cập nhật giá vốn bình quân."""
        nonlocal equity, n_scale_ins
        side = pos["side"]
        fill = price * (1 + risk.slippage) if side == "long" else price * (1 - risk.slippage)
        fee = fill * dq * risk.taker_fee
        equity -= fee
        pos["entry_price"] = (pos["entry_price"] * pos["qty"] + fill * dq) / (pos["qty"] + dq)
        pos["qty"] += dq
        pos["fees"] += fee
        pos["scale_ins"] += 1
        n_scale_ins += 1

    def close_position(price, time_i, reason):
        nonlocal pos
        scale_out(pos["qty"], price, time_i)  # đóng nốt phần còn lại
        pnl = pos["realized"] - pos["fees"] - pos["funding"]
        denom = pos["init_stop_dist"] * pos["base_qty"]
        r = pnl / denom if denom > 0 else 0.0
        fill = price * (1 - risk.slippage) if pos["side"] == "long" else price * (1 + risk.slippage)
        trades.append(Trade(
            symbol=symbol, side=pos["side"], entry_time=pos["entry_time"],
            entry_price=pos["entry_price"], qty=pos["base_qty"], exit_time=time_i,
            exit_price=fill, pnl=pnl, fees=pos["fees"], funding=pos["funding"],
            r_multiple=r, reason=reason, scale_outs=pos["scale_outs"],
            scale_ins=pos["scale_ins"],
        ))
        pos = None

    def reconcile(side, weight, price, time_i, atr_for_new, reason,
                  sw_low=np.nan, sw_high=np.nan):
        """Đưa vị thế hiện tại về (side, weight) mục tiêu."""
        nonlocal pos
        # đổi chiều hoặc về flat -> đóng trước
        if pos is not None and (side == "flat" or pos["side"] != side):
            close_position(price, time_i, reason)
        if side in ("long", "short"):
            if pos is None:
                newpos = open_position(side, price, time_i, atr_for_new, weight, sw_low, sw_high)
                if newpos:
                    pos = newpos
            elif pos["side"] == side:
                target = pos["base_qty"] * weight
                eps = strat.min_rebalance_frac * pos["base_qty"]
                if target < pos["qty"] - eps:
                    scale_out(pos["qty"] - target, price, time_i)
                elif target > pos["qty"] + eps:
                    scale_in(target - pos["qty"], price, time_i)

    # ---- vòng lặp chính ----
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # 1) Khớp trạng thái mục tiêu quyết định ở nến trước, tại OPEN nến này
        if pending is not None:
            side, weight, reason, sw_low, sw_high = pending
            pending = None
            reconcile(side, weight, o, times[i], atr[i - 1] if i > 0 else atr[i],
                      reason, sw_low, sw_high)

        # 2) Quản trị vị thế đang mở: funding, trailing, stop (trong nến)
        if pos is not None:
            if risk.apply_funding and _funding_due(times[i]):
                f = pos["qty"] * c * risk.funding_rate
                f = f if pos["side"] == "long" else -f
                pos["funding"] += f
                equity -= f

            if risk.use_trailing:
                if pos["side"] == "long":
                    pos["extreme"] = max(pos["extreme"], h)
                    pos["stop"] = max(pos["stop"], pos["extreme"] - risk.atr_trail_mult * pos["atr"])
                else:
                    pos["extreme"] = min(pos["extreme"], l)
                    pos["stop"] = min(pos["stop"], pos["extreme"] + risk.atr_trail_mult * pos["atr"])

            if pos["side"] == "long" and l <= pos["stop"]:
                close_position(pos["stop"], times[i], "stop")
            elif pos["side"] == "short" and h >= pos["stop"]:
                close_position(pos["stop"], times[i], "stop")

        # 3) Quyết định trạng thái mục tiêu cho nến kế tiếp (dựa trên CLOSE nến i)
        #    Trigger execution chỉ được "ăn" 1 lần / nến execution mới (edge detect).
        is_new_exec = exec_ct[i] != last_exec_ct
        if is_new_exec:
            last_exec_ct = exec_ct[i]
        n_req = strat.confluence_n

        if pos is None:
            squeeze_ok = not strat.filter_squeeze or not exec_prev_squeeze[i]
            dbl_ok_long = not strat.require_double_pattern or exec_dbl_bottom[i]
            dbl_ok_short = not strat.require_double_pattern or exec_dbl_top[i]
            sw_l, sw_h = exec_swing_low[i], exec_swing_high[i]
            if (is_new_exec and strat.allow_long and tlong[i]
                    and clong[i] >= n_req and squeeze_ok and dbl_ok_long):
                pending = ("long", 1.0, "entry", sw_l, sw_h)
            elif (is_new_exec and strat.allow_short and tshort[i]
                    and cshort[i] >= n_req and squeeze_ok and dbl_ok_short):
                pending = ("short", 1.0, "entry", sw_l, sw_h)
        else:
            side = pos["side"]
            # tín hiệu execution ngược chiều (chỉ tính trên nến execution mới)
            if side == "long":
                opp = is_new_exec and strat.allow_short and tshort[i] and cshort[i] >= n_req
                exec_invalid = exec_state[i] == -1   # execution mất phe long
            else:
                opp = is_new_exec and strat.allow_long and tlong[i] and clong[i] >= n_req
                exec_invalid = exec_state[i] == 1

            sw_l, sw_h = exec_swing_low[i], exec_swing_high[i]
            if opp and strat.allow_flip:
                new_side = "short" if side == "long" else "long"
                pending = (new_side, 1.0, "flip", sw_l, sw_h)
            elif strat.exit_on_state_flip and exec_invalid:
                pending = ("flat", 0.0, "exec_exit", np.nan, np.nan)
            elif opp:
                pending = ("flat", 0.0, "opp_exit", np.nan, np.nan)
            else:
                # còn trong xu hướng execution -> chỉ tăng/giảm volume theo manage_tf
                w = _manage_weight(side, rsi[i], rsi_ema[i], strat)
                pending = (side, w, "rebalance", np.nan, np.nan)

        # 4) Mark-to-market equity tại close
        if pos is not None:
            if pos["side"] == "long":
                unreal = (c - pos["entry_price"]) * pos["qty"]
            else:
                unreal = (pos["entry_price"] - c) * pos["qty"]
            equity_curve.append(equity + unreal)
        else:
            equity_curve.append(equity)

    if pos is not None:
        close_position(closes[-1], times[-1], "eod")
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=df.index, name="equity")
    return BacktestResult(symbol=symbol, equity_curve=eq, trades=trades,
                          config_label=label, n_scale_outs=n_scale_outs,
                          n_scale_ins=n_scale_ins)
