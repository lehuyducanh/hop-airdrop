"""
Event-driven backtest engine cho Binance USDT-M Futures (long + short).

Nguyên tắc thực thi (chống look-ahead):
- Tín hiệu quyết định trên CLOSE của nến t.
- Lệnh khớp ở OPEN của nến t+1 (cộng slippage).
- Trong một nến, ưu tiên kiểm tra STOP trước khi xét tín hiệu thoát/đảo.

Quản trị vốn:
- Rủi ro cố định risk_per_trade theo equity hiện tại.
- qty = risk$ / khoảng cách tới stop; chặn notional <= max_leverage * equity.
- Stop ATR + trailing kiểu Chandelier (tùy chọn).
- Phí taker + slippage 2 chiều; funding mỗi 8h (xấp xỉ).
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
    entry_price: float
    qty: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0          # lãi/lỗ ròng (đã trừ phí + funding)
    fees: float = 0.0
    funding: float = 0.0
    r_multiple: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    symbol: str
    equity_curve: pd.Series
    trades: list = field(default_factory=list)
    config_label: str = ""


def _funding_due(open_time: pd.Timestamp) -> bool:
    """Funding Binance ~ 00:00 / 08:00 / 16:00 UTC. Khung 4h chạm các giờ này."""
    return open_time.hour in (0, 8, 16)


def run_backtest(df: pd.DataFrame, symbol: str, cfg, label: str = "") -> BacktestResult:
    """
    df: khung thực thi đã có open/high/low/close, atr, trigger_long/short, confl_long/short.
    """
    risk = cfg.risk
    strat = cfg.strategy

    equity = risk.initial_equity
    equity_curve = []
    trades: list[Trade] = []

    pos = None           # dict mô tả vị thế đang mở
    n = len(df)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    tlong = df["trigger_long"].to_numpy()
    tshort = df["trigger_short"].to_numpy()
    clong = df["confl_long"].to_numpy()
    cshort = df["confl_short"].to_numpy()
    state = df["state"].to_numpy()
    times = df.index

    pending = None  # tín hiệu chốt ở nến t, khớp ở nến t+1: ('long'/'short'/'exit')

    def open_position(side, price, time_i, atr_val):
        nonlocal equity
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            return None
        stop_dist = risk.atr_stop_mult * atr_val
        if stop_dist <= 0:
            return None
        risk_cash = equity * risk.risk_per_trade
        qty = risk_cash / stop_dist
        # chặn đòn bẩy
        max_notional = risk.max_leverage * equity
        if qty * price > max_notional:
            qty = max_notional / price
        fill = price * (1 + risk.slippage) if side == "long" else price * (1 - risk.slippage)
        fee = fill * qty * risk.taker_fee
        equity -= fee
        stop = fill - stop_dist if side == "long" else fill + stop_dist
        return {
            "side": side, "entry_price": fill, "qty": qty, "stop": stop,
            "entry_time": time_i, "atr": atr_val, "fees": fee, "funding": 0.0,
            "extreme": fill, "init_stop_dist": stop_dist,
        }

    def close_position(price, time_i, reason):
        nonlocal equity, pos
        side = pos["side"]
        qty = pos["qty"]
        fill = price * (1 - risk.slippage) if side == "long" else price * (1 + risk.slippage)
        fee = fill * qty * risk.taker_fee
        if side == "long":
            gross = (fill - pos["entry_price"]) * qty
        else:
            gross = (pos["entry_price"] - fill) * qty
        equity += gross - fee
        total_fees = pos["fees"] + fee
        pnl = gross - fee - pos["funding"]
        r = pnl / (pos["init_stop_dist"] * pos["qty"]) if pos["init_stop_dist"] > 0 else 0.0
        trades.append(Trade(
            symbol=symbol, side=side, entry_time=pos["entry_time"],
            entry_price=pos["entry_price"], qty=qty, exit_time=time_i,
            exit_price=fill, pnl=pnl, fees=total_fees, funding=pos["funding"],
            r_multiple=r, reason=reason,
        ))
        pos = None

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]

        # 1) Khớp lệnh pending (quyết định từ nến trước) ở open nến này
        if pending is not None:
            action = pending
            pending = None
            if action in ("long", "short"):
                if pos is not None and pos["side"] != action and strat.allow_flip:
                    close_position(o, times[i], "flip")
                if pos is None:
                    newpos = open_position(action, o, times[i], atr[i - 1] if i > 0 else atr[i])
                    if newpos:
                        pos = newpos
            elif action == "exit" and pos is not None:
                close_position(o, times[i], "signal_exit")

        # 2) Quản trị vị thế đang mở: funding, trailing, stop (trong nến)
        if pos is not None:
            if risk.apply_funding and _funding_due(times[i]):
                notional = pos["qty"] * c
                f = notional * risk.funding_rate  # long trả khi funding>0 (giả định +)
                f = f if pos["side"] == "long" else -f
                pos["funding"] += f
                equity -= f

            # cập nhật trailing chandelier
            if risk.use_trailing:
                if pos["side"] == "long":
                    pos["extreme"] = max(pos["extreme"], h)
                    trail = pos["extreme"] - risk.atr_trail_mult * pos["atr"]
                    pos["stop"] = max(pos["stop"], trail)
                else:
                    pos["extreme"] = min(pos["extreme"], l)
                    trail = pos["extreme"] + risk.atr_trail_mult * pos["atr"]
                    pos["stop"] = min(pos["stop"], trail)

            # kiểm tra chạm stop trong nến (ưu tiên trước tín hiệu)
            if pos["side"] == "long" and l <= pos["stop"]:
                close_position(pos["stop"], times[i], "stop")
            elif pos["side"] == "short" and h >= pos["stop"]:
                close_position(pos["stop"], times[i], "stop")

        # 3) Sinh tín hiệu trên CLOSE nến i -> khớp ở nến i+1
        if pos is None:
            if strat.allow_long and tlong[i] and clong[i] >= strat.confluence_n:
                pending = "long"
            elif strat.allow_short and tshort[i] and cshort[i] >= strat.confluence_n:
                pending = "short"
        else:
            # thoát khi mất phe trên khung thực thi
            flip_long = strat.allow_short and tshort[i] and cshort[i] >= strat.confluence_n
            flip_short = strat.allow_long and tlong[i] and clong[i] >= strat.confluence_n
            if pos["side"] == "long":
                if flip_long and strat.allow_flip:
                    pending = "short"
                elif strat.exit_on_state_flip and state[i] == -1:
                    pending = "exit"
                elif flip_long:
                    pending = "exit"
            else:
                if flip_short and strat.allow_flip:
                    pending = "long"
                elif strat.exit_on_state_flip and state[i] == 1:
                    pending = "exit"
                elif flip_short:
                    pending = "exit"

        # 4) Mark-to-market equity tại close
        if pos is not None:
            if pos["side"] == "long":
                unreal = (c - pos["entry_price"]) * pos["qty"]
            else:
                unreal = (pos["entry_price"] - c) * pos["qty"]
            equity_curve.append(equity + unreal)
        else:
            equity_curve.append(equity)

    # đóng vị thế còn mở ở cuối
    if pos is not None:
        close_position(closes[-1], times[-1], "eod")
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=df.index, name="equity")
    return BacktestResult(symbol=symbol, equity_curve=eq, trades=trades, config_label=label)
