"""
Tính các chỉ số hiệu suất từ equity curve + danh sách lệnh, kèm benchmark Buy&Hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BARS_PER_YEAR


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def compute_metrics(equity: pd.Series, trades: list, timeframe: str,
                    initial_equity: float) -> dict:
    bars_yr = BARS_PER_YEAR[timeframe]
    rets = equity.pct_change().dropna()

    total_return = float(equity.iloc[-1] / initial_equity - 1.0)
    n_bars = len(equity)
    years = n_bars / bars_yr if bars_yr else np.nan
    cagr = (equity.iloc[-1] / initial_equity) ** (1 / years) - 1 if years and years > 0 else np.nan

    ann = np.sqrt(bars_yr)
    sharpe = float(rets.mean() / rets.std() * ann) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * ann) if len(downside) and downside.std() > 0 else 0.0
    mdd = _max_drawdown(equity)
    calmar = float(cagr / abs(mdd)) if mdd != 0 and not np.isnan(cagr) else np.nan

    pnls = np.array([t.pnl for t in trades], dtype=float)
    n_trades = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / n_trades) if n_trades else 0.0
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf
    avg_r = float(np.mean([t.r_multiple for t in trades])) if n_trades else 0.0
    expectancy = float(pnls.mean()) if n_trades else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    return {
        "total_return_%": round(total_return * 100, 2),
        "CAGR_%": round(cagr * 100, 2) if not np.isnan(cagr) else None,
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "MaxDD_%": round(mdd * 100, 2),
        "Calmar": round(calmar, 2) if not np.isnan(calmar) else None,
        "n_trades": n_trades,
        "win_rate_%": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else None,
        "avg_R": round(avg_r, 3),
        "expectancy_$": round(expectancy, 2),
        "avg_win_$": round(avg_win, 2),
        "avg_loss_$": round(avg_loss, 2),
        "final_equity_$": round(float(equity.iloc[-1]), 2),
    }


def buy_and_hold(close: pd.Series, initial_equity: float, timeframe: str) -> dict:
    """Benchmark mua-và-giữ trên cùng giai đoạn."""
    equity = initial_equity * close / close.iloc[0]
    m = compute_metrics(equity, [], timeframe, initial_equity)
    return {k: m[k] for k in ("total_return_%", "CAGR_%", "Sharpe", "MaxDD_%", "final_equity_$")}
