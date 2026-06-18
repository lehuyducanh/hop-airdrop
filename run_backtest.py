#!/usr/bin/env python3
"""
Chạy backtest chiến lược RSI đa khung cho BTC/ETH trên Binance Futures.

Mặc định kéo data thật từ Binance (USDT-M Futures). Nếu vùng bị chặn (451/403)
hoặc dùng cờ --synthetic, sẽ chạy bằng dữ liệu giả lập để kiểm thử engine.

Ví dụ:
    python run_backtest.py                 # data thật, test N=2 và N=3
    python run_backtest.py --synthetic     # data giả lập (offline)
    python run_backtest.py --n 3           # chỉ N=3
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

import config as C
from src import data as datamod
from src.signals import prepare_signal_frame
from src.engine import run_backtest
from src.metrics import compute_metrics, buy_and_hold


def _needed_tfs(cfg) -> list[str]:
    """Tập khung cần nạp: nền (manage) + execution + confluence (không trùng)."""
    tfs = [cfg.base_tf, cfg.execution_tf] + list(cfg.confluence_tfs)
    seen, out = set(), []
    for tf in tfs:
        if tf not in seen:
            seen.add(tf)
            out.append(tf)
    return out


def load_all_timeframes(symbol: str, cfg, synthetic: bool) -> dict[str, pd.DataFrame]:
    """Trả về {tf: OHLCV} cho mọi khung cần dùng."""
    tfs = _needed_tfs(cfg)
    out = {}
    if synthetic:
        # sinh ở khung nền mịn nhất rồi gộp lên các khung thô hơn
        base = datamod.synthetic_ohlcv(symbol, cfg.base_tf,
                                       start=cfg.start or "2020-01-01", periods=40_000)
        for tf in tfs:
            out[tf] = base.copy() if tf == cfg.base_tf else datamod.resample_ohlcv(base, tf)
        return out

    for tf in tfs:
        df = datamod.load_ohlcv(symbol, tf, start=cfg.start, end=cfg.end,
                                data_dir=C.DATA_DIR, market="futures")
        if df.empty:
            raise RuntimeError(f"Không có dữ liệu {symbol} {tf}.")
        out[tf] = df
    return out


def run_for_n(cfg, synthetic: bool, n: int) -> dict:
    cfg.strategy.confluence_n = n
    rows = {}
    equity_curves = {}
    for symbol in cfg.symbols:
        ohlcv = load_all_timeframes(symbol, cfg, synthetic)
        sig = prepare_signal_frame(ohlcv, cfg)
        sig = sig.dropna(subset=["atr", "rsi_wma"])  # bỏ giai đoạn warmup chỉ báo
        res = run_backtest(sig, symbol, cfg, label=f"N={n}")
        m = compute_metrics(res.equity_curve, res.trades, cfg.base_tf,
                            cfg.risk.initial_equity)
        bh = buy_and_hold(sig["close"], cfg.risk.initial_equity, cfg.base_tf)
        m["scale_outs"] = res.n_scale_outs
        m["scale_ins"] = res.n_scale_ins
        rows[symbol] = {"strategy": m, "buy_hold": bh}
        equity_curves[symbol] = res.equity_curve
    return {"metrics": rows, "equity": equity_curves}


def print_report(label: str, result: dict):
    print(f"\n{'=' * 78}\n  KẾT QUẢ {label}\n{'=' * 78}")
    keys = ["total_return_%", "CAGR_%", "Sharpe", "Sortino", "MaxDD_%", "Calmar",
            "n_trades", "win_rate_%", "profit_factor", "avg_R",
            "scale_outs", "scale_ins", "final_equity_$"]
    for symbol, data in result["metrics"].items():
        m = data["strategy"]
        bh = data["buy_hold"]
        print(f"\n[{symbol}] Chiến lược:")
        for k in keys:
            print(f"    {k:<16}: {m.get(k)}")
        print(f"  [{symbol}] Buy & Hold:")
        for k in ("total_return_%", "CAGR_%", "Sharpe", "MaxDD_%"):
            print(f"    {k:<16}: {bh.get(k)}")


def save_outputs(all_results: dict, cfg):
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    # JSON metrics
    serial = {}
    for label, res in all_results.items():
        serial[label] = {s: {"strategy": d["strategy"], "buy_hold": d["buy_hold"]}
                         for s, d in res["metrics"].items()}
    with open(os.path.join(C.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)

    # Equity curve PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for label, res in all_results.items():
            for symbol, eq in res["equity"].items():
                plt.figure(figsize=(11, 5))
                eq.plot(label=f"{symbol} {label}")
                plt.title(f"Equity Curve - {symbol} ({label}, "
                          f"exec={cfg.execution_tf}, manage={cfg.base_tf})")
                plt.ylabel("Equity ($)")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                fn = os.path.join(C.RESULTS_DIR, f"equity_{symbol}_{label.replace('=', '')}.png")
                plt.savefig(fn, dpi=110)
                plt.close()
    except Exception as e:  # pragma: no cover
        print(f"(Bỏ qua vẽ biểu đồ: {e})")
    print(f"\nĐã lưu kết quả vào ./{C.RESULTS_DIR}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="Dùng dữ liệu giả lập (offline)")
    ap.add_argument("--n", type=int, default=None, help="Chỉ chạy 1 ngưỡng N cụ thể")
    args = ap.parse_args()

    cfg = C.BacktestConfig()
    ns = [args.n] if args.n else [2, 3]

    print(f"Execution TF : {cfg.execution_tf}")
    print(f"Manage TF    : {cfg.base_tf} (quản lý tăng/giảm volume)")
    print(f"Confluence   : {cfg.confluence_tfs}")
    print(f"Symbols      : {cfg.symbols}")

    all_results = {}
    for n in ns:
        try:
            res = run_for_n(cfg, args.synthetic, n)
        except PermissionError as e:
            print(f"\n[!] {e}\n[!] Chuyển sang dữ liệu synthetic để minh họa engine.\n")
            res = run_for_n(cfg, True, n)
        label = f"N={n}"
        all_results[label] = res
        print_report(label, res)

    save_outputs(all_results, cfg)


if __name__ == "__main__":
    main()
