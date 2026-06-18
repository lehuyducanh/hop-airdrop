"""
Tải dữ liệu OHLCV từ Binance (USDT-M Futures, fallback Spot) + cache parquet.

- Kéo klines theo lô 1000 nến, phân trang qua startTime.
- Cache theo (symbol, timeframe) tại data/<symbol>_<tf>.parquet.
- Có generator synthetic để chạy/kiểm thử khi không truy cập được Binance
  (ví dụ vùng bị chặn 451/403).

OHLCV trả về: DataFrame index = open_time (UTC, tz-naive), các cột
open, high, low, close, volume, close_time.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

FUTURES_BASE = "https://fapi.binance.com/fapi/v1/klines"
SPOT_BASE = "https://api.binance.com/api/v3/klines"

_TF_MS = {
    "15m": 15 * 60_000,
    "1h": 3_600_000,
    "4h": 4 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": 24 * 3_600_000,
    "3d": 3 * 24 * 3_600_000,
    "1w": 7 * 24 * 3_600_000,
}

# Quy tắc resample pandas tương ứng mỗi khung
_TF_RULE = {
    "15m": "15min", "1h": "1h", "4h": "4h", "12h": "12h",
    "1d": "1D", "3d": "3D", "1w": "1W",
}


def _to_ms(date_str: str | None) -> int | None:
    if date_str is None:
        return None
    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_klines(raw: list) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tbqv", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    return df.set_index("open_time")


def _fetch_paginated(base: str, symbol: str, interval: str,
                     start_ms: int | None, end_ms: int | None,
                     max_retries: int = 4) -> pd.DataFrame:
    out = []
    cur = start_ms
    step = _TF_MS[interval]
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if cur is not None:
            params["startTime"] = cur
        if end_ms is not None:
            params["endTime"] = end_ms

        raw = None
        for attempt in range(max_retries):
            try:
                r = requests.get(base, params=params, timeout=20)
                if r.status_code == 200:
                    raw = r.json()
                    break
                # 451/403: bị chặn vùng -> báo lỗi rõ ràng để dùng fallback
                if r.status_code in (403, 451):
                    raise PermissionError(
                        f"Binance chặn truy cập ({r.status_code}) cho {base}. "
                        f"Chạy ở vùng khác hoặc dùng dữ liệu cache/synthetic."
                    )
                time.sleep(2 ** attempt)
            except requests.RequestException:
                time.sleep(2 ** attempt)
        if raw is None:
            raise RuntimeError(f"Không lấy được dữ liệu {symbol} {interval} sau retry.")
        if not raw:
            break

        out.extend(raw)
        last_open = raw[-1][0]
        cur = last_open + step
        if len(raw) < 1000:
            break
        if end_ms is not None and cur >= end_ms:
            break
        time.sleep(0.25)  # tôn trọng rate limit

    if not out:
        return pd.DataFrame()
    df = _parse_klines(out)
    return df[~df.index.duplicated(keep="first")].sort_index()


def fetch_klines(symbol: str, interval: str, start: str | None = None,
                 end: str | None = None, market: str = "futures") -> pd.DataFrame:
    """Tải klines từ Binance. market = 'futures' (fapi) hoặc 'spot'."""
    base = FUTURES_BASE if market == "futures" else SPOT_BASE
    return _fetch_paginated(base, symbol, interval, _to_ms(start), _to_ms(end))


def load_ohlcv(symbol: str, interval: str, start: str | None = None,
               end: str | None = None, data_dir: str = "data",
               market: str = "futures", use_cache: bool = True) -> pd.DataFrame:
    """
    Tải OHLCV có cache. Ưu tiên parquet local; nếu thiếu thì fetch Binance.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}_{interval}.parquet")

    if use_cache and os.path.exists(path):
        df = pd.read_parquet(path)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    df = fetch_klines(symbol, interval, start, end, market=market)
    if not df.empty:
        df.to_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Synthetic data (chạy offline / kiểm thử engine khi Binance bị chặn)
# ---------------------------------------------------------------------------
def synthetic_ohlcv(symbol: str = "BTCUSDT", interval: str = "1h",
                    start: str = "2020-01-01", periods: int = 30_000,
                    seed: int = 7) -> pd.DataFrame:
    """
    Sinh OHLCV giả lập theo GBM có đổi chế độ (regime) để tín hiệu RSI thực sự kích hoạt.
    Chỉ dùng cho phát triển/kiểm thử, KHÔNG dùng cho kết quả thật.
    """
    rng = np.random.default_rng(seed + hash(symbol) % 1000)
    step = _TF_MS[interval]
    idx = pd.to_datetime(
        np.arange(periods) * step + _to_ms(start), unit="ms"
    )

    # Drift thay đổi theo từng đoạn để tạo bull/bear/sideways
    drift = np.zeros(periods)
    i = 0
    while i < periods:
        seg = rng.integers(200, 1200)
        mu = rng.choice([0.0006, 0.0, -0.0004, 0.0002, -0.0002])
        drift[i:i + seg] = mu
        i += seg
    vol = 0.012 if symbol == "BTCUSDT" else 0.016
    rets = drift + rng.normal(0, vol, periods)
    price0 = 8000.0 if symbol == "BTCUSDT" else 130.0
    close = price0 * np.exp(np.cumsum(rets))

    open_ = np.empty(periods)
    open_[0] = price0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, periods)))
    volume = rng.uniform(100, 1000, periods)

    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "close_time": idx + pd.Timedelta(milliseconds=step - 1),
    }, index=idx)
    df.index.name = "open_time"
    return df


def resample_ohlcv(df_base: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Gộp khung từ dữ liệu nền mịn hơn (dùng cho synthetic để các khung nhất quán)."""
    rule = _TF_RULE[interval]
    agg = df_base.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    step = _TF_MS[interval]
    agg["close_time"] = agg.index + pd.Timedelta(milliseconds=step - 1)
    return agg
