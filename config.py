"""
Cấu hình tập trung cho hệ thống backtest RSI đa khung (BTC/ETH - Binance Futures).

Chiến lược: trên mỗi khung tính RSI(14), EMA(9) của RSI và WMA(45) của RSI.
- State bullish của một khung: RSI > WMA45  (RSI nằm trên đường nền chậm)
- State bearish của một khung: RSI < WMA45
- Trigger long: RSI cắt LÊN trên cả EMA9 và WMA45 (vừa từ dưới lên)
- Trigger short: RSI cắt XUỐNG dưới cả EMA9 và WMA45
Vào lệnh khi trigger ở khung thực thi + đủ N khung cao đồng thuận state.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Thị trường & dữ liệu
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT"]

# Các khung Binance hỗ trợ native: 1h, 4h, 12h, 1d, 3d, 1w
ALL_TIMEFRAMES = ["1h", "4h", "12h", "1d", "3d", "1w"]

# Khung dùng để bấm cò vào/thoát lệnh
EXECUTION_TF = "4h"

# Các khung cao dùng để đếm đồng thuận (confluence). KHÔNG gồm execution_tf.
CONFLUENCE_TFS = ["12h", "1d", "3d", "1w"]

# Lịch sử kéo về (đủ bao gồm bull 2020-21, bear 2022, hồi 2023-24)
START_DATE = "2020-01-01"
END_DATE = None  # None = tới hiện tại

DATA_DIR = "data"
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Tham số chỉ báo
# ---------------------------------------------------------------------------
@dataclass
class IndicatorParams:
    rsi_period: int = 14
    ema_period: int = 9       # EMA của RSI (đường tín hiệu nhanh)
    wma_period: int = 45      # WMA của RSI (đường nền chậm)
    atr_period: int = 14      # ATR cho stop/trailing


# ---------------------------------------------------------------------------
# Tham số chiến lược
# ---------------------------------------------------------------------------
@dataclass
class StrategyParams:
    # Ngưỡng đồng thuận đa khung (số khung cao phải cùng phe)
    confluence_n: int = 2
    # Cho phép short? (futures long/short)
    allow_long: bool = True
    allow_short: bool = True
    # Thoát khi RSI khung thực thi mất phe (cắt ngược qua WMA45)
    exit_on_state_flip: bool = True
    # Đảo chiều ngay sang lệnh ngược khi có tín hiệu ngược (thay vì chỉ đóng)
    allow_flip: bool = False

    # --- Quản lý khối lượng động (scale-out / scale-in) ---
    # Mỗi khi nến khung thực thi (khung nhỏ nhất trong rổ) đóng, đánh giá rủi ro
    # đảo chiều bằng quan hệ RSI với EMA9(RSI):
    #   - STRONG : RSI còn trên EMA9        -> giữ full   (weight = 1.0)
    #   - CAUTION: RSI mất EMA9 nhưng > WMA45 -> giảm volume (weight = reduced_weight)
    #   - BROKEN : RSI < WMA45              -> thoát hẳn (weight = 0)
    # Khi khung nhỏ vào lại chu kỳ (RSI lấy lại EMA9) -> tăng volume về full.
    tiered_management: bool = True
    reduced_weight: float = 0.5       # tỉ lệ giữ lại khi có rủi ro đảo chiều
    min_rebalance_frac: float = 0.05  # bỏ qua điều chỉnh quá nhỏ (theo base_qty)


# ---------------------------------------------------------------------------
# Tham số rủi ro & chi phí (Binance USDT-M Futures)
# ---------------------------------------------------------------------------
@dataclass
class RiskParams:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01      # rủi ro 1% equity / lệnh
    atr_stop_mult: float = 2.5        # stop = entry ± 2.5*ATR
    atr_trail_mult: float = 3.0       # chandelier trailing = 3*ATR
    use_trailing: bool = True
    max_leverage: float = 5.0         # trần đòn bẩy cho notional
    taker_fee: float = 0.0004         # 0.04% taker / chiều
    slippage: float = 0.0003          # 0.03% trượt giá / chiều
    funding_rate: float = 0.0001      # ~0.01% / 8h (xấp xỉ; có thể fetch thật)
    apply_funding: bool = True


@dataclass
class BacktestConfig:
    symbols: list = field(default_factory=lambda: list(SYMBOLS))
    execution_tf: str = EXECUTION_TF
    confluence_tfs: list = field(default_factory=lambda: list(CONFLUENCE_TFS))
    indicators: IndicatorParams = field(default_factory=IndicatorParams)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    start: str = START_DATE
    end: str = END_DATE


# Số nến / năm để annualize Sharpe theo từng khung
BARS_PER_YEAR = {
    "1h": 24 * 365,
    "4h": 6 * 365,
    "12h": 2 * 365,
    "1d": 365,
    "3d": 365 / 3,
    "1w": 52,
}
