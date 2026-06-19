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

# Các khung Binance hỗ trợ native: 15m, 1h, 4h, 12h, 1d, 3d, 1w
ALL_TIMEFRAMES = ["15m", "1h", "4h", "12h", "1d", "3d", "1w"]

# Khung dùng để bấm cò VÀO/THOÁT lệnh
EXECUTION_TF = "4h"

# Khung MỊN hơn execution để quản lý tăng/giảm volume (scale-out/in).
# Ví dụ: execution=4h -> manage=1h; execution=1h -> manage=15m.
# Để None nếu muốn quản lý ngay trên execution_tf.
MANAGE_TF = "1h"

# Các khung cao dùng để đếm đồng thuận (confluence). KHÔNG gồm execution/manage.
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
    swing_window: int = 10    # cửa sổ tìm đỉnh/đáy gần nhất (nến execution_tf)


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

    # --- Lọc tín hiệu nâng cao ---
    filter_squeeze: bool = False     # bỏ qua trigger khi bar trước đang kẹp giữa 2 đường
    require_double_pattern: bool = False  # yêu cầu W/M trên RSI khung execution
    double_pattern_window: int = 30  # số nến execution để tìm W/M
    double_pattern_min_bounce: float = 8.0   # bounce tối thiểu giữa 2 đáy/đỉnh (RSI)
    double_pattern_tolerance: float = 5.0    # sai lệch tối đa giữa 2 đáy/đỉnh (RSI)

    # --- Bỏ qua tín hiệu đầu tiên trong 1 chu kì tạo đáy/đỉnh ---
    skip_first_signal: bool = False   # chỉ vào lệnh ở signal thứ 2 trong chu kì
    cycle_reset_bars: int = 2         # số nến execution bearish liên tiếp để reset chu kì


# ---------------------------------------------------------------------------
# Tham số rủi ro & chi phí (Binance USDT-M Futures)
# ---------------------------------------------------------------------------
@dataclass
class RiskParams:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01      # rủi ro 1% equity / lệnh
    atr_stop_mult: float = 2.5        # stop = entry ± 2.5*ATR (dùng khi use_swing_stop=False)
    atr_trail_mult: float = 3.0       # chandelier trailing = 3*ATR
    use_trailing: bool = True
    use_swing_stop: bool = True       # dùng đáy/đỉnh gần nhất làm stop thay vì ATR×mult
    max_swing_atr_mult: float = 4.0   # trần khoảng cách stop (lần ATR) khi dùng swing
    min_swing_atr_mult: float = 0.5   # sàn khoảng cách stop tối thiểu (lần ATR)
    use_fixed_rr: bool = False        # đặt TP cố định theo tỉ lệ R:R
    rr_ratio: float = 3.0             # TP = entry ± rr_ratio × stop_distance (1:3)
    max_leverage: float = 5.0         # trần đòn bẩy cho notional
    taker_fee: float = 0.0004         # 0.04% taker / chiều
    slippage: float = 0.0003          # 0.03% trượt giá / chiều
    funding_rate: float = 0.0001      # ~0.01% / 8h (xấp xỉ; có thể fetch thật)
    apply_funding: bool = True


@dataclass
class BacktestConfig:
    symbols: list = field(default_factory=lambda: list(SYMBOLS))
    execution_tf: str = EXECUTION_TF
    manage_tf: str = MANAGE_TF       # None -> quản lý ngay trên execution_tf
    confluence_tfs: list = field(default_factory=lambda: list(CONFLUENCE_TFS))
    indicators: IndicatorParams = field(default_factory=IndicatorParams)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    start: str = START_DATE
    end: str = END_DATE

    @property
    def base_tf(self) -> str:
        """Khung nền mà engine chạy theo = manage_tf nếu có, ngược lại execution_tf."""
        return self.manage_tf or self.execution_tf


# Số nến / năm để annualize Sharpe theo từng khung
BARS_PER_YEAR = {
    "15m": 4 * 24 * 365,
    "1h": 24 * 365,
    "4h": 6 * 365,
    "12h": 2 * 365,
    "1d": 365,
    "3d": 365 / 3,
    "1w": 52,
}
