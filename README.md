# Hệ thống Backtest RSI Đa Khung — BTC/ETH (Binance Futures)

Backtest chiến lược giao dịch dựa trên **RSI(14)** cùng hai đường trung bình của
chính RSI — **EMA(9)** và **WMA(45)** — theo nguyên lý **đa khung thời gian
(multi-timeframe confluence)**, cho **BTCUSDT / ETHUSDT** trên **Binance USDT-M
Futures (long + short)**.

## Ý tưởng chiến lược

Trên mỗi khung tính 3 đường: `RSI`, `EMA9(RSI)` (tín hiệu nhanh), `WMA45(RSI)` (nền chậm).

- **State bullish** của một khung: `RSI > WMA45`. **Bearish**: `RSI < WMA45`.
- **Trigger long**: `RSI` cắt **lên** trên *cả* EMA9 và WMA45 (vừa từ dưới lên).
- **Trigger short**: `RSI` cắt **xuống** dưới *cả* hai đường.
- **Đồng thuận đa khung**: đếm số khung CAO đang cùng phe. Vào lệnh khi có
  trigger ở khung thực thi **và** ≥ `N` khung cao đồng thuận (test cả **N=2** và **N=3**).

Cấu trúc khung mặc định (sửa trong `config.py`):

| Vai trò | Khung |
|---|---|
| Thực thi (bấm cò) | `4h` |
| Đồng thuận (lọc) | `12h`, `1d`, `3d`, `1w` |

## Quản trị lệnh (Futures)

- Rủi ro cố định **1% equity/lệnh**, size theo khoảng cách stop, chặn đòn bẩy `max_leverage`.
- Stop **ATR×2.5**, trailing **Chandelier ATR×3** (tùy chọn).
- Phí taker **0.04%**, slippage **0.03%** mỗi chiều, **funding ~0.01%/8h**.
- Thoát khi: chạm stop / RSI khung thực thi mất phe / (tùy chọn) đảo chiều.

### Quản lý khối lượng động — scale-out / scale-in

Mỗi khi **nến khung thực thi đóng**, hệ thống đánh giá **rủi ro đảo chiều khung
nhỏ** qua quan hệ giữa RSI và đường nhanh EMA9(RSI), rồi điều chỉnh khối lượng
về **trọng số mục tiêu** (khớp ở open nến kế tiếp):

| Bậc | Điều kiện (vị thế long) | Hành động | Trọng số |
|---|---|---|---|
| **STRONG** | RSI còn trên EMA9 | giữ/khôi phục full | `1.0` |
| **CAUTION** | RSI mất EMA9 nhưng còn trên WMA45 | **giảm volume** | `reduced_weight` (0.5) |
| **BROKEN** | RSI xuống dưới WMA45 | **thoát hẳn** | `0` |

→ Khi khung nhỏ **vào lại chu kỳ** (RSI lấy lại EMA9), volume được **tăng trở
lại full** (không vượt quá size ban đầu). Phần scale-out hiện thực hóa PnL ngay;
phần scale-in dùng **giá vốn bình quân gia quyền**. Mỗi vòng lệnh vẫn chỉ tính là
1 trade; báo cáo có thêm `scale_outs` / `scale_ins`.

Bật/tắt và tinh chỉnh trong `config.py` → `StrategyParams.tiered_management`,
`reduced_weight`, `min_rebalance_frac`.

## Chống look-ahead (quan trọng)

- Tín hiệu chốt trên **close** nến `t`, lệnh khớp ở **open** nến `t+1`.
- State khung cao chỉ dùng **sau khi nến khung đó đã đóng** (`merge_asof` theo
  `close_time`) — không bao giờ "nhìn trộm" tương lai. Có test kiểm chứng.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
python run_backtest.py             # kéo data thật từ Binance Futures, test N=2 & N=3
python run_backtest.py --n 3       # chỉ N=3
python run_backtest.py --synthetic # data giả lập (chạy offline / kiểm thử engine)
```

> ⚠️ Một số vùng bị Binance chặn API (HTTP 451/403). Khi đó hãy chạy ở nơi không
> bị chặn, hoặc dùng `--synthetic` để kiểm thử engine. Code tự fallback synthetic
> nếu gặp chặn vùng.

Kết quả lưu ở `./results/`: `metrics.json` + biểu đồ equity curve từng coin/ngưỡng.

## Kiểm thử

```bash
python tests/test_core.py        # sanity: chỉ báo, no-lookahead, engine
```

## Cấu trúc

```
config.py            # tham số: symbols, khung, chỉ báo, rủi ro, phí, funding
run_backtest.py      # điểm chạy chính (N=2 & N=3, report, biểu đồ)
src/
  data.py            # tải Binance klines + cache parquet + synthetic
  indicators.py      # RSI(Wilder), EMA, WMA, ATR
  signals.py         # state/trigger từng khung + ghép đa khung (no-lookahead)
  engine.py          # backtest event-driven futures long/short
  metrics.py         # CAGR, Sharpe, Sortino, MaxDD, Calmar... + Buy&Hold
tests/test_core.py   # kiểm thử
```

## Lộ trình tối ưu (khuyến nghị)

1. **Walk-forward**: tối ưu 12 tháng → kiểm 3 tháng out-of-sample, lăn tới.
2. **Sensitivity grid**: quét `N∈{2,3}`, execution `{1h,4h}`, ATR mult `{2,2.5,3}`,
   WMA `{36,45,54}` — chọn vùng tham số **ổn định**, không phải điểm nhọn (tránh overfit).
3. **Funding thật**: thay funding hằng số bằng dữ liệu funding lịch sử của Binance.
4. Luôn so với **Buy & Hold** làm chuẩn.

> Miễn trừ trách nhiệm: đây là công cụ nghiên cứu/backtest, không phải lời khuyên đầu tư.
