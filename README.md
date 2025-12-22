# Kraken Bot Overview

This repository contains a simple Kraken trading bot that evaluates 1-hour candles with Supertrend, EMAs, and ATR-based volatility filtering. Configuration is driven by environment variables (see `.env.example`).

## Setup

1. Create a `.env` file using `.env.example` as a template and fill in your credentials/parameters.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the bot:
   ```bash
   python main.py
   ```

## Entry conditions

All signals are derived from the **previous closed candle** (`iloc[-2]`):

- **Volatility gate:** `ATR / close` must be at least `ATR_VOLATILITY_MIN_PCT` (default `0.003`).
- **BUY:** Supertrend direction is bullish, price is above the Supertrend band, and `EMA_FAST` > `EMA_SLOW`.
- **SELL:** Supertrend direction is bearish, price is below the Supertrend band, and `EMA_FAST` < `EMA_SLOW`.
- **Whipsaw protection:** If the signal flips against the last signal but the EMA gap is smaller than `EMA_GAP_ENTRY_PCT` (default `0.0015`), the bot holds instead of entering.
- **Confidence boost:** Confidence rises from 0.6 to 0.8 when both `ATR / close` is at least double `ATR_VOLATILITY_MIN_PCT` and the EMA separation exceeds `EMA_SEPARATION_THRESHOLD` (default `0.001`).

## Exit conditions

- The bot runs long-only. A SELL signal will only place an order if a LONG position exists; otherwise, it is treated as HOLD.
- There are no separate take-profit/stop-loss rules—exits occur when the strategy flips to bearish alignment on a closed candle.

## Tuning risk up/down

- **More selective (fewer trades):**
  - Increase `ATR_VOLATILITY_MIN_PCT` to demand higher volatility before trading.
  - Increase `EMA_SEPARATION_THRESHOLD` so the confidence boost (and thus higher conviction) only happens on stronger trends.
  - Increase `SUPERTREND_MULTIPLIER` to widen bands and reduce churn.
  - Raise `EMA_GAP_ENTRY_PCT` to delay entries until EMAs diverge further after a reversal.
- **More aggressive (more trades):**
  - Decrease `ATR_VOLATILITY_MIN_PCT` to allow trades in quieter markets.
  - Decrease `SUPERTREND_MULTIPLIER` to tighten bands and react sooner to direction changes.
  - Lower `EMA_GAP_ENTRY_PCT` so reversals trigger earlier, and consider lowering `EMA_SEPARATION_THRESHOLD` for more frequent high-confidence calls.
- **Position sizing and exposure:**
  - Use `POSITION_SIZE_USD`, `MAX_PAIR_EXPOSURE_USD`, and `MAX_TOTAL_EXPOSURE_USD` to scale absolute risk up or down without changing signal logic.

Adjust these values in your `.env` file to suit your risk tolerance and market conditions.

## Historical backfill and training

Kraken's REST OHLC endpoint only returns the most recent ~720 candles, so long-range training relies on the downloadable trade ZIPs Kraken publishes. Use `import_trades.py` to backfill SQLite with hourly OHLCV derived from those time-and-sales files:

```bash
python import_trades.py /path/to/kraken-trades.zip --pair XBTUSD
python import_trades.py /path/to/kraken-ethusd.zip --pair ETHUSD
```

The importer streams each CSV (or ZIP member) in chunks, aggregates to 1h candles, and upserts into the shared SQLite DB using primary keys `(exchange, pair, timeframe, ts)` to stay idempotent. Training then loads the most recent `TRAIN_LOOKBACK_BARS` (default 8000) from SQLite rather than the Kraken OHLC endpoint, while the live bot still polls the API for the freshest bars.

## Trading parameter reference

- `TRADING_INTERVAL`: Sleep between cycles in seconds (default `3600`). Keep this aligned with the bar interval to avoid reprocessing the same candle too frequently.
- `POSITION_SIZE_USD`: Target notional per new entry in USD terms. Lower this to reduce per-trade exposure; raise it to size up.
- `OHLC_INTERVAL`: Minute interval for Kraken OHLC fetches (default `60`). Must match the timeframe you’re evaluating; changing it also changes how indicators are computed.
- `LIMIT_SLIPPAGE_PCT`: Percentage added/subtracted from the signal price when submitting limit orders (default `0.0005`, i.e., 0.05%). Increase for more conservative fills; decrease to chase fills.
- `MAX_PAIR_EXPOSURE_USD`: Cap on total LONG exposure per pair in USD terms. New orders that would exceed this are skipped.
- `MAX_TOTAL_EXPOSURE_USD`: Global cap on aggregate LONG exposure across all pairs. Orders that would breach the cap are skipped.
- 
