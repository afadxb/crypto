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
