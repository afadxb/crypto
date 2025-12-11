"""Central configuration for the Kraken trading bot."""
import os

# Trading pairs to monitor
PAIRS = ["XBTUSD", "ETHUSD"]

# Database path
DB_PATH = os.getenv("DB_PATH", "bot.db")

# Trading parameters
TRADING_PARAMS = {
    "trading_interval": 3600,  # seconds (1h)
    "position_size_usd": 50.0,
    "ohlc_interval": 60,  # minutes for Kraken OHLC endpoint
    "limit_slippage_pct": 0.0005,
    "max_pair_exposure_usd": 1000.0,
    "max_total_exposure_usd": 5000.0,
}

# Indicator parameters
INDICATORS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "atr_volatility_min_pct": 0.003,
    "supertrend_period": 10,
    "supertrend_multiplier": 3.0,
    "ema_gap_entry_pct": 0.0015,
    "ema_gap_exit_pct": 0.0005,
}
