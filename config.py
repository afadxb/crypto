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
}

# Indicator parameters
INDICATORS = {
    "ema_fast": 9,
    "ema_slow": 21,
    "atr_period": 14,
    "atr_volatility_min_pct": 0.003,
    "supertrend_period": 10,
    "supertrend_multiplier": 3.0,
}
