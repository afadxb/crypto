"""Central configuration for the Kraken trading bot."""
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


def _env_list(key: str, default):
    value = os.getenv(key)
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Trading pairs to monitor
PAIRS = _env_list("PAIRS", ["XBTUSD", "ETHUSD"])

# Database path
DB_PATH = os.getenv("DB_PATH", "bot.db")

# Local timezone for scheduling and timestamps
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOCAL_TZ = ZoneInfo(TIMEZONE)

# Trading parameters
TRADING_PARAMS = {
    "trading_interval": _env_int("TRADING_INTERVAL", 3600),  # seconds (1h)
    "position_size_usd": _env_float("POSITION_SIZE_USD", 50.0),
    "ohlc_interval": _env_int("OHLC_INTERVAL", 60),  # minutes for Kraken OHLC endpoint
    "limit_slippage_pct": _env_float("LIMIT_SLIPPAGE_PCT", 0.0005),
    "max_pair_exposure_usd": _env_float("MAX_PAIR_EXPOSURE_USD", 1000.0),
    "max_total_exposure_usd": _env_float("MAX_TOTAL_EXPOSURE_USD", 5000.0),
}

# Indicator and entry/exit parameters
INDICATORS = {
    "ema_fast": _env_int("EMA_FAST", 9),
    "ema_slow": _env_int("EMA_SLOW", 21),
    "atr_period": _env_int("ATR_PERIOD", 14),
    "atr_volatility_min_pct": _env_float("ATR_VOLATILITY_MIN_PCT", 0.003),
    "supertrend_period": _env_int("SUPERTREND_PERIOD", 10),
    "supertrend_multiplier": _env_float("SUPERTREND_MULTIPLIER", 3.0),
    "ema_gap_entry_pct": _env_float("EMA_GAP_ENTRY_PCT", 0.0015),
    "ema_gap_exit_pct": _env_float("EMA_GAP_EXIT_PCT", 0.0005),
    "ema_separation_threshold": _env_float("EMA_SEPARATION_THRESHOLD", 0.001),
}


def _derive_timeframe_label(minutes: int) -> str:
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours}h"
    return f"{minutes}m"


# Machine learning parameters
ML_ENABLED = bool(int(os.getenv("ML_ENABLED", "0")))
ML_MODEL_DIR = os.getenv("ML_MODEL_DIR", "models")
ML_PROBA_TH = _env_float("ML_PROBA_TH", 0.55)
ML_EXIT_PROBA_TH = _env_float("ML_EXIT_PROBA_TH", 0.45)
ML_MIN_ATR_PCT = _env_float("ML_MIN_ATR_PCT", INDICATORS["atr_volatility_min_pct"])
ML_TRAIN_LOOKBACK_BARS = _env_int("ML_TRAIN_LOOKBACK_BARS", 5000)
ML_LABEL_HORIZON_BARS = _env_int("ML_LABEL_HORIZON_BARS", 3)
ML_TIMEFRAME_LABEL = _derive_timeframe_label(TRADING_PARAMS["ohlc_interval"])
