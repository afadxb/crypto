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


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip() not in {"0", "false", "False", ""}


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

# Indicator, entry/exit, and ML parameters
INDICATORS = {
    "ema_fast": _env_int("EMA_FAST", 12),
    "ema_slow": _env_int("EMA_SLOW", 26),
    "ATR_LEN": _env_int("ATR_LEN", _env_int("ATR_PERIOD", 14)),
    "atr_period": _env_int("ATR_PERIOD", 14),
    "atr_volatility_min_pct": _env_float("MIN_ATR_PCT", _env_float("ATR_VOLATILITY_MIN_PCT", 0.008)),
    "supertrend_period": _env_int("SUPERTREND_PERIOD", 10),
    "supertrend_multiplier": _env_float("SUPERTREND_MULTIPLIER", 3.0),
    "ema_gap_entry_pct": _env_float("EMA_GAP_ENTRY_PCT", 0.0015),
    "ema_gap_exit_pct": _env_float("EMA_GAP_EXIT_PCT", 0.0005),
    "ema_separation_threshold": _env_float("EMA_SEPARATION_THRESHOLD", 0.001),
    "vol_z_win": _env_int("VOL_Z_WIN", 20),
    "ml_enabled": _env_bool("ML_ENABLED", True),
    "ml_proba_th": _env_float("ML_PROBA_TH", 0.60),
    "ml_exit_proba_th": _env_float("ML_EXIT_PROBA_TH", 0.45),
    "min_atr_pct": _env_float("MIN_ATR_PCT", 0.008),
    "label_horizon_bars": _env_int("LABEL_HORIZON_BARS", 6),
    "label_atr_k": _env_float("LABEL_ATR_K", 0.5),
    "min_label_pct": _env_float("MIN_LABEL_PCT", 0.003),
}
