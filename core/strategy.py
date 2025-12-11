"""Pure strategy logic for the Kraken bot."""
from typing import Any, Dict, Optional

from .indicators import ema, atr_wilder, supertrend


EMA_SEPARATION_THRESHOLD = 0.001  # 0.1%
LOW_CONFIDENCE = 0.2
BASE_CONFIDENCE = 0.6
HIGH_CONFIDENCE = 0.8


def analyze(df, cfg: Dict[str, Any], last_signal: Optional[str] = None) -> Dict[str, Any]:
    """Analyze market data and return a trading decision.

    Notes
    -----
    Uses the second-to-last (closed) candle for all decisions.
    """
    if len(df) < 120:
        return {"signal": "HOLD", "confidence": 0.0}

    decision_idx = -2
    price = df["close"].iloc[decision_idx]
    bar_time = df["time"].iloc[decision_idx]

    ema_fast_series = ema(df["close"], cfg["ema_fast"])
    ema_slow_series = ema(df["close"], cfg["ema_slow"])
    atr_series = atr_wilder(df, cfg["atr_period"])
    st_series, st_dir_series = supertrend(df, cfg["supertrend_period"], cfg["supertrend_multiplier"])

    ema_fast_val = ema_fast_series.iloc[decision_idx]
    ema_slow_val = ema_slow_series.iloc[decision_idx]
    atr_val = atr_series.iloc[decision_idx]
    atr_pct = atr_val / price if price else 0
    st_val = st_series.iloc[decision_idx]
    st_dir = st_dir_series.iloc[decision_idx]

    result = {
        "bar_time": bar_time,
        "price": price,
        "st_dir": st_dir,
        "st_value": st_val,
        "ema_fast": ema_fast_val,
        "ema_slow": ema_slow_val,
        "atr": atr_val,
        "atr_pct": atr_pct,
    }

    # Volatility filter
    if atr_pct < cfg["atr_volatility_min_pct"]:
        result.update({"signal": "HOLD", "confidence": LOW_CONFIDENCE})
        return result

    ema_gap_pct = abs(ema_fast_val - ema_slow_val) / price if price else 0

    is_bullish = st_dir == "bull" and price > st_val and ema_fast_val > ema_slow_val
    is_bearish = st_dir == "bear" and price < st_val and ema_fast_val < ema_slow_val

    if is_bullish:
        signal = "BUY"
        confidence = BASE_CONFIDENCE
    elif is_bearish:
        signal = "SELL"
        confidence = BASE_CONFIDENCE
    else:
        signal = "HOLD"
        confidence = 0.4

    ema_separation = abs(ema_fast_val - ema_slow_val) / price if price else 0
    if signal in {"BUY", "SELL"} and atr_pct >= 2 * cfg["atr_volatility_min_pct"] and ema_separation >= EMA_SEPARATION_THRESHOLD:
        confidence = HIGH_CONFIDENCE

    if signal == "BUY" and last_signal == "SELL" and ema_gap_pct < cfg.get("ema_gap_entry_pct", 0):
        signal = "HOLD"
        confidence = min(confidence, 0.4)
    elif signal == "SELL" and last_signal == "BUY" and ema_gap_pct < cfg.get("ema_gap_entry_pct", 0):
        signal = "HOLD"
        confidence = min(confidence, 0.4)

    result.update({"signal": signal, "confidence": confidence, "ema_gap_pct": ema_gap_pct})
    return result
