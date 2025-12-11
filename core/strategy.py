from .indicators import ema, atr_wilder, supertrend

def analyze(df, cfg):
    if len(df) < 120:
        return {"signal": "HOLD", "confidence": 0}

    c = df["close"].iloc[-2]
    bar_time = df["time"].iloc[-2]

    ema_fast_s = ema(df["close"], cfg["ema_fast"])
    ema_slow_s = ema(df["close"], cfg["ema_slow"])
    atr_s = atr_wilder(df, cfg["atr_period"])
    st_s, st_dir_s = supertrend(df, cfg["supertrend_period"], cfg["supertrend_multiplier"])

    ema_fast = ema_fast_s.iloc[-2]
    ema_slow = ema_slow_s.iloc[-2]
    atr_val = atr_s.iloc[-2]
    atr_pct = atr_val / c
    st_val = st_s.iloc[-2]
    st_dir = st_dir_s.iloc[-2]

    if atr_pct < cfg["atr_volatility_min_pct"]:
        return {
            "signal": "HOLD",
            "confidence": 0.2,
            "bar_time": bar_time,
            "st_dir": st_dir,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr_val,
            "atr_pct": atr_pct,
            "price": c,
        }

    bullish = st_dir == "bull" and c > st_val and ema_fast > ema_slow
    bearish = st_dir == "bear" and c < st_val and ema_fast < ema_slow

    if bullish:
        conf = 0.6
        if atr_pct >= 2 * cfg["atr_volatility_min_pct"]:
            conf = 0.8
        signal = "BUY"
    elif bearish:
        conf = 0.6
        if atr_pct >= 2 * cfg["atr_volatility_min_pct"]:
            conf = 0.8
        signal = "SELL"
    else:
        signal = "HOLD"
        conf = 0.4

    return {
        "signal": signal,
        "confidence": conf,
        "bar_time": bar_time,
        "st_dir": st_dir,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "atr": atr_val,
        "atr_pct": atr_pct,
        "price": c,
    }
