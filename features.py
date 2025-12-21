"""Shared feature engineering for training and live inference."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "time": "time",
        "Time": "time",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "Vwap",
        "count": "Count",
    }
    renamed = df.rename(columns=columns)
    if "time" in renamed.columns:
        renamed = renamed.sort_values("time").reset_index(drop=True)
        renamed["time"] = pd.to_datetime(renamed["time"], utc=True)
        renamed = renamed.set_index("time")
    return renamed


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr_components = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    tr = tr_components.max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_supertrend(df: pd.DataFrame, atr_len: int, mult: float) -> pd.DataFrame:
    base = _normalize_ohlc(df)
    if base.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "st", "st_direction", "st_flip"])

    hl2 = (base["High"] + base["Low"]) / 2
    atr = _atr(base, atr_len)

    upperband = hl2 + mult * atr
    lowerband = hl2 - mult * atr

    st = pd.Series(index=base.index, dtype=float)
    direction = pd.Series(index=base.index, dtype=object)

    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = "bear"

    for i in range(1, len(base)):
        current_upper = upperband.iloc[i]
        current_lower = lowerband.iloc[i]

        if direction.iloc[i - 1] == "bull":
            current_upper = min(current_upper, st.iloc[i - 1])
        else:
            current_lower = max(current_lower, st.iloc[i - 1])

        close_price = base["Close"].iloc[i]
        if close_price > current_upper:
            st.iloc[i] = current_lower
            direction.iloc[i] = "bull"
        elif close_price < current_lower:
            st.iloc[i] = current_upper
            direction.iloc[i] = "bear"
        else:
            st.iloc[i] = st.iloc[i - 1]
            direction.iloc[i] = direction.iloc[i - 1]

    flip = direction.ne(direction.shift(1)).fillna(False)
    out = base.copy()
    out["st"] = st
    out["st_direction"] = direction
    out["st_flip"] = flip.astype(int)
    return out


def build_feature_frame(df: pd.DataFrame, cfg: dict, symbol: str, timeframe: str, st_params) -> pd.DataFrame:
    atr_len, mult, _meta = st_params if st_params else (
        cfg.get("supertrend_period", 10),
        cfg.get("supertrend_multiplier", 3.0),
        None,
    )

    base = _normalize_ohlc(df).tail(cfg.get("ML_TRAIN_LOOKBACK_BARS", 5000))
    if base.empty:
        return base

    ema_fast = base["Close"].ewm(span=cfg.get("ema_fast", 9), adjust=False).mean()
    ema_slow = base["Close"].ewm(span=cfg.get("ema_slow", 21), adjust=False).mean()
    atr_series = _atr(base, cfg.get("atr_period", 14))

    st_df = compute_supertrend(base, atr_len=atr_len, mult=mult)

    features = pd.DataFrame(index=base.index)
    features["close"] = base["Close"]
    features["ema_fast"] = ema_fast
    features["ema_slow"] = ema_slow
    features["ema_gap_pct"] = (ema_fast - ema_slow) / base["Close"]
    features["atr_pct"] = atr_series / base["Close"]
    features["st_dir_bull"] = (st_df["st_direction"] == "bull").astype(int)
    features["st_distance_pct"] = (base["Close"] - st_df["st"]) / base["Close"]
    features["ret_1"] = base["Close"].pct_change()
    features["ret_3"] = base["Close"].pct_change(3)
    features["vol_z"] = (base["Volume"] - base["Volume"].rolling(20).mean()) / base["Volume"].rolling(20).std()
    features["st_flip"] = st_df["st_flip"]

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features

