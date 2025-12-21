"""Shared feature engineering for training and live inference."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence
import pandas as pd

from core.indicators import atr_wilder, ema, supertrend

FEATURE_COLUMNS: list[str] = [
    "st_dir",
    "st_dist_pct",
    "atr_pct",
    "ema_gap_pct",
    "ret_1",
    "ret_3",
    "ret_6",
    "vol_z",
]


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def compute_supertrend(df: pd.DataFrame, atr_len: int, mult: float) -> pd.DataFrame:
    st_val, st_dir = supertrend(df, period=atr_len, multiplier=mult)
    out = pd.DataFrame(index=df.index)
    out["st_value"] = st_val
    out["st_dir_label"] = st_dir
    out["st_dir"] = out["st_dir_label"].map({"bull": 1, "bear": -1}).astype(float)
    out["st_direction"] = out["st_dir"]
    return out


def compute_indicators(ohlc: pd.DataFrame, cfg: Any, st_params: tuple[int, float, dict | None] | None = None) -> pd.DataFrame:
    """Compute indicators required for trading and ML features.

    Parameters
    ----------
    ohlc : pd.DataFrame
        DataFrame with Open, High, Low, Close, Volume columns and UTC datetime index.
    cfg : Any
        Config object or mapping with indicator parameters.
    st_params : optional
        Tuple of (atr_len, multiplier, meta) to override supertrend defaults.
    """

    df = ohlc.copy()
    df.columns = [c.capitalize() for c in df.columns]

    close = df["Close"]
    atr_len = st_params[0] if st_params else _get(cfg, "supertrend_period", _get(cfg, "ATR_LEN", 10))
    atr_mult = st_params[1] if st_params else _get(cfg, "supertrend_multiplier", 3.0)
    ema_fast_len = _get(cfg, "ema_fast", _get(cfg, "EMA_FAST", 12))
    ema_slow_len = _get(cfg, "ema_slow", _get(cfg, "EMA_SLOW", 26))
    atr_period = _get(cfg, "atr_period", _get(cfg, "ATR_LEN", 14))
    vol_z_win = _get(cfg, "vol_z_win", _get(cfg, "VOL_Z_WIN", 20))

    df_l = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})

    indicators = pd.DataFrame(index=df.index)
    indicators["ema_fast"] = ema(close, ema_fast_len)
    indicators["ema_slow"] = ema(close, ema_slow_len)
    indicators["ema_gap_pct"] = (indicators["ema_fast"] - indicators["ema_slow"]) / close

    indicators["atr"] = atr_wilder(df_l, period=atr_period)
    indicators["atr_pct"] = indicators["atr"] / close

    st_df = compute_supertrend(df_l, atr_len=atr_len, mult=atr_mult)
    indicators = indicators.join(st_df)
    indicators["st_dist_pct"] = (close - indicators["st_value"]) / close

    indicators["ret_1"] = close.pct_change(1)
    indicators["ret_3"] = close.pct_change(3)
    indicators["ret_6"] = close.pct_change(6)

    vol_roll = df["Volume"].rolling(window=vol_z_win, min_periods=vol_z_win)
    vol_mean = vol_roll.mean()
    vol_std = vol_roll.std(ddof=0)
    indicators["vol_z"] = (df["Volume"] - vol_mean) / vol_std

    indicators["Close"] = close
    indicators["Volume"] = df["Volume"]
    indicators["time"] = df.index
    return indicators


def build_feature_frame(
    ohlc: pd.DataFrame,
    cfg: Any,
    pair: str,
    timeframe: str,
    st_params: tuple[int, float, dict | None] | None = None,
) -> pd.DataFrame:
    indicators = compute_indicators(ohlc, cfg, st_params=st_params)
    feature_df = indicators.copy()
    feature_df["pair"] = pair
    feature_df["timeframe"] = timeframe
    return feature_df


def latest_closed_row(feature_df: pd.DataFrame) -> pd.Series:
    clean = feature_df.dropna()
    if len(clean) < 2:
        raise ValueError("Not enough data for a closed bar")
    return clean.iloc[-2]


def features_checksum(features: Sequence[str] | Iterable[str]) -> str:
    joined = "|".join(features)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
