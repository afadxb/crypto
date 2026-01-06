"""Pure strategy logic for the Kraken bot with an ML gate."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from features import build_feature_frame, latest_closed_row
from .ml_store import load_model_bundle


EMA_SEPARATION_THRESHOLD = 0.001  # 0.1%
LOW_CONFIDENCE = 0.2
BASE_CONFIDENCE = 0.6
HIGH_CONFIDENCE = 0.8
TIMEFRAME_LABEL = "1h"


def _as_indexed_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    if "time" not in df.columns:
        raise ValueError("OHLC data must include a 'time' column")
    ohlc = df[list(cols.keys())].rename(columns=cols)
    ohlc.index = pd.to_datetime(df["time"], utc=True)
    return ohlc


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def _apply_calibration(p_raw: float, cal: Dict[str, Any]) -> float:
    method = (cal.get("method") or "none").lower()
    if method in ("none", "", None):
        return p_raw
    if method == "platt":
        coef = float(cal.get("coef", 1.0))
        intercept = float(cal.get("intercept", 0.0))
        return _sigmoid(coef * _logit(p_raw) + intercept)
    if method == "isotonic":
        x = np.array(cal.get("x", []), dtype=float)
        y = np.array(cal.get("y", []), dtype=float)
        if len(x) < 2 or len(y) < 2:
            return p_raw
        return float(np.interp(p_raw, x, y, left=y[0], right=y[-1]))
    return p_raw


def _ml_score(row: pd.Series, pair: str, cfg: Dict[str, Any]):
    model_bundle = load_model_bundle(pair, TIMEFRAME_LABEL)
    if not model_bundle:
        return None, "missing_model"

    # Per-pair quality gate written by trainer (meta.json)
    mq = (model_bundle.meta or {}).get("ml_quality") or {}
    if mq.get("enabled") is False:
        auc = mq.get("oof_auc")
        auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "n/a"
        floor = mq.get("auc_floor", "n/a")
        return None, f"ml_disabled({mq.get('reason','disabled')}, oof_auc={auc_str}, floor={floor})"

    try:
        X = row[model_bundle.features].to_frame().T
        p_raw = float(model_bundle.model.predict_proba(X)[0, 1])

        cal = (model_bundle.meta or {}).get("calibration") or {}
        method = (cal.get("method") or "none").lower()

        p = _apply_calibration(p_raw, cal)
        return float(p), f"ok(cal={method}, raw={p_raw:.4f})"
    except Exception:
        return None, "inference_error"


def analyze(
    pair: str,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    last_signal: Optional[str] = None,
    position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze market data and return a trading decision.

    Notes
    -----
    Uses the second-to-last (closed) candle for all decisions.
    """
    if len(df) < 120:
        return {"signal": "HOLD", "confidence": 0.0}

    ohlc = _as_indexed_ohlc(df)
    features = build_feature_frame(ohlc, cfg, pair=pair, timeframe=TIMEFRAME_LABEL)
    try:
        row = latest_closed_row(features)
    except ValueError:
        return {"signal": "HOLD", "confidence": 0.0}

    price = row["Close"]
    bar_time = row["time"]

    atr_pct = row["atr_pct"]
    ema_gap_pct_signed = row["ema_gap_pct"]
    ema_gap_pct = abs(ema_gap_pct_signed)
    st_dir_label = row.get("st_dir_label") or ("bull" if row.get("st_dir", 0) > 0 else "bear")
    st_val = row["st_value"]

    result = {
        "bar_time": bar_time,
        "price": price,
        "st_dir": st_dir_label,
        "st_value": st_val,
        "ema_fast": row.get("ema_fast"),
        "ema_slow": row.get("ema_slow"),
        "atr": row.get("atr"),
        "atr_pct": atr_pct,
        "ema_gap_pct": ema_gap_pct_signed,
    }

    if atr_pct < cfg.get("min_atr_pct", 0.0):
        result.update({"signal": "HOLD", "confidence": LOW_CONFIDENCE, "ml_gate": "vol_filter"})
        return result

    is_bullish = st_dir_label == "bull" and price > st_val and row["ema_fast"] > row["ema_slow"]
    is_bearish = st_dir_label == "bear" and price < st_val and row["ema_fast"] < row["ema_slow"]

    has_position = position is not None and position.get("side") == "LONG" and position.get("size", 0) > 0

    if is_bullish:
        signal = "BUY"
        confidence = BASE_CONFIDENCE
    elif is_bearish and has_position:
        signal = "SELL"
        confidence = BASE_CONFIDENCE
    else:
        signal = "HOLD"
        confidence = 0.4

    ema_separation = ema_gap_pct
    separation_threshold = cfg.get("ema_separation_threshold", EMA_SEPARATION_THRESHOLD)
    if signal in {"BUY", "SELL"} and atr_pct >= 2 * cfg.get("min_atr_pct", 0) and ema_separation >= separation_threshold:
        confidence = HIGH_CONFIDENCE

    if signal == "BUY" and last_signal == "SELL" and ema_gap_pct < cfg.get("ema_gap_entry_pct", 0):
        signal = "HOLD"
        confidence = min(confidence, 0.4)
    elif signal == "SELL" and last_signal == "BUY" and ema_gap_pct < cfg.get("ema_gap_entry_pct", 0):
        signal = "HOLD"
        confidence = min(confidence, 0.4)

    ml_enabled = bool(cfg.get("ml_enabled", False))
    ml_proba = None
    ml_gate = "skipped"
    ml_reason = ""

    if ml_enabled:
        ml_proba, ml_reason = _ml_score(row, pair, cfg)
        if ml_proba is None:
            ml_gate = "fail"
            if signal == "BUY":
                signal = "HOLD"
                confidence = min(confidence, LOW_CONFIDENCE)
        else:
            if signal == "BUY" and ml_proba < cfg.get("ml_proba_th", 0.0):
                ml_gate = "fail"
                ml_reason = "below_entry_threshold"
                signal = "HOLD"
                confidence = min(confidence, LOW_CONFIDENCE)
            else:
                ml_gate = "pass"

    if has_position and ml_enabled and ml_proba is not None and ml_proba <= cfg.get("ml_exit_proba_th", 0.0):
        signal = "SELL"
        ml_gate = ml_gate or "pass"
        ml_reason = ml_reason or "exit_threshold"

    result.update({
        "signal": signal,
        "confidence": confidence,
        "ema_gap_pct_abs": ema_gap_pct,
        "ml_proba": ml_proba,
        "ml_gate": ml_gate,
        "ml_reason": ml_reason,
    })
    return result
