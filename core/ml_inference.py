"""Shared inference helpers for the live trading loop."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

import config as CFG
from features import build_feature_frame
from .ml_store import load_model


def infer_signal(df_ohlc: pd.DataFrame, pair: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = {"ml_proba": None, "ml_gate": False, "ml_confidence": 0.0, "ml_reason": "unavailable"}

    timeframe = CFG.ML_TIMEFRAME_LABEL
    try:
        model, feature_cols, meta = load_model(pair, timeframe)
    except FileNotFoundError as exc:
        result["ml_reason"] = str(exc)
        return result

    st_params = None
    if meta and meta.get("st_params"):
        st_meta = meta["st_params"]
        st_params = (st_meta.get("atr_len"), st_meta.get("mult"), st_meta)

    feats = build_feature_frame(df_ohlc, {**cfg, "ML_TRAIN_LOOKBACK_BARS": CFG.ML_TRAIN_LOOKBACK_BARS}, pair, timeframe, st_params)
    if len(feats) < 2:
        result["ml_reason"] = "insufficient_features"
        return result

    latest = feats.iloc[-2]
    row = latest.reindex(feature_cols)
    if row.isna().any():
        result["ml_reason"] = "nan_features"
        return result

    proba = float(model.predict_proba(row.to_frame().T)[:, 1][0])
    gate = proba >= CFG.ML_PROBA_TH
    confidence = float(abs(proba - 0.5) * 2)

    result.update({
        "ml_proba": proba,
        "ml_gate": bool(gate),
        "ml_confidence": confidence,
        "ml_reason": "ok",
    })
    return result

