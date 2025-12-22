from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBClassifier

import config as CFG
from core.data_feed import DataFeed
from core.ohlcv_store import OHLCVStore
from features import FEATURE_COLUMNS, build_feature_frame, features_checksum

load_dotenv()

TIMEFRAME_LABEL = "1h"


def _artifact_paths(pair: str) -> Tuple[Path, Path, Path]:
    base = Path("models") / pair / TIMEFRAME_LABEL
    base.mkdir(parents=True, exist_ok=True)
    return base / "model.json", base / "features.txt", base / "meta.json"


def _label_frame(df: pd.DataFrame) -> pd.DataFrame:
    horizon = CFG.INDICATORS["label_horizon_bars"]
    fwd_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    threshold = np.maximum(CFG.INDICATORS["min_label_pct"], CFG.INDICATORS["label_atr_k"] * df["atr_pct"])
    df = df.copy()
    df["label"] = (fwd_ret >= threshold).astype(int)
    df["fwd_ret"] = fwd_ret
    return df.iloc[:-horizon]


def _train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    # Newer xgboost versions dropped the sklearn mixins that set `_estimator_type`,
    # so ensure it's present to allow saving through the sklearn wrapper API.
    if not hasattr(model, "_estimator_type"):
        model._estimator_type = "classifier"
    model.fit(X, y, verbose=False)
    return model


def _prepare_ohlc(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    ohlc = raw_df[list(cols.keys())].rename(columns=cols)
    ohlc.index = pd.to_datetime(raw_df["time"], utc=True)
    return ohlc


def _df_from_store(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df = df.sort_values("ts")
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return _prepare_ohlc(df)


def _refresh_recent_from_api(pair: str, store: OHLCVStore) -> None:
    """Optionally refresh the last ~720 bars from Kraken for recency."""

    feed = DataFeed(
        os.getenv("KRAKEN_API_KEY", ""), os.getenv("KRAKEN_API_SECRET", ""), CFG.TRADING_PARAMS["ohlc_interval"]
    )
    recent = feed.fetch_ohlc(pair, interval=CFG.TRADING_PARAMS["ohlc_interval"])
    if recent.empty:
        return

    payload = []
    for _, row in recent.iterrows():
        ts = int(pd.to_datetime(row["time"]).timestamp())
        payload.append(
            {
                "pair": pair,
                "timeframe": TIMEFRAME_LABEL,
                "ts": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "vwap": float(row.get("vwap", 0)) if "vwap" in row else None,
                "trades": int(row.get("count", 0)) if "count" in row else None,
            }
        )

    store.upsert_ohlcv(payload, pair=pair, timeframe=TIMEFRAME_LABEL)


def _load_training_ohlc(pair: str, store: OHLCVStore) -> pd.DataFrame:
    _refresh_recent_from_api(pair, store)
    rows = store.fetch_ohlcv(pair, timeframe=TIMEFRAME_LABEL, limit=CFG.TRAIN_LOOKBACK_BARS)
    if not rows:
        raise RuntimeError(f"No OHLCV history found in SQLite for {pair}")
    return _df_from_store(rows)


def train_pair(pair: str) -> dict:
    store = OHLCVStore()
    ohlc = _load_training_ohlc(pair, store)
    raw_bars = len(ohlc)
    if raw_bars < 200:
        raise RuntimeError(f"Not enough data to train model for {pair}")
    feature_df = build_feature_frame(ohlc, CFG.INDICATORS, pair=pair, timeframe=TIMEFRAME_LABEL)
    after_indicators = len(feature_df)
    feature_df = feature_df.iloc[:-1]  # drop potentially forming bar
    feature_df.dropna(inplace=True)
    after_dropna = len(feature_df)
    labeled = _label_frame(feature_df)
    after_label_trim = len(labeled)
    final_samples = len(labeled)

    print(
        json.dumps(
            {
                "pair": pair,
                "raw_bars": raw_bars,
                "after_indicators": after_indicators,
                "after_dropna": after_dropna,
                "after_label_trim": after_label_trim,
                "final_samples": final_samples,
            }
        )
    )

    X = labeled[FEATURE_COLUMNS]
    y = labeled["label"]

    pos = int(y.sum())
    neg = int(len(y) - pos)
    pos_rate = float(pos / len(y)) if len(y) else 0.0
    print(json.dumps({"pair": pair, "pos": pos, "neg": neg, "pos_rate": pos_rate}))

    model = _train_model(X, y)

    model_path, feats_path, meta_path = _artifact_paths(pair)
    model.save_model(model_path)
    feats_path.write_text("\n".join(FEATURE_COLUMNS), encoding="utf-8")

    meta = {
        "pair": pair,
        "timeframe": TIMEFRAME_LABEL,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_start": labeled.index.min().isoformat(),
        "train_end": labeled.index.max().isoformat(),
        "label": {
            "horizon_bars": CFG.INDICATORS["label_horizon_bars"],
            "atr_k": CFG.INDICATORS["label_atr_k"],
            "min_label_pct": CFG.INDICATORS["min_label_pct"],
        },
        "thresholds": {
            "ml_proba_th": CFG.INDICATORS["ml_proba_th"],
            "ml_exit_proba_th": CFG.INDICATORS["ml_exit_proba_th"],
            "min_atr_pct": CFG.INDICATORS["min_atr_pct"],
        },
        "features_checksum": features_checksum(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "samples": len(labeled),
        "class_balance": {"pos": pos, "neg": neg, "pos_rate": pos_rate},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"pair": pair, "model_path": str(model_path), "samples": len(labeled)}


def main():
    results = []
    for pair in CFG.PAIRS:
        print(f"=== TRAIN {pair} ({TIMEFRAME_LABEL}) ===")
        res = train_pair(pair)
        results.append(res)
        print(res)
    return results


if __name__ == "__main__":
    main()
