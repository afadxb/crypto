from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import config as CFG
from core.data_feed import DataFeed
from core.db import DB
from features import build_feature_frame
from supertrend_params import save_st_recommendations, select_supertrend_params
from utils import features_checksum, utc_now_iso


load_dotenv()


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def timestamp_folds(idx: pd.DatetimeIndex, n_folds: int = 3) -> Iterable[Fold]:
    if idx.empty:
        return []
    idx = pd.to_datetime(idx)
    cut_points = np.linspace(0, len(idx), n_folds + 2, dtype=int)[1:-1]
    folds = []
    for cp in cut_points:
        val_start = idx[cp]
        val_end = idx[min(len(idx) - 1, cp + max(1, len(idx) // (n_folds + 1)))]
        train_end = val_start
        train_start = idx[0]
        folds.append(Fold(train_start, train_end, val_start, val_end))
    return folds


def ensure_schema(db_path: str) -> None:
    DB(db_path).conn.close()


def make_dataset(df: pd.DataFrame, st_params: Tuple[int, float, dict | None]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    feats = build_feature_frame(df, {**CFG.INDICATORS, "ML_TRAIN_LOOKBACK_BARS": CFG.ML_TRAIN_LOOKBACK_BARS}, "", CFG.ML_TIMEFRAME_LABEL, st_params)
    if feats.empty:
        return feats, pd.Series(dtype=float), pd.Series(dtype=float)

    horizon = CFG.ML_LABEL_HORIZON_BARS
    feats["fwd_ret"] = feats["close"].shift(-horizon) / feats["close"] - 1
    feats["direction"] = (feats["fwd_ret"] > 0).astype(int)
    feats = feats.iloc[:-horizon].dropna()

    y = feats.pop("direction")
    fwd_ret = feats.pop("fwd_ret")
    feats = feats.drop(columns=["close"], errors="ignore")
    return feats, y, fwd_ret


def optuna_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, idx: pd.DatetimeIndex) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    scores = []
    for fold in timestamp_folds(idx):
        train_mask = (idx >= fold.train_start) & (idx < fold.train_end)
        val_mask = (idx >= fold.val_start) & (idx <= fold.val_end)
        if train_mask.sum() < 50 or val_mask.sum() < 20:
            continue
        model = XGBClassifier(**params)
        model.fit(X.loc[train_mask], y.loc[train_mask], verbose=False)
        proba = model.predict_proba(X.loc[val_mask])[:, 1]
        auc = roc_auc_score(y.loc[val_mask], proba)
        scores.append(auc)

    return float(np.nanmean(scores)) if scores else -1.0


def train_pair(feed: DataFeed, pair: str) -> dict:
    raw = feed.fetch_ohlc(pair, interval=CFG.TRADING_PARAMS["ohlc_interval"])
    df = raw.tail(CFG.ML_TRAIN_LOOKBACK_BARS).copy()
    timeframe = CFG.ML_TIMEFRAME_LABEL

    bars = df.copy()
    bars = bars.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    bars["time"] = pd.to_datetime(bars["time"], utc=True)
    bars = bars.set_index("time")

    st_row = select_supertrend_params(bars, cost_bps=6.0, timeframe=timeframe, symbol=pair)
    st_params: Tuple[int, float, dict | None]
    if st_row:
        save_st_recommendations(CFG.DB_PATH, [st_row])
        st_params = (st_row.atr_len, st_row.mult, {"score": st_row.score, "as_of": st_row.as_of, "run_id": st_row.run_id, "symbol": st_row.symbol})
    else:
        st_params = (CFG.INDICATORS["supertrend_period"], CFG.INDICATORS["supertrend_multiplier"], None)

    X, y, _ = make_dataset(df, st_params)
    if X.empty or y.empty:
        raise RuntimeError(f"Insufficient data to train for {pair}")

    idx = X.index
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: optuna_objective(t, X, y, idx), n_trials=10, show_progress_bar=False)
    best_params = {**study.best_params, "random_state": 42, "n_jobs": -1, "objective": "binary:logistic", "eval_metric": "logloss"}

    base = XGBClassifier(**best_params)
    base.fit(X, y, verbose=False)
    booster = base.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    ranked = sorted(gain_scores, key=gain_scores.get, reverse=True) or list(X.columns)
    top_k = min(20, len(ranked))
    pruned = ranked[:top_k]

    final = XGBClassifier(**best_params)
    final.fit(X[pruned], y, verbose=False)

    hold_cut = int(len(X) * 0.8)
    hold_X = X[pruned].iloc[hold_cut:]
    hold_y = y.iloc[hold_cut:]
    hold_proba = final.predict_proba(hold_X)[:, 1]
    hold_auc = float(roc_auc_score(hold_y, hold_proba)) if len(np.unique(hold_y)) > 1 else None

    out_dir = os.path.join(CFG.ML_MODEL_DIR, pair, timeframe)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.json")
    feats_path = os.path.join(out_dir, "features.txt")
    meta_path = os.path.join(out_dir, "meta.json")

    final.save_model(model_path)
    with open(feats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pruned))

    meta = {
        "pair": pair,
        "timeframe": timeframe,
        "trained_at_utc": utc_now_iso(),
        "train_start": idx.min().isoformat(),
        "train_end": idx.max().isoformat(),
        "features_checksum": features_checksum(pruned),
        "features_count": len(pruned),
        "holdout_auc": hold_auc,
        "params": best_params,
        "st_params": {"atr_len": st_params[0], "mult": st_params[1], "meta": st_params[2]},
        "ml_threshold": CFG.ML_PROBA_TH,
        "label_horizon_bars": CFG.ML_LABEL_HORIZON_BARS,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    conn = sqlite3.connect(CFG.DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO model_registry (pair, timeframe, model_version, trained_at_utc, train_start, train_end, features_checksum, features_count, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pair,
                timeframe,
                meta["trained_at_utc"],
                meta["trained_at_utc"],
                meta["train_start"],
                meta["train_end"],
                meta["features_checksum"],
                meta["features_count"],
                json.dumps(meta),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {"pair": pair, "model_path": model_path, "features": len(pruned), "holdout_auc": hold_auc}


def main() -> None:
    ensure_schema(CFG.DB_PATH)
    feed = DataFeed(os.getenv("KRAKEN_API_KEY", ""), os.getenv("KRAKEN_API_SECRET", ""), CFG.TRADING_PARAMS["ohlc_interval"])
    for pair in CFG.PAIRS:
        print(f"=== TRAIN {pair} ({CFG.ML_TIMEFRAME_LABEL}) ===")
        summary = train_pair(feed, pair)
        print(summary)


if __name__ == "__main__":
    main()
