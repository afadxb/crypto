from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

import config as CFG
from core.data_feed import DataFeed
from core.ohlcv_store import OHLCVStore
from features import FEATURE_COLUMNS, build_feature_frame, features_checksum

load_dotenv()

TIMEFRAME_LABEL = "1h"

# ---- WFO / calibration knobs (env-driven, safe defaults) ----
WFO_FOLDS = int(os.getenv("WFO_FOLDS", "5"))
WFO_VAL_SIZE = int(os.getenv("WFO_VAL_SIZE", "120"))  # bars per fold validation
WFO_MIN_TRAIN = int(os.getenv("WFO_MIN_TRAIN", "300"))  # minimum bars before first split
CAL_METHOD = os.getenv("CAL_METHOD", "platt").strip().lower()  # "platt" or "isotonic"
CAL_MIN_SAMPLES = int(os.getenv("CAL_MIN_SAMPLES", "200"))
ML_AUC_FLOOR = float(os.getenv("ML_AUC_FLOOR", "0.52"))


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


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {}
    # logloss always defined
    out["logloss"] = float(log_loss(y_true, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y_true, p))
    # auc needs both classes present
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p))
    else:
        out["auc"] = float("nan")
    return out


def _build_time_folds(n: int, folds: int, val_size: int, min_train: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window folds:
      train = [0 : split)
      val   = [split : split+val_size)
    Splits are evenly spaced across the usable tail.
    """
    if n < (min_train + val_size + 1):
        return []
    usable = n - min_train - val_size
    if usable <= 0:
        return []
    folds = max(1, int(folds))
    # choose split points across [min_train, n - val_size)
    split_points = np.linspace(min_train, n - val_size, num=folds, dtype=int)
    out = []
    last_split = -1
    for split in split_points:
        if split <= last_split:
            continue
        train_idx = np.arange(0, split)
        val_idx = np.arange(split, split + val_size)
        if val_idx[-1] >= n:
            break
        out.append((train_idx, val_idx))
        last_split = split
    return out


def _fit_calibrator(oof_p: np.ndarray, oof_y: np.ndarray, method: str) -> Dict[str, Any]:
    """
    Fit calibrator on OOF predictions to avoid leakage.
    Returns a JSON-serializable dict describing the calibrator.
    """
    method = (method or "platt").lower()
    oof_p = np.asarray(oof_p, dtype=float)
    oof_y = np.asarray(oof_y, dtype=int)

    if len(oof_p) < CAL_MIN_SAMPLES or len(np.unique(oof_y)) < 2:
        return {"method": "none", "reason": "insufficient_samples_or_single_class"}

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_p, oof_y)
        # store knots; sklearn exposes x_thresholds_ / y_thresholds_ in newer versions,
        # but x_ / y_ are the safest across versions.
        return {
            "method": "isotonic",
            "x": iso.X_thresholds_.tolist() if hasattr(iso, "X_thresholds_") else iso.x_.tolist(),
            "y": iso.y_thresholds_.tolist() if hasattr(iso, "y_thresholds_") else iso.y_.tolist(),
        }

    # default: "platt" using logit(p) as feature
    Xc = _logit(oof_p).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(Xc, oof_y)
    return {"method": "platt", "coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}


def _apply_calibrator(p: np.ndarray, cal: Dict[str, Any]) -> np.ndarray:
    if not cal or cal.get("method") in (None, "none"):
        return np.asarray(p, dtype=float)
    method = cal.get("method", "").lower()
    p = np.asarray(p, dtype=float)
    if method == "isotonic":
        x = np.asarray(cal["x"], dtype=float)
        y = np.asarray(cal["y"], dtype=float)
        # piecewise linear interpolation of isotonic curve
        return np.interp(p, x, y, left=y[0], right=y[-1])
    if method == "platt":
        z = cal["coef"] * _logit(p) + cal["intercept"]
        return _sigmoid(z)
    return p


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

    # ---- Time-based WFO folds + out-of-fold predictions ----
    n = len(labeled)
    folds = _build_time_folds(n, WFO_FOLDS, WFO_VAL_SIZE, WFO_MIN_TRAIN)
    oof_p = np.full(shape=(n,), fill_value=np.nan, dtype=float)
    fold_rows: List[Dict[str, Any]] = []

    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # guard against single-class fold training
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            fold_rows.append(
                {
                    "fold": i,
                    "train_n": int(len(tr_idx)),
                    "val_n": int(len(va_idx)),
                    "skipped": True,
                    "reason": "single_class_in_train_or_val",
                    "train_start": labeled.index[tr_idx[0]].isoformat(),
                    "train_end": labeled.index[tr_idx[-1]].isoformat(),
                    "val_start": labeled.index[va_idx[0]].isoformat(),
                    "val_end": labeled.index[va_idx[-1]].isoformat(),
                }
            )
            continue

        m = _train_model(X_tr, y_tr)
        p_va = m.predict_proba(X_va)[:, 1]
        oof_p[va_idx] = p_va

        met = _metrics(y_va.to_numpy(), p_va)
        fold_rows.append(
            {
                "fold": i,
                "train_n": int(len(tr_idx)),
                "val_n": int(len(va_idx)),
                "skipped": False,
                "train_start": labeled.index[tr_idx[0]].isoformat(),
                "train_end": labeled.index[tr_idx[-1]].isoformat(),
                "val_start": labeled.index[va_idx[0]].isoformat(),
                "val_end": labeled.index[va_idx[-1]].isoformat(),
                "metrics": met,
            }
        )

    # Compute OOF metrics (base) on indices we actually predicted
    valid_mask = np.isfinite(oof_p)
    oof_base_metrics = {}
    cal = {"method": "none"}
    oof_cal_metrics = {}
    ml_quality: Dict[str, Any] = {
        "enabled": False,
        "auc_floor": float(ML_AUC_FLOOR),
        "oof_auc": None,
        "oof_n": 0,
        "reason": "insufficient_oof",
    }

    if valid_mask.any():
        oof_base_metrics = _metrics(y.to_numpy()[valid_mask], oof_p[valid_mask])
        cal = _fit_calibrator(oof_p[valid_mask], y.to_numpy()[valid_mask], CAL_METHOD)
        oof_p_cal = _apply_calibrator(oof_p[valid_mask], cal)
        oof_cal_metrics = _metrics(y.to_numpy()[valid_mask], oof_p_cal)

        # ---- Automatic per-pair ML enable/disable based on OOF base AUC ----
        oof_auc = float(oof_base_metrics.get("auc", float("nan")))
        enabled = bool(np.isfinite(oof_auc) and (oof_auc >= ML_AUC_FLOOR))
        ml_quality = {
            "enabled": enabled,
            "auc_floor": float(ML_AUC_FLOOR),
            "oof_auc": None if not np.isfinite(oof_auc) else float(oof_auc),
            "oof_n": int(valid_mask.sum()),
            "reason": "ok" if enabled else "auc_below_floor",
        }

        print(
            json.dumps(
                {
                    "pair": pair,
                    "wfo": {
                        "folds": len(folds),
                        "val_size": WFO_VAL_SIZE,
                        "min_train": WFO_MIN_TRAIN,
                        "oof_n": int(valid_mask.sum()),
                        "base_metrics": oof_base_metrics,
                        "calibration": {"method": cal.get("method", "none")},
                        "cal_metrics": oof_cal_metrics,
                    },
                    "ml_quality": ml_quality,
                }
            )
        )

    # ---- Final fit on all data (what you actually deploy) ----
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
        "wfo": {
            "folds_requested": WFO_FOLDS,
            "folds_built": len(folds),
            "val_size": WFO_VAL_SIZE,
            "min_train": WFO_MIN_TRAIN,
            "oof_n": int(valid_mask.sum()) if "valid_mask" in locals() else 0,
            "fold_details": fold_rows,
            "oof_base_metrics": oof_base_metrics,
            "oof_cal_metrics": oof_cal_metrics,
        },
        "calibration": cal,
        "ml_quality": ml_quality,
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
