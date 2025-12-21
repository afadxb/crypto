"""Model loading utilities with simple caching."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from xgboost import XGBClassifier

import config as CFG


_CACHE: dict[tuple[str, str], tuple[XGBClassifier, List[str], Dict[str, Any]]] = {}


def _paths(pair: str, timeframe: str) -> dict[str, str]:
    base = os.path.join(CFG.ML_MODEL_DIR, pair, timeframe)
    return {
        "model": os.path.join(base, "model.json"),
        "features": os.path.join(base, "features.txt"),
        "meta": os.path.join(base, "meta.json"),
    }


def load_model(pair: str, timeframe: str) -> Tuple[XGBClassifier, List[str], Dict[str, Any]]:
    key = (pair, timeframe)
    if key in _CACHE:
        return _CACHE[key]

    paths = _paths(pair, timeframe)
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} artifact for {pair} {timeframe}: {path}")

    model = XGBClassifier()
    model.load_model(paths["model"])

    with open(paths["features"], "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)

    _CACHE[key] = (model, feature_cols, meta)
    return _CACHE[key]

