"""Lightweight loader for persisted ML models and metadata."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, features_checksum


@dataclass
class ModelBundle:
    model: XGBClassifier
    features: List[str]
    meta: Dict


_cache: Dict[Tuple[str, str], ModelBundle] = {}


def load_model_bundle(pair: str, timeframe: str = "1h") -> Optional[ModelBundle]:
    key = (pair, timeframe)
    if key in _cache:
        return _cache[key]

    base = os.path.join("models", pair, timeframe)
    model_path = os.path.join(base, "model.json")
    feats_path = os.path.join(base, "features.txt")
    meta_path = os.path.join(base, "meta.json")

    if not os.path.exists(model_path) or not os.path.exists(feats_path):
        return None

    with open(feats_path, "r", encoding="utf-8") as f:
        features = [f.strip() for f in f.read().splitlines() if f.strip()]
    if not features:
        return None

    model = XGBClassifier()
    # xgboost>=2.1 requires the sklearn wrapper to have `_estimator_type` set
    # before calling `load_model`, otherwise it raises a TypeError when reading
    # the saved booster. Explicitly set it to keep loading models compatible
    # across library versions.
    if not getattr(model, "_estimator_type", None):
        model._estimator_type = "classifier"

    model.load_model(model_path)

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # ---- Feature alignment guard ----
    # If training features drift from live features, fail closed (no trading on bad inputs).
    trained_checksum = (meta or {}).get("features_checksum")
    live_checksum = features_checksum(FEATURE_COLUMNS)

    # Require both: (1) checksum match, (2) exact feature list match
    if trained_checksum and trained_checksum != live_checksum:
        return None
    if features != list(FEATURE_COLUMNS):
        return None

    bundle = ModelBundle(model=model, features=features, meta=meta)
    _cache[key] = bundle
    return bundle
