"""Utility helpers shared across the training and inference pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Iterable


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def features_checksum(features: Iterable[str]) -> str:
    """Stable checksum for a list of features so we can track schema drift."""
    joined = "|".join(list(features))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

