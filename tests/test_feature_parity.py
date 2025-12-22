import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config as CFG
from features import FEATURE_COLUMNS, build_feature_frame, latest_closed_row


PAIR = "TESTPAIR"
TIMEFRAME = "1h"


def _load_sample():
    df = pd.read_csv("tests/data/sample_ohlc.csv")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ohlc = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    return ohlc


def test_feature_columns_present_and_clean():
    ohlc = _load_sample()
    frame = build_feature_frame(ohlc, CFG.INDICATORS, pair=PAIR, timeframe=TIMEFRAME)
    for col in FEATURE_COLUMNS:
        assert col in frame.columns

    row = latest_closed_row(frame)
    assert not row[FEATURE_COLUMNS].isna().any()


def test_feature_determinism():
    ohlc = _load_sample()
    frame_a = build_feature_frame(ohlc, CFG.INDICATORS, pair=PAIR, timeframe=TIMEFRAME)
    frame_b = build_feature_frame(ohlc, CFG.INDICATORS, pair=PAIR, timeframe=TIMEFRAME)
    pd.testing.assert_frame_equal(frame_a[FEATURE_COLUMNS].tail(5), frame_b[FEATURE_COLUMNS].tail(5))
