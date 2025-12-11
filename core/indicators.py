"""Indicator calculations used by the trading strategy."""
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1)
    tr = tr_components.max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    """Calculate a non-repainting Supertrend indicator.

    Returns
    -------
    supertrend_line : pd.Series
    direction : pd.Series of "bull" | "bear"
    """
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=object)

    hl2 = (df["high"] + df["low"]) / 2
    atr = atr_wilder(df, period)

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=object)

    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = "bear"

    for i in range(1, len(df)):
        current_upper = upperband.iloc[i]
        current_lower = lowerband.iloc[i]

        if direction.iloc[i - 1] == "bull":
            current_upper = min(current_upper, st.iloc[i - 1])
        else:
            current_lower = max(current_lower, st.iloc[i - 1])

        close_price = df["close"].iloc[i]
        if close_price > current_upper:
            st.iloc[i] = current_lower
            direction.iloc[i] = "bull"
        elif close_price < current_lower:
            st.iloc[i] = current_upper
            direction.iloc[i] = "bear"
        else:
            st.iloc[i] = st.iloc[i - 1]
            direction.iloc[i] = direction.iloc[i - 1]

    return st, direction
