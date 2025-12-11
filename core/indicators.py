import pandas as pd

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def atr_wilder(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def supertrend(df, period=10, multiplier=3.0):
    hl2 = (df["high"] + df["low"]) / 2
    atr = atr_wilder(df, period)

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=object)

    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = upperband.iloc[i]
            direction.iloc[i] = "bear"
            continue

        if direction.iloc[i-1] == "bull":
            lower = max(lowerband.iloc[i], st.iloc[i-1])
            st.iloc[i] = lower
            direction.iloc[i] = "bull" if df["close"].iloc[i] > lower else "bear"
        else:
            upper = min(upperband.iloc[i], st.iloc[i-1])
            st.iloc[i] = upper
            direction.iloc[i] = "bear" if df["close"].iloc[i] < upper else "bull"

    return st, direction
