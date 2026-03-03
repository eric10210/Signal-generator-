"""
indicators.py  v3
Full technical analysis suite — hardened against NaN / zero-division / short series.
ATR · HV · Bollinger+Keltner Squeeze · RSI+slope · MACD+cross flags
StochRSI+cross flags · Williams%R · CCI · EMAs(9/20/50/100/200)+alignment
ADX+DI · Supertrend+flip · Ichimoku cloud · Volume/OBV/MFI/VWAP
Pivots (classic+Fibonacci) · Swing S/R clustering · Market Regime
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b, fill=np.nan):
    """Element-wise safe division returning fill where b==0."""
    if isinstance(b, pd.Series):
        return a / b.replace(0, np.nan)
    return a / b if b != 0 else fill


def _tr(df: pd.DataFrame) -> pd.Series:
    prev = df['close'].shift(1)
    return pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - prev).abs(),
        (df['low']  - prev).abs(),
    ], axis=1).max(axis=1)


def _need(df: pd.DataFrame, n: int) -> bool:
    return len(df) >= n


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    tr = _tr(df)
    df['atr']     = tr.ewm(span=period, min_periods=1, adjust=False).mean()
    df['atr_pct'] = _safe_div(df['atr'], df['close']) * 100
    df['hv_20']   = df['close'].pct_change().rolling(20, min_periods=10).std() * np.sqrt(365) * 100
    return df


def calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df    = df.copy()
    sma   = df['close'].rolling(period, min_periods=10).mean()
    sigma = df['close'].rolling(period, min_periods=10).std()
    atr14 = _tr(df).ewm(span=14, min_periods=1, adjust=False).mean()

    df['bb_upper']   = sma + std_dev * sigma
    df['bb_mid']     = sma
    df['bb_lower']   = sma - std_dev * sigma
    width_denom      = sma.replace(0, np.nan)
    df['bb_width']   = (df['bb_upper'] - df['bb_lower']) / width_denom * 100
    band_range       = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_pct']     = (df['close'] - df['bb_lower']) / band_range
    df['kc_upper']   = sma + 1.5 * atr14
    df['kc_lower']   = sma - 1.5 * atr14
    df['bb_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    # Fill NaN squeeze with False
    df['bb_squeeze'] = df['bb_squeeze'].fillna(False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df   = df.copy()
    d    = df['close'].diff()
    gain = d.clip(lower=0).ewm(span=period, min_periods=1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=period, min_periods=1, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    df['rsi']       = (100 - 100 / (1 + rs)).fillna(50)
    df['rsi_slope'] = df['rsi'].diff(3).fillna(0)
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    df = df.copy()
    ef = df['close'].ewm(span=fast, min_periods=1, adjust=False).mean()
    es = df['close'].ewm(span=slow, min_periods=1, adjust=False).mean()
    df['macd']        = ef - es
    df['macd_signal'] = df['macd'].ewm(span=sig, min_periods=1, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    prev_macd = df['macd'].shift(1)
    prev_sig  = df['macd_signal'].shift(1)
    df['macd_cross_up']   = ((df['macd'] > df['macd_signal']) & (prev_macd <= prev_sig)).fillna(False)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (prev_macd >= prev_sig)).fillna(False)
    return df


def calculate_stoch_rsi(df: pd.DataFrame, rp: int = 14, sp: int = 14, sk: int = 3, sd: int = 3) -> pd.DataFrame:
    df   = df.copy()
    d    = df['close'].diff()
    gain = d.clip(lower=0).ewm(span=rp, min_periods=1, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=rp, min_periods=1, adjust=False).mean()
    rsi  = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)
    rmin = rsi.rolling(sp, min_periods=1).min()
    rmax = rsi.rolling(sp, min_periods=1).max()
    raw  = (rsi - rmin) / (rmax - rmin).replace(0, np.nan)
    df['stoch_k'] = (raw.rolling(sk, min_periods=1).mean() * 100).fillna(50)
    df['stoch_d'] = df['stoch_k'].rolling(sd, min_periods=1).mean().fillna(50)
    prev_k = df['stoch_k'].shift(1)
    prev_d = df['stoch_d'].shift(1)
    df['stoch_cross_up']   = ((df['stoch_k'] > df['stoch_d']) & (prev_k <= prev_d)).fillna(False)
    df['stoch_cross_down'] = ((df['stoch_k'] < df['stoch_d']) & (prev_k >= prev_d)).fillna(False)
    return df


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    hh = df['high'].rolling(period, min_periods=1).max()
    ll = df['low'].rolling(period, min_periods=1).min()
    df['williams_r'] = (-100 * (hh - df['close']) / (hh - ll).replace(0, np.nan)).fillna(-50)
    return df


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df  = df.copy()
    tp  = (df['high'] + df['low'] + df['close']) / 3
    ma  = tp.rolling(period, min_periods=5).mean()
    mad = tp.rolling(period, min_periods=5).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    df['cci'] = ((tp - ma) / (0.015 * mad.replace(0, np.nan))).fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TREND
# ─────────────────────────────────────────────────────────────────────────────

def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for s in [9, 20, 50, 100, 200]:
        df[f'ema_{s}'] = df['close'].ewm(span=s, min_periods=1, adjust=False).mean()
    pairs = [(9, 20), (20, 50), (50, 100), (100, 200)]
    df['ema_aligned_bull'] = df.apply(
        lambda r: all(r[f'ema_{a}'] > r[f'ema_{b}'] for a, b in pairs), axis=1
    )
    df['ema_aligned_bear'] = df.apply(
        lambda r: all(r[f'ema_{a}'] < r[f'ema_{b}'] for a, b in pairs), axis=1
    )
    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df   = df.copy()
    tr   = _tr(df)
    up   = df['high'].diff().fillna(0)
    down = (-df['low'].diff()).fillna(0)
    pdm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    ndm  = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_s = tr.ewm(span=period, min_periods=1, adjust=False).mean()
    pdi   = (100 * pdm.ewm(span=period, min_periods=1, adjust=False).mean()
             / atr_s.replace(0, np.nan)).fillna(0)
    ndi   = (100 * ndm.ewm(span=period, min_periods=1, adjust=False).mean()
             / atr_s.replace(0, np.nan)).fillna(0)
    denom = (pdi + ndi).replace(0, np.nan)
    dx    = (100 * (pdi - ndi).abs() / denom).fillna(0)
    df['adx']      = dx.ewm(span=period, min_periods=1, adjust=False).mean().fillna(0)
    df['plus_di']  = pdi
    df['minus_di'] = ndi
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    df    = df.copy()
    atr   = _tr(df).ewm(span=period, min_periods=1, adjust=False).mean()
    hl2   = (df['high'] + df['low']) / 2
    ub    = (hl2 + mult * atr).values.copy()
    lb    = (hl2 - mult * atr).values.copy()
    close = df['close'].values
    n     = len(df)
    st    = np.full(n, np.nan)
    d     = np.ones(n, dtype=int)

    for i in range(1, n):
        # Upper band
        if ub[i] > ub[i - 1] or close[i - 1] > ub[i - 1]:
            pass  # keep computed ub[i]
        else:
            ub[i] = ub[i - 1]
        # Lower band
        if lb[i] < lb[i - 1] or close[i - 1] < lb[i - 1]:
            pass  # keep computed lb[i]
        else:
            lb[i] = lb[i - 1]
        # Direction
        if close[i] > ub[i - 1]:
            d[i] = 1
        elif close[i] < lb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
        st[i] = lb[i] if d[i] == 1 else ub[i]

    df['supertrend']      = st
    df['supertrend_dir']  = d
    df['supertrend_flip'] = np.concatenate([[False], d[1:] != d[:-1]])
    return df


def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    h9  = df['high'].rolling(9,  min_periods=1).max()
    l9  = df['low'].rolling(9,   min_periods=1).min()
    h26 = df['high'].rolling(26, min_periods=1).max()
    l26 = df['low'].rolling(26,  min_periods=1).min()
    h52 = df['high'].rolling(52, min_periods=1).max()
    l52 = df['low'].rolling(52,  min_periods=1).min()

    df['ichi_tenkan']      = (h9  + l9)  / 2
    df['ichi_kijun']       = (h26 + l26) / 2
    df['ichi_span_a']      = ((df['ichi_tenkan'] + df['ichi_kijun']) / 2).shift(26)
    df['ichi_span_b']      = ((h52 + l52) / 2).shift(26)
    df['ichi_chikou']      = df['close'].shift(-26)

    sa = df['ichi_span_a'].ffill()
    sb = df['ichi_span_b'].ffill()

    df['ichi_above_cloud'] = (df['close'] > sa) & (df['close'] > sb)
    df['ichi_below_cloud'] = (df['close'] < sa) & (df['close'] < sb)
    df['ichi_bull_cloud']  = sa > sb

    prev_t = df['ichi_tenkan'].shift(1)
    prev_k = df['ichi_kijun'].shift(1)
    df['ichi_tk_cross_up'] = (
        (df['ichi_tenkan'] > df['ichi_kijun']) & (prev_t <= prev_k)
    ).fillna(False)
    df['ichi_tk_cross_dn'] = (
        (df['ichi_tenkan'] < df['ichi_kijun']) & (prev_t >= prev_k)
    ).fillna(False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME
# ─────────────────────────────────────────────────────────────────────────────

def calculate_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vol_sma        = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_sma']  = vol_sma
    df['rvol']     = (df['volume'] / vol_sma.replace(0, np.nan)).fillna(1.0)

    # OBV
    signed        = np.where(
        df['close'] > df['close'].shift(1),  df['volume'],
        np.where(df['close'] < df['close'].shift(1), -df['volume'], 0)
    )
    df['obv']      = pd.Series(signed, index=df.index).cumsum()
    df['obv_ema']  = df['obv'].ewm(span=20, min_periods=1, adjust=False).mean()
    df['obv_slope']= df['obv'].diff(3).fillna(0)

    # MFI
    tp  = (df['high'] + df['low'] + df['close']) / 3
    mf  = tp * df['volume']
    pmf = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=5).sum()
    nmf = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=5).sum()
    df['mfi']  = (100 - 100 / (1 + pmf / nmf.replace(0, np.nan))).fillna(50)

    # VWAP (session cumulative)
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum().replace(0, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORT / RESISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_pivots(df: pd.DataFrame, lookback: int = 20) -> dict:
    r   = df.tail(lookback)
    H   = float(r['high'].max())
    L   = float(r['low'].min())
    C   = float(r['close'].iloc[-1])
    P   = (H + L + C) / 3
    rng = H - L if H != L else 1.0
    return {
        'pivot':   P,
        'r1': 2*P - L,       'r2': P + rng,       'r3': H + 2*(P-L),
        's1': 2*P - H,       's2': P - rng,        's3': L - 2*(H-P),
        'fib_236': H - 0.236*rng,
        'fib_382': H - 0.382*rng,
        'fib_500': H - 0.500*rng,
        'fib_618': H - 0.618*rng,
        'fib_786': H - 0.786*rng,
    }


def find_swing_levels(df: pd.DataFrame, lookback: int = 100, tol: float = 0.005) -> dict:
    recent = df.tail(lookback).reset_index(drop=True)
    highs, lows = [], []
    for i in range(2, len(recent) - 2):
        h = float(recent['high'].iloc[i])
        l = float(recent['low'].iloc[i])
        window_h = recent['high'].iloc[i-2:i+3]
        window_l = recent['low'].iloc[i-2:i+3]
        if h == float(window_h.max()):
            highs.append(h)
        if l == float(window_l.min()):
            lows.append(l)

    def cluster(vals: list) -> list:
        if not vals:
            return []
        vals = sorted(vals)
        clusters, grp = [], [vals[0]]
        for v in vals[1:]:
            if grp[-1] > 0 and abs(v - grp[-1]) / grp[-1] < tol:
                grp.append(v)
            else:
                clusters.append(float(np.mean(grp)))
                grp = [v]
        clusters.append(float(np.mean(grp)))
        return clusters

    return {
        'resistance_zones': cluster(highs),
        'support_zones':    cluster(lows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME
# ─────────────────────────────────────────────────────────────────────────────

def calculate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    adx_v  = float(df['adx'].iloc[-1])  if 'adx'  in df.columns else 20.0
    e_bull = bool(df['ema_aligned_bull'].iloc[-1]) if 'ema_aligned_bull' in df.columns else False
    e_bear = bool(df['ema_aligned_bear'].iloc[-1]) if 'ema_aligned_bear' in df.columns else False

    hv     = np.nan
    hv_avg = np.nan
    if 'hv_20' in df.columns:
        hv     = float(df['hv_20'].iloc[-1])
        series = df['hv_20'].rolling(50, min_periods=20).mean()
        hv_avg = float(series.iloc[-1])

    vol_spike = (
        not np.isnan(hv) and not np.isnan(hv_avg)
        and hv_avg > 0 and hv > hv_avg * 1.5
    )

    if vol_spike and adx_v < 25:
        regime = 'VOLATILE'
    elif adx_v >= 30 and e_bull:
        regime = 'TRENDING_UP'
    elif adx_v >= 30 and e_bear:
        regime = 'TRENDING_DOWN'
    elif adx_v >= 25:
        regime = 'TRENDING'
    else:
        regime = 'RANGING'

    df['market_regime'] = regime
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Run all indicators in dependency order. Safe for any series >= 30 rows."""
    if len(df) < 10:
        return df
    df = calculate_atr(df)
    df = calculate_bollinger(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_stoch_rsi(df)
    df = calculate_williams_r(df)
    df = calculate_cci(df)
    df = calculate_emas(df)
    df = calculate_adx(df)
    df = calculate_supertrend(df)
    df = calculate_ichimoku(df)
    df = calculate_volume(df)
    df = calculate_market_regime(df)
    return df