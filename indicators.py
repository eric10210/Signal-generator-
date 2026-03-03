"""
indicators.py  —  Complete technical analysis engine
Covers: ATR · HV · Bollinger + Keltner Squeeze · RSI · MACD
        StochRSI · Williams %R · CCI · EMAs (9/20/50/100/200)
        ADX · Supertrend · Ichimoku · Volume · OBV · MFI
        VWAP · Pivots (Classic + Fibonacci) · Swing Levels
        Market Regime Classifier
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _tr(df: pd.DataFrame) -> pd.Series:
    prev = df['close'].shift(1)
    return pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev).abs(),
        (df['low']  - prev).abs(),
    ], axis=1).max(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df['atr']      = _tr(df).ewm(span=period, adjust=False).mean()
    df['atr_pct']  = df['atr'] / df['close'] * 100
    df['hv_20']    = df['close'].pct_change().rolling(20).std() * np.sqrt(365) * 100
    return df


def calculate_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df     = df.copy()
    sma    = df['close'].rolling(period).mean()
    sigma  = df['close'].rolling(period).std()
    atr14  = _tr(df).ewm(span=14, adjust=False).mean()
    df['bb_upper']   = sma + std_dev * sigma
    df['bb_mid']     = sma
    df['bb_lower']   = sma - std_dev * sigma
    df['bb_width']   = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
    df['bb_pct']     = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['kc_upper']   = sma + 1.5 * atr14
    df['kc_lower']   = sma - 1.5 * atr14
    df['bb_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df    = df.copy()
    d     = df['close'].diff()
    gain  = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    df['rsi']       = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    df['rsi_slope'] = df['rsi'].diff(3)
    return df


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.DataFrame:
    df = df.copy()
    ef = df['close'].ewm(span=fast, adjust=False).mean()
    es = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd']            = ef - es
    df['macd_signal']     = df['macd'].ewm(span=sig, adjust=False).mean()
    df['macd_hist']       = df['macd'] - df['macd_signal']
    df['macd_cross_up']   = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    return df


def calculate_stoch_rsi(df: pd.DataFrame, rp: int = 14, sp: int = 14, sk: int = 3, sd: int = 3) -> pd.DataFrame:
    df   = df.copy()
    d    = df['close'].diff()
    gain = d.clip(lower=0).ewm(span=rp, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=rp, adjust=False).mean()
    rsi  = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    rmin = rsi.rolling(sp).min()
    rmax = rsi.rolling(sp).max()
    raw  = (rsi - rmin) / (rmax - rmin).replace(0, np.nan)
    df['stoch_k']          = raw.rolling(sk).mean() * 100
    df['stoch_d']          = df['stoch_k'].rolling(sd).mean()
    df['stoch_cross_up']   = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
    df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
    return df


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    hh = df['high'].rolling(period).max()
    ll = df['low'].rolling(period).min()
    df['williams_r'] = -100 * (hh - df['close']) / (hh - ll).replace(0, np.nan)
    return df


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df  = df.copy()
    tp  = (df['high'] + df['low'] + df['close']) / 3
    ma  = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['cci'] = (tp - ma) / (0.015 * mad.replace(0, np.nan))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TREND
# ══════════════════════════════════════════════════════════════════════════════

def calculate_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for s in [9, 20, 50, 100, 200]:
        df[f'ema_{s}'] = df['close'].ewm(span=s, adjust=False).mean()
    pairs = [(9,20),(20,50),(50,100),(100,200)]
    df['ema_aligned_bull'] = df.apply(lambda r: all(r[f'ema_{a}'] > r[f'ema_{b}'] for a,b in pairs), axis=1)
    df['ema_aligned_bear'] = df.apply(lambda r: all(r[f'ema_{a}'] < r[f'ema_{b}'] for a,b in pairs), axis=1)
    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df       = df.copy()
    tr       = _tr(df)
    up       = df['high'].diff()
    down     = -df['low'].diff()
    pdm      = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    ndm      = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_s    = tr.ewm(span=period, adjust=False).mean()
    pdi      = 100 * pdm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    ndi      = 100 * ndm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx       = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df['adx']      = dx.ewm(span=period, adjust=False).mean()
    df['plus_di']  = pdi
    df['minus_di'] = ndi
    return df


def calculate_supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    df  = df.copy()
    atr = _tr(df).ewm(span=period, adjust=False).mean()
    hl2 = (df['high'] + df['low']) / 2
    ub  = (hl2 + mult * atr).values
    lb  = (hl2 - mult * atr).values
    close = df['close'].values
    st  = np.full(len(df), np.nan)
    d   = np.ones(len(df), dtype=int)
    for i in range(1, len(df)):
        if close[i] > ub[i-1]: d[i] = 1
        elif close[i] < lb[i-1]: d[i] = -1
        else:
            d[i] = d[i-1]
            if d[i] == 1 and lb[i] < lb[i-1]: lb[i] = lb[i-1]
            if d[i] == -1 and ub[i] > ub[i-1]: ub[i] = ub[i-1]
        st[i] = lb[i] if d[i] == 1 else ub[i]
    df['supertrend']      = st
    df['supertrend_dir']  = d
    df['supertrend_flip'] = np.concatenate([[False], d[1:] != d[:-1]])
    return df


def calculate_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    h9  = df['high'].rolling(9).max();   l9  = df['low'].rolling(9).min()
    h26 = df['high'].rolling(26).max();  l26 = df['low'].rolling(26).min()
    h52 = df['high'].rolling(52).max();  l52 = df['low'].rolling(52).min()
    df['ichi_tenkan']     = (h9  + l9)  / 2
    df['ichi_kijun']      = (h26 + l26) / 2
    df['ichi_span_a']     = ((df['ichi_tenkan'] + df['ichi_kijun']) / 2).shift(26)
    df['ichi_span_b']     = ((h52 + l52) / 2).shift(26)
    df['ichi_chikou']     = df['close'].shift(-26)
    df['ichi_above_cloud']= (df['close'] > df['ichi_span_a']) & (df['close'] > df['ichi_span_b'])
    df['ichi_below_cloud']= (df['close'] < df['ichi_span_a']) & (df['close'] < df['ichi_span_b'])
    df['ichi_bull_cloud'] = df['ichi_span_a'] > df['ichi_span_b']
    df['ichi_tk_cross_up']= (df['ichi_tenkan'] > df['ichi_kijun']) & (df['ichi_tenkan'].shift(1) <= df['ichi_kijun'].shift(1))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VOLUME
# ══════════════════════════════════════════════════════════════════════════════

def calculate_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['vol_sma']  = df['volume'].rolling(20).mean()
    df['rvol']     = df['volume'] / df['vol_sma'].replace(0, np.nan)
    signed         = np.where(df['close'] > df['close'].shift(1),  df['volume'],
                     np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['obv']      = pd.Series(signed, index=df.index).cumsum()
    df['obv_ema']  = df['obv'].ewm(span=20, adjust=False).mean()
    df['obv_slope']= df['obv'].diff(3)
    tp             = (df['high'] + df['low'] + df['close']) / 3
    mf             = tp * df['volume']
    pmf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    nmf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    df['mfi']      = 100 - (100 / (1 + pmf / nmf.replace(0, np.nan)))
    df['vwap']     = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_pivots(df: pd.DataFrame, lookback: int = 20) -> dict:
    r   = df.tail(lookback)
    H, L, C = r['high'].max(), r['low'].min(), r['close'].iloc[-1]
    P   = (H + L + C) / 3
    rng = H - L
    levels = {
        'pivot': P,
        'r1': 2*P - L,  'r2': P + rng,  'r3': H + 2*(P-L),
        's1': 2*P - H,  's2': P - rng,  's3': L - 2*(H-P),
        'fib_236': H - 0.236*rng,
        'fib_382': H - 0.382*rng,
        'fib_500': H - 0.500*rng,
        'fib_618': H - 0.618*rng,
        'fib_786': H - 0.786*rng,
    }
    return levels


def find_swing_levels(df: pd.DataFrame, lookback: int = 100, tol: float = 0.005) -> dict:
    recent = df.tail(lookback)
    highs, lows = [], []
    for i in range(2, len(recent) - 2):
        h = recent['high'].iloc[i]
        l = recent['low'].iloc[i]
        if h == recent['high'].iloc[i-2:i+3].max(): highs.append(h)
        if l == recent['low'].iloc[i-2:i+3].min():  lows.append(l)

    def cluster(vals):
        if not vals: return []
        vals = sorted(vals)
        clusters, grp = [], [vals[0]]
        for v in vals[1:]:
            if abs(v - grp[-1]) / grp[-1] < tol: grp.append(v)
            else: clusters.append(float(np.mean(grp))); grp = [v]
        clusters.append(float(np.mean(grp)))
        return clusters

    return {'resistance_zones': cluster(highs), 'support_zones': cluster(lows)}


# ══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME
# ══════════════════════════════════════════════════════════════════════════════

def calculate_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    df      = df.copy()
    adx_v   = df['adx'].iloc[-1]     if 'adx'     in df.columns else 20.0
    hv      = df['hv_20'].iloc[-1]   if 'hv_20'   in df.columns else np.nan
    hv_avg  = df['hv_20'].rolling(50).mean().iloc[-1] if 'hv_20' in df.columns else np.nan
    e_bull  = bool(df['ema_aligned_bull'].iloc[-1]) if 'ema_aligned_bull' in df.columns else False
    e_bear  = bool(df['ema_aligned_bear'].iloc[-1]) if 'ema_aligned_bear' in df.columns else False
    vol_spike = (not np.isnan(hv) and not np.isnan(hv_avg) and hv > hv_avg * 1.5)

    if vol_spike and adx_v < 25:      regime = 'VOLATILE'
    elif adx_v >= 30 and e_bull:      regime = 'TRENDING_UP'
    elif adx_v >= 30 and e_bear:      regime = 'TRENDING_DOWN'
    elif adx_v >= 25:                 regime = 'TRENDING'
    else:                             regime = 'RANGING'

    df['market_regime'] = regime
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MASTER
# ══════════════════════════════════════════════════════════════════════════════

def enrich(df: pd.DataFrame) -> pd.DataFrame:
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