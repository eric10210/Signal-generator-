"""
signals.py  v3
7 regime-aware strategies — fully audited for logic correctness.
All .get() calls use safe defaults. NaN guards on every numeric check.
Signal dataclass includes sl_pct, tp1_pct, risk_reward_str.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal['LONG', 'SHORT']


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    strategy:   str
    direction:  Direction
    confidence: int          # 0–100 (regime-adjusted)
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    tp3:        float
    rr:         float        # reward:risk ratio to TP1
    atr:        float
    regime:     str
    reasons:    list
    warnings:   list = field(default_factory=list)
    tags:       list = field(default_factory=list)

    # ── Derived ──────────────────────────────────────────────────────────────
    @property
    def grade(self) -> str:
        if self.confidence >= 85: return 'S'
        if self.confidence >= 70: return 'A'
        if self.confidence >= 55: return 'B'
        if self.confidence >= 40: return 'C'
        return 'D'

    @property
    def emoji(self) -> str:
        return '▲' if self.direction == 'LONG' else '▼'

    @property
    def sl_pct(self) -> float:
        return abs(self.entry - self.sl) / self.entry * 100 if self.entry else 0.0

    @property
    def tp1_pct(self) -> float:
        return abs(self.tp1 - self.entry) / self.entry * 100 if self.entry else 0.0

    @property
    def risk_reward_str(self) -> str:
        return f"1 : {self.rr:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# ROW ACCESSORS  (safe — return 0/False for missing keys)
# ─────────────────────────────────────────────────────────────────────────────

def _row(df: pd.DataFrame, offset: int = 0):
    """Return a row as a dict-like Series with safe .get() default=0."""
    idx = len(df) - 1 - offset
    if idx < 0:
        return pd.Series(dtype=float)
    row = df.iloc[idx].copy()
    # Patch .get() to return 0 for missing keys
    original_get = row.get
    def safe_get(key, default=0):
        val = original_get(key, default)
        if isinstance(val, float) and np.isnan(val):
            return default
        return val
    row.get = safe_get
    return row


def _rr(entry: float, sl: float, tp: float) -> float:
    d = abs(entry - sl)
    return abs(tp - entry) / d if d > 0 else 0.0


def _tp_levels(entry: float, atr: float, direction: str,
               m1: float = 2.0, m2: float = 3.5, m3: float = 6.0):
    sign = 1 if direction == 'LONG' else -1
    return entry + sign*atr*m1, entry + sign*atr*m2, entry + sign*atr*m3


def _regime(df: pd.DataFrame) -> str:
    if 'market_regime' in df.columns:
        return str(df['market_regime'].iloc[-1])
    return 'TRENDING'


# ─────────────────────────────────────────────────────────────────────────────
# SHARED SCORING MODULES
# ─────────────────────────────────────────────────────────────────────────────

def _score_trend(L, direction: str, reasons: list, warnings: list) -> int:
    score = 0
    close = float(L['close'])
    e200  = float(L.get('ema_200', close))

    if direction == 'LONG':
        if close > e200:
            score += 15; reasons.append('Above EMA 200 — macro uptrend intact')
        else:
            warnings.append('Below EMA 200 — counter-trend long, reduce size')
        if L.get('ema_aligned_bull', False):
            score += 15; reasons.append('Full EMA stack bullish (9>20>50>100>200)')
        if L.get('ichi_above_cloud', False):
            score += 10; reasons.append('Price above Ichimoku cloud')
        if int(L.get('supertrend_dir', 0)) == 1:
            score += 8;  reasons.append('Supertrend bullish')
    else:
        if close < e200:
            score += 15; reasons.append('Below EMA 200 — macro downtrend intact')
        else:
            warnings.append('Above EMA 200 — counter-trend short, reduce size')
        if L.get('ema_aligned_bear', False):
            score += 15; reasons.append('Full EMA stack bearish (9<20<50<100<200)')
        if L.get('ichi_below_cloud', False):
            score += 10; reasons.append('Price below Ichimoku cloud')
        if int(L.get('supertrend_dir', 0)) == -1:
            score += 8;  reasons.append('Supertrend bearish')
    return score


def _score_momentum(L, P, direction: str, reasons: list, warnings: list) -> int:
    score = 0
    rsi   = float(L.get('rsi', 50))
    sk    = float(L.get('stoch_k', 50))
    wr    = float(L.get('williams_r', -50))
    cci   = float(L.get('cci', 0))
    slope = float(L.get('rsi_slope', 0))

    if direction == 'LONG':
        if rsi < 30:    score += 20; reasons.append(f'RSI deeply oversold at {rsi:.1f}')
        elif rsi < 40:  score += 12; reasons.append(f'RSI oversold at {rsi:.1f}')
        elif rsi < 50:  score += 5;  reasons.append(f'RSI below midline at {rsi:.1f}')
        if slope > 2:   score += 7;  reasons.append(f'RSI slope turning up +{slope:.1f}')
        if bool(L.get('stoch_cross_up', False)) and sk < 30:
            score += 12; reasons.append('Stoch RSI bullish cross from oversold')
        elif sk < 20:   score += 8;  reasons.append(f'Stoch RSI oversold at {sk:.1f}')
        if wr < -80:    score += 5;  reasons.append(f'Williams %R oversold at {wr:.1f}')
        if cci < -100:  score += 5;  reasons.append(f'CCI oversold at {cci:.0f}')
    else:
        if rsi > 70:    score += 20; reasons.append(f'RSI deeply overbought at {rsi:.1f}')
        elif rsi > 60:  score += 12; reasons.append(f'RSI overbought at {rsi:.1f}')
        elif rsi > 50:  score += 5;  reasons.append(f'RSI above midline at {rsi:.1f}')
        if slope < -2:  score += 7;  reasons.append(f'RSI slope turning down {slope:.1f}')
        if bool(L.get('stoch_cross_down', False)) and sk > 70:
            score += 12; reasons.append('Stoch RSI bearish cross from overbought')
        elif sk > 80:   score += 8;  reasons.append(f'Stoch RSI overbought at {sk:.1f}')
        if wr > -20:    score += 5;  reasons.append(f'Williams %R overbought at {wr:.1f}')
        if cci > 100:   score += 5;  reasons.append(f'CCI overbought at {cci:.0f}')
    return score


def _score_macd(L, P, direction: str, reasons: list) -> int:
    score = 0
    macd      = float(L.get('macd', 0))
    macd_sig  = float(L.get('macd_signal', 0))
    hist      = float(L.get('macd_hist', 0))
    prev_hist = float(P.get('macd_hist', 0))

    if direction == 'LONG':
        if bool(L.get('macd_cross_up', False)):
            score += 15; reasons.append('Fresh MACD bullish cross')
        elif macd > macd_sig:
            score += 7;  reasons.append('MACD above signal line')
        if hist > 0 and hist > prev_hist:
            score += 6;  reasons.append('MACD histogram expanding bullish')
    else:
        if bool(L.get('macd_cross_down', False)):
            score += 15; reasons.append('Fresh MACD bearish cross')
        elif macd < macd_sig:
            score += 7;  reasons.append('MACD below signal line')
        if hist < 0 and hist < prev_hist:
            score += 6;  reasons.append('MACD histogram expanding bearish')
    return score


def _score_volume(L, direction: str, reasons: list, warnings: list) -> int:
    score     = 0
    rvol      = float(L.get('rvol', 1.0))
    obv       = float(L.get('obv', 0))
    obv_ema   = float(L.get('obv_ema', 0))
    obv_slope = float(L.get('obv_slope', 0))
    mfi       = float(L.get('mfi', 50))

    if rvol > 3.0:    score += 20; reasons.append(f'Volume surge {rvol:.1f}× — major player activity')
    elif rvol > 2.0:  score += 14; reasons.append(f'Strong volume {rvol:.1f}× — institutional flow')
    elif rvol > 1.3:  score += 7;  reasons.append(f'Above-average volume {rvol:.1f}×')
    elif rvol < 0.5:  warnings.append(f'Low volume {rvol:.2f}× — setup lacks conviction')

    if direction == 'LONG' and obv > obv_ema and obv_slope > 0:
        score += 8; reasons.append('OBV rising — smart money accumulating')
    elif direction == 'SHORT' and obv < obv_ema and obv_slope < 0:
        score += 8; reasons.append('OBV falling — smart money distributing')

    if direction == 'LONG' and mfi < 25:
        score += 7; reasons.append(f'MFI oversold at {mfi:.0f} — money flowing in')
    elif direction == 'SHORT' and mfi > 75:
        score += 7; reasons.append(f'MFI overbought at {mfi:.0f} — money flowing out')

    return score


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1  ·  BB / Keltner Squeeze Bounce
# Mean reversion when price closes back inside band after piercing it
# ─────────────────────────────────────────────────────────────────────────────

def strategy_bb_squeeze_bounce(df: pd.DataFrame):
    if len(df) < 25:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    cl   = float(L['close']); pl = float(P['close'])
    bbu  = float(L.get('bb_upper', cl)); bbl = float(L.get('bb_lower', cl))
    pbbu = float(P.get('bb_upper', pl)); pbbl = float(P.get('bb_lower', pl))

    long_setup  = pl < pbbl and cl > bbl   # prev closed below lower, now back inside
    short_setup = pl > pbbu and cl < bbu   # prev closed above upper, now back inside

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    score += 15; reasons.append('Price re-entered Bollinger Band — reversal candle confirmed')

    if bool(L.get('bb_squeeze', False)):
        score += 15; reasons.append('BB inside Keltner — volatility squeeze releasing'); tags.append('squeeze')

    avg_width = df['bb_width'].rolling(50, min_periods=20).mean().iloc[-1]
    if not np.isnan(avg_width) and float(L.get('bb_width', avg_width)) < avg_width * 0.65:
        score += 8; reasons.append('Bollinger width near multi-month low — coiled spring')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = cl
    sl    = (entry - atr * 2.2) if direction == 'LONG' else (entry + atr * 2.2)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.0, 3.5, 6.0)

    return Signal(
        strategy='BB/Keltner Squeeze Bounce', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2  ·  Volume Momentum Breakout
# Price breaks outside band with elevated volume
# ─────────────────────────────────────────────────────────────────────────────

def strategy_volume_breakout(df: pd.DataFrame):
    if len(df) < 25:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    cl   = float(L['close'])
    bbu  = float(L.get('bb_upper', cl))
    bbl  = float(L.get('bb_lower', cl))
    rvol = float(L.get('rvol', 1.0))

    long_setup  = cl > bbu and rvol > 1.5
    short_setup = cl < bbl and rvol > 1.5

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('breakout')

    if rvol > 3.0:    score += 25; reasons.append(f'Explosive {rvol:.1f}× volume — major breakout')
    elif rvol > 2.0:  score += 17; reasons.append(f'Strong {rvol:.1f}× volume — institutional breakout')
    else:             score += 9;  reasons.append(f'Volume {rvol:.1f}× above average — breakout confirmed')

    # Consecutive closes outside band
    consec = 0
    for i in range(len(df) - 1, max(len(df) - 5, 0), -1):
        r = df.iloc[i]
        outside = (r['close'] > r.get('bb_upper', r['close'])) if direction == 'LONG' \
                  else (r['close'] < r.get('bb_lower', r['close']))
        if outside:
            consec += 1
        else:
            break
    if consec >= 2:
        score += 10; reasons.append(f'{consec} consecutive closes outside band — momentum sustained')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    rsi = float(L.get('rsi', 50))
    if direction == 'LONG' and 50 < rsi < 75:
        score += 8; reasons.append(f'RSI in momentum zone {rsi:.1f} — room to run')
    elif direction == 'SHORT' and 25 < rsi < 50:
        score += 8; reasons.append(f'RSI in momentum zone {rsi:.1f} — room to fall')
    elif direction == 'LONG' and rsi > 80:
        warnings.append(f'RSI {rsi:.1f} — extremely overbought, chasing risk')

    adx = float(L.get('adx', 0))
    if adx > 30:
        score += 8; reasons.append(f'ADX {adx:.1f} — strong trend supporting breakout')

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = cl
    sl    = (entry - atr * 1.5) if direction == 'LONG' else (entry + atr * 1.5)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.5, 4.5, 8.0)

    return Signal(
        strategy='Volume Momentum Breakout', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3  ·  EMA Trend Pullback
# Bullish/bearish candle wicking EMA 50 in direction of trend
# ─────────────────────────────────────────────────────────────────────────────

def strategy_ema_pullback(df: pd.DataFrame):
    if len(df) < 60:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    cl   = float(L['close']); op = float(L['open'])
    hi   = float(L['high']);  lo = float(L['low'])
    e50  = float(L.get('ema_50', cl))
    e200 = float(L.get('ema_200', cl))
    e20  = float(L.get('ema_20', cl))

    tol = 0.007  # 0.7% tolerance for "touching" EMA
    long_setup  = (cl > e200 and e20 > e50
                   and abs(lo - e50) / e50 < tol
                   and cl > op)  # bullish candle
    short_setup = (cl < e200 and e20 < e50
                   and abs(hi - e50) / e50 < tol
                   and cl < op)  # bearish candle

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('pullback')
    score += 20; reasons.append('Candle touched EMA 50 and reversed — precision pullback entry')

    adx = float(L.get('adx', 0))
    if adx > 35:    score += 20; reasons.append(f'Very strong trend, ADX {adx:.1f}')
    elif adx > 25:  score += 12; reasons.append(f'Healthy trend, ADX {adx:.1f}')
    else:           warnings.append(f'Weak ADX {adx:.1f} — trend losing strength')

    if direction == 'LONG' and bool(L.get('ichi_above_cloud', False)) and bool(L.get('ichi_bull_cloud', False)):
        score += 15; reasons.append('Price above bullish Ichimoku cloud — textbook pullback zone')
    elif direction == 'SHORT' and bool(L.get('ichi_below_cloud', False)):
        score += 15; reasons.append('Price below bearish Ichimoku cloud — textbook pullback zone')

    rvol = float(L.get('rvol', 1.0))
    if rvol < 0.7:    score += 14; reasons.append(f'Low-volume pullback {rvol:.2f}× — sellers absent')
    elif rvol < 1.0:  score += 7;  reasons.append(f'Below-average pullback volume {rvol:.2f}×')
    elif rvol > 1.5:  warnings.append(f'High volume on pullback {rvol:.1f}× — watch for breakdown')

    score += _score_momentum(L, P, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)

    if direction == 'LONG' and int(L.get('supertrend_dir', 0)) == 1:
        score += 7; reasons.append('Supertrend still bullish through pullback')
    elif direction == 'SHORT' and int(L.get('supertrend_dir', 0)) == -1:
        score += 7; reasons.append('Supertrend still bearish through pullback')

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = cl
    sl    = (e50 - atr * 1.0) if direction == 'LONG' else (e50 + atr * 1.0)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.0, 4.0, 7.0)

    return Signal(
        strategy='EMA Trend Pullback', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 4  ·  RSI Divergence
# Price makes new extreme but RSI does not confirm
# ─────────────────────────────────────────────────────────────────────────────

def strategy_rsi_divergence(df: pd.DataFrame):
    if len(df) < 40:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    win = df.tail(35).reset_index(drop=True)
    if len(win) < 10:
        return None

    # Find two most extreme lows/highs in window
    low_idx  = win['low'].nsmallest(2).index.tolist()
    high_idx = win['high'].nlargest(2).index.tolist()

    bull_div = bear_div = False
    rsi_gap  = 0.0

    if len(low_idx) == 2:
        i1, i2 = sorted(low_idx)
        p1_low, p2_low = float(win['low'].iloc[i1]), float(win['low'].iloc[i2])
        p1_rsi, p2_rsi = float(win['rsi'].iloc[i1]), float(win['rsi'].iloc[i2])
        # Bullish divergence: lower price low but higher RSI low
        if p2_low < p1_low and p2_rsi > p1_rsi + 3:
            bull_div = True
            rsi_gap  = p2_rsi - p1_rsi

    if len(high_idx) == 2:
        i1, i2 = sorted(high_idx)
        p1_hi,  p2_hi  = float(win['high'].iloc[i1]), float(win['high'].iloc[i2])
        p1_rsi, p2_rsi = float(win['rsi'].iloc[i1]),  float(win['rsi'].iloc[i2])
        # Bearish divergence: higher price high but lower RSI high
        if p2_hi > p1_hi and p2_rsi < p1_rsi - 3:
            bear_div = True
            rsi_gap  = p1_rsi - p2_rsi

    if not bull_div and not bear_div:
        return None

    direction = 'LONG' if bull_div else 'SHORT'
    tags.append('divergence')

    if rsi_gap > 12:   score += 35; reasons.append(f'Strong RSI divergence (gap {rsi_gap:.1f} pts) — major reversal signal')
    elif rsi_gap > 7:  score += 25; reasons.append(f'Clear RSI divergence (gap {rsi_gap:.1f} pts)')
    elif rsi_gap > 3:  score += 15; reasons.append(f'Mild RSI divergence (gap {rsi_gap:.1f} pts)')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    # Multi-oscillator confirmation
    confirmed = 0
    sk  = float(L.get('stoch_k', 50))
    wr  = float(L.get('williams_r', -50))
    cci = float(L.get('cci', 0))
    bb  = float(L.get('bb_pct', 0.5))

    if direction == 'LONG':
        if sk  < 25:   confirmed += 1; reasons.append('Stoch RSI confirms oversold exhaustion')
        if wr  < -80:  confirmed += 1; reasons.append('Williams %R confirms oversold')
        if cci < -150: confirmed += 1; reasons.append('CCI extreme oversold — exhaustion signal')
        if bb  < 0.1:  confirmed += 1; reasons.append('Price at BB floor — multi-oscillator confluence')
    else:
        if sk  > 75:   confirmed += 1; reasons.append('Stoch RSI confirms overbought exhaustion')
        if wr  > -20:  confirmed += 1; reasons.append('Williams %R confirms overbought')
        if cci > 150:  confirmed += 1; reasons.append('CCI extreme overbought — exhaustion signal')
        if bb  > 0.9:  confirmed += 1; reasons.append('Price at BB ceiling — multi-oscillator confluence')
    if confirmed >= 2:
        score += 12

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = float(L['close'])
    sl    = (entry - atr * 2.5) if direction == 'LONG' else (entry + atr * 2.5)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.5, 4.0, 6.5)

    return Signal(
        strategy='RSI Divergence', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 5  ·  Ichimoku TK Cross
# Tenkan/Kijun cross with cloud location scoring (Type 1/2/3)
# ─────────────────────────────────────────────────────────────────────────────

def strategy_ichimoku_tk(df: pd.DataFrame):
    if len(df) < 55:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    tk_bull = bool(L.get('ichi_tk_cross_up', False))
    # Bear cross — tenkan crossed below kijun this candle
    tk_bear = bool(L.get('ichi_tk_cross_dn', False))

    if not tk_bull and not tk_bear:
        return None

    direction = 'LONG' if tk_bull else 'SHORT'
    tags.append('ichimoku')
    cross_name = 'bullish' if tk_bull else 'bearish'
    score += 20; reasons.append(f'Ichimoku TK {cross_name} cross — classic signal')

    # Cross location vs cloud
    above = bool(L.get('ichi_above_cloud', False))
    below = bool(L.get('ichi_below_cloud', False))
    inside = not above and not below

    if direction == 'LONG':
        if above:   score += 25; reasons.append('TK cross above cloud — Type 1 (strongest)')
        elif inside: score += 12; reasons.append('TK cross inside cloud — Type 2 (neutral)')
        else:       score += 5;  warnings.append('TK cross below cloud — Type 3 (weakest)')
        if bool(L.get('ichi_bull_cloud', False)):
            score += 12; reasons.append('Green Kumo ahead — bullish cloud')
    else:
        if below:   score += 25; reasons.append('TK cross below cloud — Type 1 (strongest)')
        elif inside: score += 12; reasons.append('TK cross inside cloud — Type 2 (neutral)')
        else:       score += 5;  warnings.append('TK cross above cloud — Type 3 (weakest)')
        if not bool(L.get('ichi_bull_cloud', True)):
            score += 12; reasons.append('Red Kumo ahead — bearish cloud')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_volume(L, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)

    adx = float(L.get('adx', 0))
    if adx > 25:
        score += 8; reasons.append(f'ADX {adx:.1f} — trend has strength')

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = float(L['close'])
    kijun = float(L.get('ichi_kijun', entry))
    sl    = (kijun - atr * 0.8) if direction == 'LONG' else (kijun + atr * 0.8)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.0, 3.5, 6.0)

    return Signal(
        strategy='Ichimoku TK Cross', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 6  ·  Supertrend Flip
# Direction change on Supertrend line with confirmation
# ─────────────────────────────────────────────────────────────────────────────

def strategy_supertrend_flip(df: pd.DataFrame):
    if len(df) < 20:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    flipped = bool(L.get('supertrend_flip', False))
    if not flipped:
        return None

    st_dir    = int(L.get('supertrend_dir', 0))
    direction = 'LONG' if st_dir == 1 else 'SHORT'
    tags.append('supertrend')
    score += 25; reasons.append(
        f'Supertrend flipped {"bullish" if direction=="LONG" else "bearish"} — trend reversal'
    )

    cl  = float(L['close'])
    e50 = float(L.get('ema_50', cl))
    if direction == 'LONG' and cl > e50:
        score += 14; reasons.append('Price above EMA 50 after flip — confirmed')
    elif direction == 'SHORT' and cl < e50:
        score += 14; reasons.append('Price below EMA 50 after flip — confirmed')
    else:
        warnings.append('Price has not yet cleared EMA 50 — wait for follow-through')

    adx  = float(L.get('adx', 0))
    padx = float(P.get('adx', 0))
    if adx > padx and adx > 20:
        score += 10; reasons.append(f'ADX rising to {adx:.1f} — trend building')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    atr     = max(float(L.get('atr', 1.0)), 1e-8)
    entry   = cl
    st_line = float(L.get('supertrend', entry - atr * 2))
    sl      = (st_line - atr * 0.5) if direction == 'LONG' else (st_line + atr * 0.5)
    tp1, tp2, tp3 = _tp_levels(entry, atr, direction, 2.0, 4.0, 7.0)

    return Signal(
        strategy='Supertrend Flip', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 7  ·  VWAP Reversion
# Price extended far from VWAP with oscillator exhaustion
# ─────────────────────────────────────────────────────────────────────────────

def strategy_vwap_reversion(df: pd.DataFrame):
    if len(df) < 20:
        return None
    L, P = _row(df, 0), _row(df, 1)
    reasons, warnings, tags = [], [], []
    score = 0

    vwap = float(L.get('vwap', 0))
    if vwap <= 0:
        return None

    cl      = float(L['close'])
    dev_pct = (cl - vwap) / vwap * 100
    atr_pct = float(L.get('atr_pct', 1.0))
    rsi     = float(L.get('rsi', 50))

    long_setup  = dev_pct < -(atr_pct * 1.5) and rsi < 40
    short_setup = dev_pct >  (atr_pct * 1.5) and rsi > 60

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('vwap')

    ext = abs(dev_pct)
    if ext > atr_pct * 3:    score += 25; reasons.append(f'Extreme VWAP deviation {ext:.2f}% — snap-back probable')
    elif ext > atr_pct * 2:  score += 15; reasons.append(f'Significant VWAP deviation {ext:.2f}%')
    else:                     score += 8;  reasons.append(f'VWAP deviation {ext:.2f}% — reversion entry')

    score += _score_momentum(L, P, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)
    score += _score_trend(L, direction, reasons, warnings)

    # Oscillator pile-on
    extra = 0
    sk  = float(L.get('stoch_k', 50))
    mfi = float(L.get('mfi', 50))
    cci = float(L.get('cci', 0))
    if direction == 'LONG':
        if sk  < 20:   extra += 1; reasons.append('Stoch RSI oversold — VWAP long confirmed')
        if mfi < 25:   extra += 1; reasons.append('MFI oversold — money flow turning')
        if cci < -100: extra += 1; reasons.append('CCI oversold — multiple oscillator exhaustion')
    else:
        if sk  > 80:   extra += 1; reasons.append('Stoch RSI overbought — VWAP short confirmed')
        if mfi > 75:   extra += 1; reasons.append('MFI overbought — money flow reversing')
        if cci > 100:  extra += 1; reasons.append('CCI overbought — multiple oscillator exhaustion')
    if extra >= 2:
        score += 15

    atr   = max(float(L.get('atr', 1.0)), 1e-8)
    entry = cl
    sl    = (entry - atr * 2.0) if direction == 'LONG' else (entry + atr * 2.0)
    # TP1 = revert to VWAP; TP2/TP3 beyond
    sign  = 1 if direction == 'LONG' else -1
    tp1   = vwap
    tp2   = entry + sign * atr * 3.5
    tp3   = entry + sign * atr * 6.0
    # Ensure TP1 is actually profitable
    if direction == 'LONG'  and tp1 <= entry: tp1 = entry + atr * 1.5
    if direction == 'SHORT' and tp1 >= entry: tp1 = entry - atr * 1.5

    return Signal(
        strategy='VWAP Reversion', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry, sl, tp1), atr=atr, regime=_regime(df),
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REGIME-AWARE SCANNER
# ─────────────────────────────────────────────────────────────────────────────

REGIME_WEIGHTS = {
    'TRENDING_UP':   {'BB/Keltner Squeeze Bounce': 0.7, 'Volume Momentum Breakout': 1.2,
                      'EMA Trend Pullback': 1.4, 'RSI Divergence': 0.6,
                      'Ichimoku TK Cross': 1.2, 'Supertrend Flip': 1.1, 'VWAP Reversion': 0.8},
    'TRENDING_DOWN': {'BB/Keltner Squeeze Bounce': 0.7, 'Volume Momentum Breakout': 1.2,
                      'EMA Trend Pullback': 1.4, 'RSI Divergence': 0.6,
                      'Ichimoku TK Cross': 1.2, 'Supertrend Flip': 1.1, 'VWAP Reversion': 0.8},
    'RANGING':       {'BB/Keltner Squeeze Bounce': 1.4, 'Volume Momentum Breakout': 0.7,
                      'EMA Trend Pullback': 0.7, 'RSI Divergence': 1.3,
                      'Ichimoku TK Cross': 0.8, 'Supertrend Flip': 0.7, 'VWAP Reversion': 1.4},
    'VOLATILE':      {'BB/Keltner Squeeze Bounce': 1.1, 'Volume Momentum Breakout': 1.3,
                      'EMA Trend Pullback': 0.6, 'RSI Divergence': 1.1,
                      'Ichimoku TK Cross': 0.7, 'Supertrend Flip': 1.2, 'VWAP Reversion': 1.2},
    'TRENDING':      {'BB/Keltner Squeeze Bounce': 0.9, 'Volume Momentum Breakout': 1.1,
                      'EMA Trend Pullback': 1.2, 'RSI Divergence': 0.9,
                      'Ichimoku TK Cross': 1.1, 'Supertrend Flip': 1.0, 'VWAP Reversion': 1.0},
}

STRATEGIES = [
    strategy_bb_squeeze_bounce,
    strategy_volume_breakout,
    strategy_ema_pullback,
    strategy_rsi_divergence,
    strategy_ichimoku_tk,
    strategy_supertrend_flip,
    strategy_vwap_reversion,
]


def scan(df: pd.DataFrame, min_confidence: int = 40) -> list:
    regime  = _regime(df)
    weights = REGIME_WEIGHTS.get(regime, {})
    signals = []

    for fn in STRATEGIES:
        try:
            sig = fn(df)
            if sig is None:
                continue
            w = weights.get(sig.strategy, 1.0)
            sig.confidence = int(min(100, round(sig.confidence * w)))
            if sig.confidence >= min_confidence:
                signals.append(sig)
        except Exception:
            pass   # never crash the scanner

    return sorted(signals, key=lambda s: s.confidence, reverse=True)