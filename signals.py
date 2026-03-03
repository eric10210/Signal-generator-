"""
signals.py  —  Professional signal engine
7 strategies, each with multi-factor confluence scoring.
Regime-aware: strategies are weighted by market condition.
Outputs ranked Signal objects with full trade plan.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

Direction = Literal['LONG', 'SHORT', 'NEUTRAL']
Regime    = Literal['TRENDING_UP', 'TRENDING_DOWN', 'TRENDING', 'RANGING', 'VOLATILE']


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    strategy:    str
    direction:   Direction
    confidence:  int            # 0–100
    entry:       float
    sl:          float
    tp1:         float
    tp2:         float
    tp3:         float
    rr:          float          # reward:risk to TP1
    atr:         float
    regime:      str
    reasons:     list[str]
    warnings:    list[str] = field(default_factory=list)
    tags:        list[str] = field(default_factory=list)   # e.g. ['squeeze','divergence']

    @property
    def emoji(self) -> str:
        return '▲' if self.direction == 'LONG' else '▼'

    @property
    def grade(self) -> str:
        if self.confidence >= 85: return 'S'
        if self.confidence >= 70: return 'A'
        if self.confidence >= 55: return 'B'
        if self.confidence >= 40: return 'C'
        return 'D'

    @property
    def sl_pct(self) -> float:
        return abs(self.entry - self.sl) / self.entry * 100

    @property
    def tp1_pct(self) -> float:
        return abs(self.tp1 - self.entry) / self.entry * 100

    @property
    def risk_reward_str(self) -> str:
        return f"1 : {self.rr:.1f}"


def _L(df):   return df.iloc[-1]
def _P(df):   return df.iloc[-2]
def _PP(df):  return df.iloc[-3]

def _rr(entry, sl, tp):
    d = abs(entry - sl)
    return abs(tp - entry) / d if d > 0 else 0.0

def _levels(entry, sl, atr, direction, tp_multiples=(2.0, 3.5, 6.0)):
    sign = 1 if direction == 'LONG' else -1
    tp1  = entry + sign * atr * tp_multiples[0]
    tp2  = entry + sign * atr * tp_multiples[1]
    tp3  = entry + sign * atr * tp_multiples[2]
    return tp1, tp2, tp3


# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONFLUENCE MODULES  (reusable across strategies)
# ══════════════════════════════════════════════════════════════════════════════

def _score_trend(L, direction, reasons, warnings) -> int:
    score = 0
    if direction == 'LONG':
        if L['close'] > L['ema_200']:
            score += 15; reasons.append('Above EMA 200 — macro uptrend intact')
        else:
            warnings.append('Below EMA 200 — counter-trend long')
        if L.get('ema_aligned_bull', False):
            score += 15; reasons.append('Full EMA stack bullish (9>20>50>100>200)')
        if L.get('ichi_above_cloud', False):
            score += 10; reasons.append('Price above Ichimoku cloud — strong trend')
        if L.get('supertrend_dir', 0) == 1:
            score += 10; reasons.append('Supertrend bullish')
    else:
        if L['close'] < L['ema_200']:
            score += 15; reasons.append('Below EMA 200 — macro downtrend intact')
        else:
            warnings.append('Above EMA 200 — counter-trend short')
        if L.get('ema_aligned_bear', False):
            score += 15; reasons.append('Full EMA stack bearish (9<20<50<100<200)')
        if L.get('ichi_below_cloud', False):
            score += 10; reasons.append('Price below Ichimoku cloud — strong downtrend')
        if L.get('supertrend_dir', 0) == -1:
            score += 10; reasons.append('Supertrend bearish')
    return score


def _score_momentum(L, P, PP, direction, reasons, warnings) -> int:
    score = 0
    rsi = L['rsi']
    if direction == 'LONG':
        if rsi < 30:    score += 20; reasons.append(f'RSI deeply oversold at {rsi:.1f}')
        elif rsi < 40:  score += 12; reasons.append(f'RSI oversold at {rsi:.1f}')
        elif rsi < 50:  score += 6;  reasons.append(f'RSI below midline at {rsi:.1f}')
        if L.get('rsi_slope', 0) > 2:
            score += 8; reasons.append(f'RSI momentum turning up (slope +{L["rsi_slope"]:.1f})')
        if L.get('stoch_cross_up', False) and L['stoch_k'] < 30:
            score += 12; reasons.append('Stoch RSI cross up from oversold')
        elif L['stoch_k'] < 20:
            score += 8; reasons.append(f'Stoch RSI deeply oversold at {L["stoch_k"]:.1f}')
        if L.get('williams_r', 0) < -80:
            score += 6; reasons.append(f'Williams %R oversold at {L["williams_r"]:.1f}')
        if L.get('cci', 0) < -100:
            score += 6; reasons.append(f'CCI oversold at {L["cci"]:.0f}')
    else:
        if rsi > 70:    score += 20; reasons.append(f'RSI deeply overbought at {rsi:.1f}')
        elif rsi > 60:  score += 12; reasons.append(f'RSI overbought at {rsi:.1f}')
        elif rsi > 50:  score += 6;  reasons.append(f'RSI above midline at {rsi:.1f}')
        if L.get('rsi_slope', 0) < -2:
            score += 8; reasons.append(f'RSI momentum turning down (slope {L["rsi_slope"]:.1f})')
        if L.get('stoch_cross_down', False) and L['stoch_k'] > 70:
            score += 12; reasons.append('Stoch RSI cross down from overbought')
        elif L['stoch_k'] > 80:
            score += 8; reasons.append(f'Stoch RSI deeply overbought at {L["stoch_k"]:.1f}')
        if L.get('williams_r', 0) > -20:
            score += 6; reasons.append(f'Williams %R overbought at {L["williams_r"]:.1f}')
        if L.get('cci', 0) > 100:
            score += 6; reasons.append(f'CCI overbought at {L["cci"]:.0f}')
    return score


def _score_macd(L, P, direction, reasons) -> int:
    score = 0
    if direction == 'LONG':
        if L.get('macd_cross_up', False):
            score += 15; reasons.append('Fresh MACD bullish cross')
        elif L['macd'] > L['macd_signal']:
            score += 8;  reasons.append('MACD above signal line')
        if L['macd_hist'] > 0 and L['macd_hist'] > P['macd_hist']:
            score += 7; reasons.append('MACD histogram expanding bullish')
    else:
        if L.get('macd_cross_down', False):
            score += 15; reasons.append('Fresh MACD bearish cross')
        elif L['macd'] < L['macd_signal']:
            score += 8;  reasons.append('MACD below signal line')
        if L['macd_hist'] < 0 and L['macd_hist'] < P['macd_hist']:
            score += 7; reasons.append('MACD histogram expanding bearish')
    return score


def _score_volume(L, direction, reasons, warnings) -> int:
    score = 0
    rvol = L.get('rvol', 1.0)
    if rvol > 3.0:    score += 20; reasons.append(f'Extreme volume surge {rvol:.1f}× — major player activity')
    elif rvol > 2.0:  score += 15; reasons.append(f'Strong volume {rvol:.1f}× average — institutional presence')
    elif rvol > 1.3:  score += 8;  reasons.append(f'Above-average volume {rvol:.1f}×')
    elif rvol < 0.5:  warnings.append(f'Very low volume {rvol:.1f}× — low conviction')
    obv_slope = L.get('obv_slope', 0)
    if direction == 'LONG' and L.get('obv', 0) > L.get('obv_ema', 0) and obv_slope > 0:
        score += 10; reasons.append('OBV trending up — smart money accumulating')
    elif direction == 'SHORT' and L.get('obv', 0) < L.get('obv_ema', 0) and obv_slope < 0:
        score += 10; reasons.append('OBV trending down — smart money distributing')
    mfi = L.get('mfi', 50)
    if direction == 'LONG' and mfi < 25:
        score += 8; reasons.append(f'MFI oversold at {mfi:.0f} — money flowing in')
    elif direction == 'SHORT' and mfi > 75:
        score += 8; reasons.append(f'MFI overbought at {mfi:.0f} — money flowing out')
    return score


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — BB Keltner Squeeze Bounce
# ══════════════════════════════════════════════════════════════════════════════

def strategy_bb_squeeze_bounce(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    long_setup  = P['close'] < P['bb_lower'] and L['close'] > L['bb_lower']
    short_setup = P['close'] > P['bb_upper'] and L['close'] < L['bb_upper']
    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    score += 15; reasons.append('Price re-entered Bollinger Band — reversal candle confirmed')

    # Squeeze bonus
    if L.get('bb_squeeze', False):
        score += 15; reasons.append('BB inside Keltner — squeeze breakout imminent'); tags.append('squeeze')
    avg_width = df['bb_width'].rolling(50).mean().iloc[-1]
    if not np.isnan(avg_width) and L['bb_width'] < avg_width * 0.6:
        score += 8; reasons.append('Bollinger width at 3-month low — compressed volatility')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, _PP(df), direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    atr   = L['atr']
    entry = L['close']
    sl    = (entry - atr * 2.2) if direction == 'LONG' else (entry + atr * 2.2)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.0, 3.5, 6.0))

    return Signal(
        strategy='BB/Keltner Squeeze Bounce', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — Volume Momentum Breakout
# ══════════════════════════════════════════════════════════════════════════════

def strategy_volume_breakout(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    long_setup  = L['close'] > L['bb_upper'] and L['rvol'] > 1.5
    short_setup = L['close'] < L['bb_lower'] and L['rvol'] > 1.5
    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('breakout')

    rvol = L['rvol']
    if rvol > 3.0:    score += 25; reasons.append(f'Explosive {rvol:.1f}× volume — major breakout')
    elif rvol > 2.0:  score += 18; reasons.append(f'Strong {rvol:.1f}× volume — institutional breakout')
    elif rvol > 1.5:  score += 10; reasons.append(f'Volume {rvol:.1f}× above average')

    # Consecutive closes outside band = stronger
    consec = 0
    for i in range(1, min(4, len(df))):
        row = df.iloc[-i]
        if direction == 'LONG' and row['close'] > row['bb_upper']: consec += 1
        elif direction == 'SHORT' and row['close'] < row['bb_lower']: consec += 1
        else: break
    if consec >= 2:
        score += 10; reasons.append(f'{consec} consecutive closes outside band — sustained momentum')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    rsi = L['rsi']
    if direction == 'LONG' and 50 < rsi < 75:
        score += 10; reasons.append(f'RSI in momentum zone {rsi:.1f} — not yet overbought')
    elif direction == 'SHORT' and 25 < rsi < 50:
        score += 10; reasons.append(f'RSI in momentum zone {rsi:.1f} — not yet oversold')
    elif direction == 'LONG' and rsi > 80:
        warnings.append(f'RSI {rsi:.1f} — extremely overbought, chasing risk')

    adx = L.get('adx', 20)
    if adx > 30:
        score += 8; reasons.append(f'ADX {adx:.1f} — strong trending conditions')

    atr   = L['atr']
    entry = L['close']
    sl    = (entry - atr * 1.5) if direction == 'LONG' else (entry + atr * 1.5)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.5, 4.5, 8.0))

    return Signal(
        strategy='Volume Momentum Breakout', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3 — EMA Trend Pullback
# ══════════════════════════════════════════════════════════════════════════════

def strategy_ema_pullback(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    # Touch EMA 50 with bounce candle in established trend
    tol = 0.006
    long_setup  = (L['close'] > L['ema_200'] and L['ema_20'] > L['ema_50'] and
                   abs(L['low'] - L['ema_50']) / L['ema_50'] < tol and
                   L['close'] > L['open'])
    short_setup = (L['close'] < L['ema_200'] and L['ema_20'] < L['ema_50'] and
                   abs(L['high'] - L['ema_50']) / L['ema_50'] < tol and
                   L['close'] < L['open'])

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('pullback')
    score += 20; reasons.append('Candle wicked EMA 50 and reversed — precision pullback entry')

    # Trend quality
    adx = L.get('adx', 20)
    if adx > 35:    score += 20; reasons.append(f'Very strong trend, ADX {adx:.1f}')
    elif adx > 25:  score += 12; reasons.append(f'Healthy trend, ADX {adx:.1f}')
    else:           warnings.append(f'Weak ADX {adx:.1f} — trend may be fading')

    # Ichimoku confirmation
    if direction == 'LONG' and L.get('ichi_above_cloud', False) and L.get('ichi_bull_cloud', False):
        score += 15; reasons.append('Price above bullish Ichimoku cloud — ideal pullback')
    elif direction == 'SHORT' and L.get('ichi_below_cloud', False):
        score += 15; reasons.append('Price below bearish Ichimoku cloud — ideal pullback')

    # Volume drying up on pullback = healthy
    rvol = L.get('rvol', 1.0)
    if rvol < 0.7:    score += 15; reasons.append(f'Low-volume pullback {rvol:.2f}× — sellers absent')
    elif rvol < 1.0:  score += 8;  reasons.append(f'Below-average pullback volume {rvol:.2f}×')
    elif rvol > 1.5:  warnings.append(f'High volume on pullback {rvol:.1f}× — monitor for breakdown')

    score += _score_momentum(L, P, _PP(df), direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)

    # Supertrend still aligned
    if direction == 'LONG' and L.get('supertrend_dir', 0) == 1:
        score += 8; reasons.append('Supertrend still bullish through pullback')
    elif direction == 'SHORT' and L.get('supertrend_dir', 0) == -1:
        score += 8; reasons.append('Supertrend still bearish through pullback')

    atr   = L['atr']
    entry = L['close']
    sl_base = L['ema_50']
    sl    = (sl_base - atr * 1.0) if direction == 'LONG' else (sl_base + atr * 1.0)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.0, 4.0, 7.0))

    return Signal(
        strategy='EMA Trend Pullback', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4 — RSI Divergence
# ══════════════════════════════════════════════════════════════════════════════

def strategy_rsi_divergence(df: pd.DataFrame) -> 'Signal | None':
    if len(df) < 35:
        return None
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    win = df.tail(35).copy()
    plows  = win['low'].nsmallest(2).index.tolist()
    phighs = win['high'].nlargest(2).index.tolist()

    bull_div = bear_div = False
    if len(plows) == 2:
        i1, i2 = sorted(plows)
        if win.loc[i2,'low'] < win.loc[i1,'low'] and win.loc[i2,'rsi'] > win.loc[i1,'rsi'] + 3:
            bull_div = True
    if len(phighs) == 2:
        i1, i2 = sorted(phighs)
        if win.loc[i2,'high'] > win.loc[i1,'high'] and win.loc[i2,'rsi'] < win.loc[i1,'rsi'] - 3:
            bear_div = True

    if not bull_div and not bear_div:
        return None

    direction = 'LONG' if bull_div else 'SHORT'
    tags.append('divergence')
    rsi_gap = abs(win.loc[sorted(plows if bull_div else phighs)[1],'rsi'] -
                  win.loc[sorted(plows if bull_div else phighs)[0],'rsi'])
    if rsi_gap > 10: score += 35; reasons.append(f'Strong RSI divergence (gap {rsi_gap:.1f} pts) — major reversal signal')
    elif rsi_gap > 5: score += 25; reasons.append(f'Clear RSI divergence (gap {rsi_gap:.1f} pts)')
    else: score += 15; reasons.append('Mild RSI divergence detected')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, _PP(df), direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    # Multi-oscillator confirmation
    extra = 0
    if direction == 'LONG':
        if L.get('stoch_k', 50) < 25:   extra += 1
        if L.get('williams_r', -50) < -75: extra += 1
        if L.get('cci', 0) < -150:       extra += 1
    else:
        if L.get('stoch_k', 50) > 75:   extra += 1
        if L.get('williams_r', -50) > -25: extra += 1
        if L.get('cci', 0) > 150:        extra += 1
    if extra >= 2:
        score += 12; reasons.append(f'{extra}/3 oscillators confirm reversal zone')

    atr   = L['atr']
    entry = L['close']
    sl    = (entry - atr * 2.5) if direction == 'LONG' else (entry + atr * 2.5)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.5, 4.0, 6.5))

    return Signal(
        strategy='RSI Divergence', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5 — Ichimoku TK Cross
# ══════════════════════════════════════════════════════════════════════════════

def strategy_ichimoku_tk(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    # Tenkan/Kijun cross
    tk_bull = bool(L.get('ichi_tk_cross_up', False))
    tk_bear = ((L['ichi_tenkan'] < L['ichi_kijun']) and
               (P['ichi_tenkan'] >= P['ichi_kijun']))

    if not tk_bull and not tk_bear:
        return None

    direction = 'LONG' if tk_bull else 'SHORT'
    tags.append('ichimoku')
    score += 20; reasons.append(f'Ichimoku TK {"bullish" if tk_bull else "bearish"} cross — classic signal')

    # Cross location relative to cloud
    if direction == 'LONG':
        if L.get('ichi_above_cloud', False):
            score += 25; reasons.append('TK cross above cloud — Strong (Type 1)')
        elif not L.get('ichi_below_cloud', False):
            score += 12; reasons.append('TK cross inside cloud — Neutral (Type 2)')
        else:
            score += 5;  warnings.append('TK cross below cloud — Weak (Type 3)')
    else:
        if L.get('ichi_below_cloud', False):
            score += 25; reasons.append('TK cross below cloud — Strong (Type 1)')
        elif not L.get('ichi_above_cloud', False):
            score += 12; reasons.append('TK cross inside cloud — Neutral (Type 2)')
        else:
            score += 5;  warnings.append('TK cross above cloud — Weak (Type 3)')

    # Cloud color
    if direction == 'LONG' and L.get('ichi_bull_cloud', False):
        score += 15; reasons.append('Green cloud ahead — bullish kumo')
    elif direction == 'SHORT' and not L.get('ichi_bull_cloud', True):
        score += 15; reasons.append('Red cloud ahead — bearish kumo')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_volume(L, direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)

    adx = L.get('adx', 20)
    if adx > 25: score += 10; reasons.append(f'ADX {adx:.1f} confirms trend strength')

    atr   = L['atr']
    entry = L['close']
    kijun = L['ichi_kijun']
    sl    = (kijun - atr * 0.8) if direction == 'LONG' else (kijun + atr * 0.8)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.0, 3.5, 6.0))

    return Signal(
        strategy='Ichimoku TK Cross', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6 — Supertrend Flip
# ══════════════════════════════════════════════════════════════════════════════

def strategy_supertrend_flip(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    if not L.get('supertrend_flip', False):
        return None

    direction = 'LONG' if L.get('supertrend_dir', 0) == 1 else 'SHORT'
    tags.append('supertrend')
    score += 25; reasons.append(f'Supertrend flipped {"bullish" if direction=="LONG" else "bearish"} — trend reversal signal')

    # Confirm with EMA
    if direction == 'LONG' and L['close'] > L['ema_50']:
        score += 15; reasons.append('Price above EMA 50 after flip — confirmed')
    elif direction == 'SHORT' and L['close'] < L['ema_50']:
        score += 15; reasons.append('Price below EMA 50 after flip — confirmed')
    else:
        warnings.append('Price still below EMA 50 after bullish flip — wait for follow-through')

    score += _score_trend(L, direction, reasons, warnings)
    score += _score_momentum(L, P, _PP(df), direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)

    # ADX rising = trend starting
    adx = L.get('adx', 20)
    padx = P.get('adx', 20)
    if adx > padx and adx > 20:
        score += 10; reasons.append(f'ADX rising to {adx:.1f} — trend building momentum')

    atr   = L['atr']
    entry = L['close']
    st_line = L.get('supertrend', entry - atr * 2)
    sl    = (st_line - atr * 0.5) if direction == 'LONG' else (st_line + atr * 0.5)
    tp1, tp2, tp3 = _levels(entry, sl, atr, direction, (2.0, 4.0, 7.0))

    return Signal(
        strategy='Supertrend Flip', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 7 — VWAP + Volume Reversion
# ══════════════════════════════════════════════════════════════════════════════

def strategy_vwap_reversion(df: pd.DataFrame) -> 'Signal | None':
    L, P = _L(df), _P(df)
    reasons, warnings, tags = [], [], []
    score = 0

    vwap = L.get('vwap', np.nan)
    if np.isnan(vwap):
        return None

    dev_pct = (L['close'] - vwap) / vwap * 100
    atr_pct = L.get('atr_pct', 1.0)

    # Price extended from VWAP
    long_setup  = dev_pct < -atr_pct * 1.5 and L['rsi'] < 40
    short_setup = dev_pct >  atr_pct * 1.5 and L['rsi'] > 60

    if not long_setup and not short_setup:
        return None

    direction = 'LONG' if long_setup else 'SHORT'
    tags.append('vwap')

    ext = abs(dev_pct)
    if ext > atr_pct * 3:   score += 25; reasons.append(f'Extreme {ext:.2f}% deviation from VWAP — snap-back probable')
    elif ext > atr_pct * 2: score += 15; reasons.append(f'Significant {ext:.2f}% VWAP deviation — mean reversion setup')
    else:                    score += 8;  reasons.append(f'VWAP deviation {ext:.2f}% — reversion entry')

    score += _score_momentum(L, P, _PP(df), direction, reasons, warnings)
    score += _score_macd(L, P, direction, reasons)
    score += _score_volume(L, direction, reasons, warnings)
    score += _score_trend(L, direction, reasons, warnings)

    # Multi-oscillator pile-on
    extra = 0
    if direction == 'LONG':
        if L.get('stoch_k', 50) < 20: extra += 1; reasons.append('Stoch RSI oversold reinforces VWAP long')
        if L.get('mfi', 50) < 25:     extra += 1; reasons.append('MFI oversold — money flow turning')
        if L.get('cci', 0) < -100:    extra += 1; reasons.append('CCI oversold — exhaustion signal')
    else:
        if L.get('stoch_k', 50) > 80: extra += 1; reasons.append('Stoch RSI overbought reinforces VWAP short')
        if L.get('mfi', 50) > 75:     extra += 1; reasons.append('MFI overbought — money flow reversing')
        if L.get('cci', 0) > 100:     extra += 1; reasons.append('CCI overbought — exhaustion signal')
    if extra >= 2: score += 15

    atr   = L['atr']
    entry = L['close']
    sl    = (entry - atr * 2.0) if direction == 'LONG' else (entry + atr * 2.0)
    tp_target = vwap
    tp1   = tp_target
    tp2   = entry + (tp_target - entry) * 1.5  if direction == 'LONG' else entry - (entry - tp_target) * 1.5
    tp3   = entry + atr * 5 * (1 if direction == 'LONG' else -1)

    return Signal(
        strategy='VWAP Reversion', direction=direction,
        confidence=min(score, 100), entry=entry, sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        rr=_rr(entry,sl,tp1), atr=atr,
        regime=str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else '?',
        reasons=reasons, warnings=warnings, tags=tags,
    )


# ══════════════════════════════════════════════════════════════════════════════
# REGIME-AWARE SCANNER
# ══════════════════════════════════════════════════════════════════════════════

# Which strategies thrive in which regimes
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
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


def scan(df: pd.DataFrame, min_confidence: int = 40) -> list[Signal]:
    regime = str(df['market_regime'].iloc[-1]) if 'market_regime' in df.columns else 'TRENDING'
    weights = REGIME_WEIGHTS.get(regime, {})
    signals: list[Signal] = []

    for fn in STRATEGIES:
        try:
            sig = fn(df)
            if sig is None:
                continue
            # Apply regime weight
            weight = weights.get(sig.strategy, 1.0)
            sig.confidence = int(min(100, sig.confidence * weight))
            if sig.confidence >= min_confidence:
                signals.append(sig)
        except Exception:
            pass

    return sorted(signals, key=lambda s: s.confidence, reverse=True)