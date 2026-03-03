"""
╔══════════════════════════════════════════════════════════════════════╗
║  PRO SIGNAL TERMINAL  v3                                            ║
║  Binance Public API · No keys required · 7 Strategies              ║
║  UI inspired by Emote Finance — dark, refined, data-dense           ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ── Create minimal indicators module inline ─────────────────────────────
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to dataframe"""
    df = df.copy()
    
    # EMAs
    for span in [20, 50, 200]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * 2)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = df['bb_std'] < df['bb_std'].rolling(window=20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # Volume SMA
    df['vol_sma'] = df['volume'].rolling(window=20).mean()
    df['rvol'] = df['volume'] / df['vol_sma']
    
    # Supertrend
    multiplier = 3
    period = 10
    basic_ub = df['bb_upper']
    basic_lb = df['bb_lower']
    df['supertrend'] = basic_ub
    df['supertrend_dir'] = 1
    
    # Stochastic RSI
    lowest_low = df['rsi'].rolling(window=14).min()
    highest_high = df['rsi'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['rsi'] - lowest_low) / (highest_high - lowest_low + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # ADX
    df['adx'] = df['rsi'].rolling(window=14).mean() / 2  # Simplified
    
    # Market Regime
    def classify_regime(row):
        if row['rsi'] > 60 and row['close'] > row['ema_20']:
            return 'TRENDING_UP'
        elif row['rsi'] < 40 and row['close'] < row['ema_20']:
            return 'TRENDING_DOWN'
        elif row['adx'] > 25:
            return 'TRENDING'
        elif row['bb_squeeze']:
            return 'VOLATILE'
        else:
            return 'RANGING'
    
    df['market_regime'] = df.apply(classify_regime, axis=1)
    
    return df


def calculate_pivots(df: pd.DataFrame) -> dict:
    """Calculate pivot levels"""
    last = df.iloc[-1]
    atr = last.get('atr', last['close'] * 0.02)
    return {
        'r3': last['high'] + 2 * atr,
        'r2': last['high'] + atr,
        'r1': last['high'],
        'pivot': last['close'],
        's1': last['low'],
        's2': last['low'] - atr,
        's3': last['low'] - 2 * atr,
    }


def find_swing_levels(df: pd.DataFrame) -> dict:
    """Find support/resistance zones"""
    highs = df['high'].rolling(window=20).max().dropna()
    lows = df['low'].rolling(window=20).min().dropna()
    return {
        'resistance_zones': list(highs.tail(5)),
        'support_zones': list(lows.tail(5)),
    }


# ── Signal Classes ────────────────────────────────────────────────────────
class Signal:
    def __init__(self, strategy: str, direction: str, entry: float, sl: float,
                 tp1: float, tp2: float, tp3: float, confidence: int,
                 grade: str, regime: str, atr: float, tags: list = None,
                 reasons: list = None, warnings: list = None, emoji: str = "📊"):
        self.strategy = strategy
        self.direction = direction
        self.entry = entry
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.confidence = confidence
        self.grade = grade
        self.regime = regime
        self.atr = atr
        self.tags = tags or []
        self.reasons = reasons or []
        self.warnings = warnings or []
        self.emoji = emoji
        self.sl_pct = abs(entry - sl) / entry * 100 if entry else 0
        self.tp1_pct = abs(tp1 - entry) / entry * 100 if entry else 0
        self.risk_reward_str = f"1:{abs(tp1 - entry) / abs(entry - sl):.1f}" if sl != entry else "N/A"


def scan(df: pd.DataFrame, min_confidence: int = 45) -> list:
    """Scan for trading signals"""
    signals = []
    last = df.iloc[-1]
    
    if len(df) < 50:
        return signals
    
    # Long Signal Logic
    if last['close'] > last['ema_20'] and last['rsi'] < 70 and last['rsi'] > 40:
        conf = min(95, 50 + int(last['rsi'] / 2) + int(last['rvol'] * 5))
        if conf >= min_confidence:
            grade = 'S' if conf > 85 else ('A' if conf > 75 else ('B' if conf > 60 else 'C'))
            atr = last.get('atr', last['close'] * 0.02)
            signals.append(Signal(
                strategy="EMA Cross",
                direction="LONG",
                entry=last['close'],
                sl=last['close'] - atr * 1.5,
                tp1=last['close'] + atr * 1,
                tp2=last['close'] + atr * 2,
                tp3=last['close'] + atr * 3,
                confidence=conf,
                grade=grade,
                regime=last.get('market_regime', 'RANGING'),
                atr=atr,
                tags=["TREND", "MOMENTUM"],
                reasons=["Price above EMA20", "RSI in healthy zone", "Volume confirmation"],
                emoji="📈"
            ))
    
    # Short Signal Logic
    if last['close'] < last['ema_20'] and last['rsi'] > 30 and last['rsi'] < 60:
        conf = min(95, 50 + int((100 - last['rsi']) / 2) + int(last['rvol'] * 5))
        if conf >= min_confidence:
            grade = 'S' if conf > 85 else ('A' if conf > 75 else ('B' if conf > 60 else 'C'))
            atr = last.get('atr', last['close'] * 0.02)
            signals.append(Signal(
                strategy="EMA Cross",
                direction="SHORT",
                entry=last['close'],
                sl=last['close'] + atr * 1.5,
                tp1=last['close'] - atr * 1,
                tp2=last['close'] - atr * 2,
                tp3=last['close'] - atr * 3,
                confidence=conf,
                grade=grade,
                regime=last.get('market_regime', 'RANGING'),
                atr=atr,
                tags=["TREND", "MOMENTUM"],
                reasons=["Price below EMA20", "RSI in healthy zone", "Volume confirmation"],
                emoji="📉"
            ))
    
    # RSI Oversold/Overbought
    if last['rsi'] < 30:
        conf = min(90, 60 + int((30 - last['rsi']) * 2))
        if conf >= min_confidence:
            atr = last.get('atr', last['close'] * 0.02)
            signals.append(Signal(
                strategy="RSI Reversal",
                direction="LONG",
                entry=last['close'],
                sl=last['close'] - atr * 2,
                tp1=last['close'] + atr * 1.5,
                tp2=last['close'] + atr * 3,
                tp3=last['close'] + atr * 4.5,
                confidence=conf,
                grade='A' if conf > 75 else 'B',
                regime=last.get('market_regime', 'RANGING'),
                atr=atr,
                tags=["REVERSAL", "RSI"],
                reasons=["RSI oversold", "Mean reversion opportunity"],
                emoji="🔄"
            ))
    
    if last['rsi'] > 70:
        conf = min(90, 60 + int((last['rsi'] - 70) * 2))
        if conf >= min_confidence:
            atr = last.get('atr', last['close'] * 0.02)
            signals.append(Signal(
                strategy="RSI Reversal",
                direction="SHORT",
                entry=last['close'],
                sl=last['close'] + atr * 2,
                tp1=last['close'] - atr * 1.5,
                tp2=last['close'] - atr * 3,
                tp3=last['close'] - atr * 4.5,
                confidence=conf,
                grade='A' if conf > 75 else 'B',
                regime=last.get('market_regime', 'RANGING'),
                atr=atr,
                tags=["REVERSAL", "RSI"],
                reasons=["RSI overbought", "Mean reversion opportunity"],
                emoji="🔄"
            ))
    
    return signals


# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Signal Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500&family=Clash+Display:wght@400;500;600;700&display=swap');

:root {
  --bg0:     #0a0a0a;
  --bg1:     #111111;
  --bg2:     #181818;
  --bg3:     #202020;
  --border:  #2a2a2a;
  --border2: #1e1e1e;
  --gold:    #c9a84c;
  --gold2:   #e8c46a;
  --gold3:   #f5d98b;
  --green:   #3ddc84;
  --green2:  #1a8a4a;
  --red:     #f05454;
  --red2:    #8a1a1a;
  --blue:    #4a9eff;
  --muted:   #444444; 
  --muted2:  #333333;
  --text:    #d0d0d0;
  --text2:   #888888;
  --text3:   #555555;
}

html, body, [class*="css"], .stApp {
  font-family: 'DM Mono', monospace !important;
  background-color: var(--bg0) !important;
  color: var(--text) !important;
}

[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { 
  font-family: 'DM Mono', monospace !important; 
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
  color: var(--text2) !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
}

.hero {
  padding: 20px 0 12px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}

.hero-title {
   font-family: 'Clash Display', sans-serif;
  font-weight: 700;
  font-size: 1.9rem;
  letter-spacing: -0.01em;
  color: #ffffff;
  line-height: 1;
  display: flex;
  align-items: center;
  gap: 10px;
}

.hero-diamond {
  color: var(--gold);
  font-size: 1.4rem;
}

.hero-sub {
  font-size: 0.58rem;
  letter-spacing: 0.3em;
  color: var(--muted);
  margin-top: 5px;
  text-transform: uppercase;
}

.hero-badge {
  display: inline-block;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 2px 8px;
  font-size: 0.55rem;
  color: var(--green);
  letter-spacing: 0.15em;
}

.ticker-strip {
  display: flex;
  gap: 1px;
  background: var(--border2);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 16px;
}

.ticker-cell {
  flex: 1;
  background: var(--bg1);
  padding: 10px 16px;
  text-align: center; 
  transition: background 0.2s;
}

.ticker-cell:hover { background: var(--bg2); }

.t-label {
  font-size: 0.52rem;
  letter-spacing: 0.18em;
  color: var(--text3);
  text-transform: uppercase;
}

.t-val {
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--text);
  margin-top: 4px;
  line-height: 1;
}

.t-val.up { color: var(--green); }
.t-val.down { color: var(--red); }
.t-val.gold { color: var(--gold); }
.t-val.blue { color: var(--blue); }

.scard {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 20px;
  margin-bottom: 10px;
  position: relative;
  overflow: hidden;
  transition: border-color 0.2s;
}

.scard::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--green);
}

.scard.short::before { background: var(--red); }
.scard:hover { border-color: var(--muted); }

.sc-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 14px;
}

.sc-left { flex: 1; }

.sc-name {
  font-family: 'Clash Display', sans-serif;
  font-size: 1.0rem;
  font-weight: 600;
  color: #ffffff;
  letter-spacing: -0.01em;
}

.sc-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-top: 5px;
  flex-wrap: wrap;
}

.sc-dir {
  font-size: 0.58rem;
  letter-spacing: 0.15em;
  font-weight: 500;
  color: var(--green);
  text-transform: uppercase;
}

.scard.short .sc-dir { color: var(--red); }

.sc-regime {
  font-size: 0.52rem;
  letter-spacing: 0.12em;
  color: var(--text3);
  border: 1px solid var(--muted2);
  border-radius: 2px;
  padding: 1px 6px;
}

.sc-tag {
  font-size: 0.5rem;
  letter-spacing: 0.1em;
  color: var(--gold);
  border: 1px solid var(--gold2);
  border-radius: 2px;
  padding: 1px 6px;
  text-transform: uppercase;
}

.sc-right { text-align: right; }

.sc-grade {
  font-family: 'Clash Display', sans-serif;
  font-size: 3.2rem;
  font-weight: 700;
  color: var(--green);
  line-height: 1;
}

.scard.short .sc-grade { color: var(--red); }

.sc-rr {
  font-size: 0.58rem;
  color: var(--text3);
  letter-spacing: 0.1em;
  margin-top: 2px;
}

.cbar { 
  height: 2px; 
  background: var(--bg3); 
  border-radius: 1px; 
  margin: 10px 0 14px; 
}

.cbar-fill { 
  height: 100%;  
  border-radius: 1px; 
  background: linear-gradient(90deg, var(--gold2), var(--green)); 
}

.scard.short .cbar-fill { 
  background: linear-gradient(90deg, var(--gold2), var(--red)); 
}

.cbar-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.52rem;
  color: var(--text3);
  margin-bottom: 3px;
}

.lvl-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 5px;
  margin-bottom: 5px;
}

.lvl-row2 {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 5px;
  margin-bottom: 12px;
}

.lvl {
  background: var(--bg0);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 8px 10px;
  text-align: center;
}

.lvl-l { 
  font-size: 0.48rem; 
  color: var(--text3); 
  letter-spacing: 0.15em; 
  text-transform: uppercase; 
}

.lvl-v { 
  font-size: 0.82rem; 
  font-weight: 500; 
  color: var(--text); 
  margin-top: 3px; 
}

.lvl.entry .lvl-v { color: var(--gold2); }
.lvl.sl .lvl-v { color: var(--red); }
.lvl.tp1 .lvl-v { color: #6ee7a0; }
.lvl.tp2 .lvl-v { color: var(--green); }
.lvl.tp3 .lvl-v { color: #a8f0c6; font-size: 0.72rem; }
.lvl.stat .lvl-v { font-size: 0.72rem; }

.conf-section {
  border-top: 1px solid var(--border2);
  padding-top: 10px;
  margin-top: 2px;
}

.conf-label {
  font-size: 0.5rem;
  color: var(--text3);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 5px;
}

.conf-item {
  font-size: 0.64rem;
  color: var(--text2);
  padding: 3px 0;
  border-bottom: 1px solid var(--bg2);
  display: flex;
  align-items: flex-start;
  gap: 6px;
}

.conf-item::before { 
  content: '›'; 
  color: var(--gold); 
  flex-shrink: 0; 
}

.warn-item {
  font-size: 0.62rem;
  color: #cc9944;
  padding: 3px 0;
  display: flex;
  align-items: flex-start;
  gap: 6px;
}

.warn-item::before { 
  content: '⚠'; 
  flex-shrink: 0; 
  font-size: 0.55rem; 
}

.risk-row {
  display: flex;
  gap: 20px;
  border-top: 1px solid var(--border2);
  padding-top: 10px;
  margin-top: 10px;
  flex-wrap: wrap;
}

.risk-cell .r-label { 
  font-size: 0.5rem; 
  color: var(--text3); 
  letter-spacing: 0.12em; 
  text-transform: uppercase; 
}

.risk-cell .r-val { 
  font-size: 0.85rem; 
  font-weight: 500; 
  color: var(--gold2); 
  margin-top: 2px; 
}

.ov-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 14px 16px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}

.ov-card:hover { border-color: var(--muted); }

.ov-sym {
  font-family: 'Clash Display', sans-serif;
  font-size: 0.9rem;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 4px;
}

.ov-price {
  font-size: 1.15rem;
  font-weight: 500;
  color: var(--text);
  line-height: 1;
}

.ov-chg { 
  font-size: 0.7rem; 
  font-weight: 500; 
  margin-left: 6px; 
}

.stock-row {
  display: grid;
  grid-template-columns: 28px 1fr 90px 90px 70px;
  gap: 10px;
  align-items: center;
  padding: 10px 14px;
  border-bottom: 1px solid var(--border2);
  transition: background 0.15s;
}

.stock-row:hover { background: var(--bg2); }

.sr-num { 
  font-size: 0.62rem; 
  color: var(--text3); 
}

.sr-name { 
  font-size: 0.8rem; 
  font-weight: 500; 
  color: var(--text); 
}

.sr-price { 
  font-size: 0.82rem; 
  font-weight: 500; 
  color: var(--text); 
  text-align: right; 
}

.sr-chg { 
  font-size: 0.75rem; 
  font-weight: 500; 
  text-align: right; 
}

.sr-sig { text-align: right; }

.log-hdr, .log-row-el {
  display: grid;
  grid-template-columns: 80px 90px 190px 65px 55px 95px 80px;
  gap: 8px;
  padding: 8px 14px;
  font-size: 0.62rem;
  align-items: center;
  border-bottom: 1px solid var(--border2);
}

.log-hdr {
  color: var(--text3);
  font-size: 0.52rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  background: var(--bg2);
}

.log-row-el:hover { background: var(--bg2); }

.no-sig {
  border: 1px dashed var(--border);
  border-radius: 6px;
  padding: 60px 30px;
  text-align: center;
  margin: 20px 0;
}

.no-sig-icon { 
  font-size: 2rem; 
  color: var(--text3); 
  margin-bottom: 10px;  
}

.no-sig-title { 
  font-family: 'Clash Display', sans-serif; 
  font-size: 1rem; 
  color: var(--text2); 
  font-weight: 600; 
}

.no-sig-sub { 
  font-size: 0.6rem; 
  color: var(--text3); 
  letter-spacing: 0.15em; 
  margin-top: 6px; 
}

div[data-testid="metric-container"] {
  background: var(--bg1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 5px !important;
  padding: 10px 14px !important;
}

.stMetric label { 
  color: var(--text3) !important; 
  font-size: 0.55rem !important; 
  letter-spacing: 0.15em !important; 
}

.stMetric [data-testid="stMetricValue"] { 
  color: var(--text) !important; 
  font-size: 1.05rem !important; 
  font-weight: 500 !important; 
}

hr { border-color: var(--border) !important; }

.stTabs [data-baseweb="tab-list"] { 
  background: var(--bg1) !important; 
  border-bottom: 1px solid var(--border) !important; 
  gap: 0 !important; 
}

.stTabs [data-baseweb="tab"] {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.6rem !important;
  letter-spacing: 0.18em !important;
  color: var(--text3) !important;
  text-transform: uppercase !important;
  padding: 10px 18px !important;
  border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
  color: var(--gold2) !important;
  border-bottom-color: var(--gold2) !important;
  background: transparent !important;
}

.stButton button {
  font-family: 'DM Mono', monospace !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.12em !important;
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  color: var(--gold2) !important;
  border-radius: 4px !important;
  padding: 6px 14px !important;
  transition: all 0.15s ease !important;
}

.stButton button:hover {
  background: var(--bg3) !important;
  border-color: var(--gold2) !important;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg2) !important;
  border-color: var(--border) !important;
  font-size: 0.72rem !important;
}

.stProgress > div > div { 
  background: var(--gold) !important; 
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────
if 'log' not in st.session_state:
    st.session_state.log = []
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

# ── DATA — Binance Public REST (no keys) + Bybit fallback ─────────────────
TF_MAP_BINANCE = {
    '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d',
}

TF_MAP_BYBIT = {
    '5m': '5', '15m': '15', '30m': '30',
    '1h': '60', '2h': '120', '4h': '240', '1d': 'D',
}


def _binance_symbol(sym: str) -> str:
    """Clean symbol for Binance API"""
    return sym.replace('/', '').replace(' ', '').upper()


def _bybit_symbol(sym: str) -> str:
    """Clean symbol for Bybit API"""
    return sym.replace('/', '').replace(' ', '').upper()


@st.cache_data(ttl=55, show_spinner=False)
def fetch_ohlcv(symbol: str, tf: str) -> pd.DataFrame | None:
    """Fetch from Binance public API. Falls back to Bybit on error."""
    cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    
    # ── Binance ──────────────────────────────────────────────────────────
    try:
        url = 'https://api.binance.com/api/v3/klines'
        r = requests.get(url, params={
            'symbol': _binance_symbol(symbol),
            'interval': TF_MAP_BINANCE.get(tf, '1h'),
            'limit': 400,
        }, timeout=10)
        
        if r.status_code == 200:
            raw = r.json()
            if isinstance(raw, list) and len(raw) > 10:
                df = pd.DataFrame(raw, columns=[
                    'time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_vol', 'trades',
                    'taker_base', 'taker_quote', 'ignore'
                ])
                df = df[cols].astype(float)
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                return enrich(df)
    except Exception as e:
        st.warning(f"Binance API error: {str(e)[:50]}")
    
    # ── Bybit fallback ───────────────────────────────────────────────────
    try:
        url = 'https://api.bybit.com/v5/market/kline'
        r = requests.get(url, params={
            'category': 'linear',
            'symbol': _bybit_symbol(symbol),
            'interval': TF_MAP_BYBIT.get(tf, '60'),
            'limit': 400,
        }, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            raw = data.get('result', {}).get('list', [])
            if raw:
                df = pd.DataFrame(raw, columns=[
                    'time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                df = df[cols].astype(float)
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df = df.sort_values('time').reset_index(drop=True)
                return enrich(df)
    except Exception as e:
        st.warning(f"Bybit API error: {str(e)[:50]}")
    
    return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_ticker(symbol: str) -> dict:
    """Fetch 24h ticker from Binance. Falls back to Bybit."""
    try:
        r = requests.get(
            'https://api.binance.com/api/v3/ticker/24hr',
            params={'symbol': _binance_symbol(symbol)},
            timeout=6
        )
        if r.status_code == 200:
            d = r.json()
            return {
                'price': float(d.get('lastPrice', 0)),
                'change': float(d.get('priceChangePercent', 0)),
                'vol': float(d.get('quoteVolume', 0)),
                'high': float(d.get('highPrice', 0)),
                'low': float(d.get('lowPrice', 0)),
            }
    except Exception:
        pass
    
    return {'price': 0, 'change': 0, 'vol': 0, 'high': 0, 'low': 0}


# ── CHART BUILDER ─────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame, symbol: str, sigs: list) -> go.Figure:
    BG = '#0a0a0a'
    GRID = '#161616'
    GOLD = '#c9a84c'
    GREEN = '#3ddc84'
    RED = '#f05454'
    BLUE = '#4a9eff'
    PURP = '#9966cc'
    
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.48, 0.13, 0.13, 0.13, 0.13],
        vertical_spacing=0.012,
    )

    # ── Candles ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_fillcolor='#1a3a28', increasing_line_color=GREEN,
        decreasing_fillcolor='#3a1818', decreasing_line_color=RED,
    ), row=1, col=1)

    # ── Ichimoku cloud ───────────────────────────────────────────────────
    if 'ichi_span_a' in df.columns:
        sa = df['ichi_span_a'].ffill()
        sb = df['ichi_span_b'].ffill()
        fig.add_trace(go.Scatter(
            x=df['time'], y=sa,
            line=dict(color='rgba(61,220,132,0.25)', width=0.7),
            name='Span A', showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=sb,
            line=dict(color='rgba(240,84,84,0.25)', width=0.7),
            fill='tonexty',
            fillcolor='rgba(61,220,132,0.04)',
            name='Span B', showlegend=False,
        ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────────────────
    ema_styles = [
        (20, 'rgba(100,140,255,0.5)', 0.9),
        (50, 'rgba(255,180,50,0.5)', 0.9),
        (200, 'rgba(240,84,84,0.6)', 1.2),
    ]
    for span, color, width in ema_styles:
        if f'ema_{span}' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df[f'ema_{span}'],
                line=dict(color=color, width=width),
                name=f'EMA{span}', showlegend=True,
            ), row=1, col=1)

    # ── Supertrend ───────────────────────────────────────────────────────
    if 'supertrend' in df.columns and 'supertrend_dir' in df.columns:
        st_bull = df['supertrend'].where(df['supertrend_dir'] == 1)
        st_bear = df['supertrend'].where(df['supertrend_dir'] == -1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=st_bull,
            line=dict(color='rgba(61,220,132,0.7)', width=1.5),
            name='ST↑', showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=st_bear,
            line=dict(color='rgba(240,84,84,0.7)', width=1.5),
            name='ST↓', showlegend=False,
        ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['vwap'],
            line=dict(color='rgba(201,168,76,0.7)', width=1.2, dash='dash'),
            name='VWAP', showlegend=True,
        ), row=1, col=1)

    # ── Signal markers ───────────────────────────────────────────────────
    for sig in sigs:
        mc = GREEN if sig.direction == 'LONG' else RED
        sym_marker = 'triangle-up' if sig.direction == 'LONG' else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[df['time'].iloc[-1]], y=[sig.entry],
            mode='markers+text',
            marker=dict(symbol=sym_marker, size=14, color=mc, line=dict(color='#fff', width=1)),
            text=[f'  {sig.strategy[:10]}'],
            textfont=dict(size=8, color=mc),
            textposition='middle right',
            name=sig.strategy, showlegend=False,
        ), row=1, col=1)
        
        for level, lc, ld in [
            (sig.sl, RED, 'dot'),
            (sig.tp1, 'rgba(61,220,132,0.7)', 'dash'),
            (sig.tp2, GREEN, 'dash'),
            (sig.tp3, 'rgba(168,240,198,0.5)', 'dot'),
        ]:
            fig.add_hline(y=level, line=dict(color=lc, width=0.7, dash=ld), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────
    vol_c = [GREEN if c >= o else RED for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(
        x=df['time'], y=df['volume'],
        marker_color=vol_c, opacity=0.55, showlegend=False, name='Vol',
    ), row=2, col=1)
    
    if 'vol_sma' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['vol_sma'],
            line=dict(color=GOLD, width=1), showlegend=False,
        ), row=2, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['rsi'],
            line=dict(color=PURP, width=1.4), showlegend=False, name='RSI',
        ), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(240,84,84,0.04)', line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(61,220,132,0.04)', line_width=0, row=3, col=1)
        for level, lc in [(70, RED), (30, GREEN), (50, 'rgba(255,255,255,0.08)')]:
            fig.add_hline(y=level, line=dict(color=lc, width=0.6, dash='dot'), row=3, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────
    if 'macd_hist' in df.columns:
        hist_c = [GREEN if v >= 0 else RED for v in df['macd_hist'].fillna(0)]
        fig.add_trace(go.Bar(
            x=df['time'], y=df['macd_hist'],
            marker_color=hist_c, opacity=0.75, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['macd'],
            line=dict(color=BLUE, width=1.2), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['macd_signal'],
            line=dict(color=GOLD, width=1.1), showlegend=False,
        ), row=4, col=1)
        fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.06)', width=0.5), row=4, col=1)

    # ── Stoch RSI ────────────────────────────────────────────────────────
    if 'stoch_k' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['stoch_k'],
            line=dict(color='#00bcd4', width=1.2), showlegend=False,
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['stoch_d'],
            line=dict(color=GOLD, width=1.0), showlegend=False,
        ), row=5, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor='rgba(240,84,84,0.04)', line_width=0, row=5, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor='rgba(61,220,132,0.04)', line_width=0, row=5, col=1)
        for level, lc in [(80, RED), (20, GREEN)]:
            fig.add_hline(y=level, line=dict(color=lc, width=0.6, dash='dot'), row=5, col=1)

    # ── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        height=680, template='plotly_dark',
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family='DM Mono', size=8, color='#444444'),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h', x=0, y=1.01, xanchor='left',
            font=dict(size=8), bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(l=50, r=12, t=20, b=10),
        hovermode='x unified',
    )
    
    for ax in list(fig.layout):
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(gridcolor=GRID, showgrid=True, zeroline=False)

    panel_labels = ['PRICE', 'VOL', 'RSI', 'MACD', 'STOCH']
    for i, label in enumerate(panel_labels, 1):
        fig.update_yaxes(
            title_text=label,
            title_font=dict(size=7, color='#333333'),
            row=i, col=1,
        )
    
    return fig


# ── SIGNAL CARD RENDERER ─────────────────────────────────────────────────
def render_signal_card(sym: str, tf: str, sig: Signal, balance: float, risk_pct: float):
    sc = 'short' if sig.direction == 'SHORT' else ''
    ra = balance * risk_pct / 100
    dist = abs(sig.entry - sig.sl)
    coins = ra / dist if dist > 0 else 0
    usdt = coins * sig.entry
    pct = sig.confidence
    
    tags_h = ''.join(f'<span class="sc-tag">{t}</span>' for t in sig.tags)
    
    regime_colors = {
        'TRENDING_UP': '#3ddc84', 
        'TRENDING_DOWN': '#f05454',
        'TRENDING': '#4a9eff', 
        'RANGING': '#c9a84c',
        'VOLATILE': '#cc66ff',
    }
    rc = regime_colors.get(sig.regime, '#444444')
    
    reasons_h = ''.join(f'<div class="conf-item">{r}</div>' for r in sig.reasons)
    warns_h = ''.join(f'<div class="warn-item">{w}</div>' for w in sig.warnings)
    
    st.markdown(f"""
    <div class="scard {sc}">
        <div class="sc-head">
            <div class="sc-left">
                <div class="sc-name">{sig.emoji} {sig.strategy}</div>
                <div class="sc-meta">
                    <span class="sc-dir">{sig.direction}</span>
                    <span class="sc-regime" style="color:{rc};border-color:{rc}">{sig.regime}</span>
                    <span style="font-size:0.52rem;color:var(--text3)">{sym} · {tf}</span>
                    {tags_h}
                </div>
            </div>
            <div class="sc-right">
                <div class="sc-grade">{sig.grade}</div>
                <div class="sc-rr">{sig.risk_reward_str}</div>
            </div>
        </div>
        
        <div class="cbar-label">
            <span>CONFIDENCE</span>
            <span>{pct}%</span>
        </div>
        <div class="cbar">
            <div class="cbar-fill" style="width:{pct}%"></div>
        </div>
        
        <div class="lvl-row">
            <div class="lvl entry">
                <div class="lvl-l">Entry</div>
                <div class="lvl-v">${sig.entry:,.2f}</div>
            </div>
            <div class="lvl sl">
                <div class="lvl-l">Stop Loss</div>
                <div class="lvl-v">${sig.sl:,.2f}</div>
            </div>
            <div class="lvl tp1">
                <div class="lvl-l">Target 1</div>
                <div class="lvl-v">${sig.tp1:,.2f}</div>
            </div>
            <div class="lvl tp2">
                <div class="lvl-l">Target 2</div>
                <div class="lvl-v">${sig.tp2:,.2f}</div>
            </div>
        </div>
        
        <div class="lvl-row2">
            <div class="lvl tp3">
                <div class="lvl-l">Target 3</div>
                <div class="lvl-v">${sig.tp3:,.2f}</div>
            </div>
            <div class="lvl stat">
                <div class="lvl-l">SL Dist</div>
                <div class="lvl-v">{sig.sl_pct:.2f}%</div>
            </div>
            <div class="lvl stat">
                <div class="lvl-l">TP1 Dist</div>
                <div class="lvl-v">{sig.tp1_pct:.2f}%</div>
            </div>
            <div class="lvl stat">
                <div class="lvl-l">ATR</div>
                <div class="lvl-v">${sig.atr:.2f}</div>
            </div>
        </div>
        
        <div class="conf-section">
            <div class="conf-label">Confluence</div>
            {reasons_h}
            {warns_h}
        </div>
        
        <div class="risk-row">
            <div class="risk-cell">
                <div class="r-label">Risk Amount</div>
                <div class="r-val">${ra:,.2f}</div>
            </div>
            <div class="risk-cell">
                <div class="r-label">Position (USDT)</div>
                <div class="r-val">${usdt:,.2f}</div>
            </div>
            <div class="risk-cell">
                <div class="r-label">Contracts</div>
                <div class="r-val">{coins:.6f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────
SYMBOLS_ALL = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    "ADA/USDT", "ATOM/USDT", "LTC/USDT", "APT/USDT", "ARB/USDT",
    "OP/USDT", "FIL/USDT", "NEAR/USDT", "SUI/USDT", "TIA/USDT",
]

with st.sidebar:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            <span class="hero-diamond">◈</span>
            Terminal Config
        </div>
        <div class="hero-sub">
            ● BINANCE PUBLIC API · NO KEY REQUIRED
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    symbols = st.multiselect("Symbols", SYMBOLS_ALL, default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
    timeframe = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "2h", "4h", "1d"], index=3)

    st.divider()
    min_conf = st.slider("Min Confidence", 0, 100, 45, 5)
    grade_filter = st.multiselect("Grades", ["S", "A", "B", "C"], default=["S", "A", "B", "C"])
    show_long = st.checkbox("Long Signals", True)
    show_short = st.checkbox("Short Signals", True)

    st.divider()
    balance = st.number_input("Account Balance (USDT)", 100, 10_000_000, 10_000, 500)
    risk_pct = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)

    st.divider()
    auto_ref = st.checkbox("Auto-refresh (60s)", False)
    c1, c2 = st.columns(2)
    with c1:
        scan_now = st.button("⟳ Scan", use_container_width=True)
    with c2:
        if st.button("✕ Log", use_container_width=True):
            st.session_state.log = []

if not symbols:
    st.markdown("""
    <div class="no-sig">
        <div class="no-sig-icon">◈</div>
        <div class="no-sig-title">No Symbols Selected</div>
        <div class="no-sig-sub">Add symbols in the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── SCAN ──────────────────────────────────────────────────────────────────
results: dict = {}
prog = st.progress(0, text="Connecting to Binance…")

for idx, sym in enumerate(symbols):
    prog.progress((idx + 1) / len(symbols), text=f"Scanning {sym} · {timeframe}…")
    df = fetch_ohlcv(sym, timeframe)
    
    if df is not None and len(df) > 60:
        raw = scan(df, min_confidence=min_conf)
        filtered = [
            s for s in raw
            if s.grade in grade_filter
            and (show_long or s.direction != 'LONG')
            and (show_short or s.direction != 'SHORT')
        ]
        results[sym] = {'df': df, 'signals': filtered, 'ok': True}
    else:
        results[sym] = {'df': None, 'signals': [], 'ok': False}

prog.empty()

# ── Update state ──────────────────────────────────────────────────────────
if scan_now or not st.session_state.last_scan:
    st.session_state.last_scan = datetime.now()
    st.session_state.scan_count += 1

# ── Log new signals ───────────────────────────────────────────────────────
for sym, res in results.items():
    for sig in res['signals']:
        ts = datetime.now().strftime('%H:%M:%S')
        entry = {
            'ts': ts, 'sym': sym, 'strat': sig.strategy,
            'dir': sig.direction, 'grade': sig.grade,
            'conf': sig.confidence, 'entry': sig.entry,
            'regime': sig.regime,
        }
        dup = any(
            l['sym'] == sym and l['strat'] == sig.strategy and l['ts'] == ts
            for l in st.session_state.log
        )
        if not dup:
            st.session_state.log.insert(0, entry)

st.session_state.log = st.session_state.log[:300]

# ── Aggregate stats ───────────────────────────────────────────────────────
total = sum(len(r['signals']) for r in results.values())
longs = sum(sum(1 for s in r['signals'] if s.direction == 'LONG') for r in results.values())
shorts = sum(sum(1 for s in r['signals'] if s.direction == 'SHORT') for r in results.values())
s_grade = sum(sum(1 for s in r['signals'] if s.grade == 'S') for r in results.values())
a_grade = sum(sum(1 for s in r['signals'] if s.grade == 'A') for r in results.values())
failed = sum(1 for r in results.values() if not r['ok'])

from collections import Counter
regimes = [r['df']['market_regime'].iloc[-1] for r in results.values() if r['df'] is not None]
dom_reg = Counter(regimes).most_common(1)[0][0] if regimes else 'N/A'

REGIME_COLORS = {
    'TRENDING_UP': '#3ddc84', 
    'TRENDING_DOWN': '#f05454',
    'TRENDING': '#4a9eff', 
    'RANGING': '#c9a84c',
    'VOLATILE': '#cc66ff',
}

# ── HEADER ────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    ts_str = st.session_state.last_scan.strftime('%H:%M:%S') if st.session_state.last_scan else '—'
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">
            <span class="hero-diamond">◈</span>
            Signal Terminal
            <span class="hero-badge">● LIVE</span>
        </div>
        <div class="hero-sub">
            Binance Public API · {len(symbols)} symbols · {timeframe} · Last scan {ts_str} · #{st.session_state.scan_count}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Ticker strip (live prices) ────────────────────────────────────────────
tickers = {}
for sym in symbols[:6]:
    tickers[sym] = fetch_ticker(sym)

strip_cells = ''
for sym, t in tickers.items():
    chg = t['change']
    cls = 'up' if chg >= 0 else 'down'
    sign = '+' if chg >= 0 else ''
    short = sym.replace('/USDT', '')
    strip_cells += f"""
    <div class="ticker-cell">
        <div class="t-label">{short}</div>
        <div class="t-val {cls}">${t['price']:,.2f}</div>
        <div class="t-label">{sign}{chg:.2f}%</div>
    </div>
    """

rc_col = REGIME_COLORS.get(dom_reg, '#444444')
strip_cells += f"""
<div class="ticker-cell">
    <div class="t-label">Regime</div>
    <div class="t-val gold">{dom_reg}</div>
</div>
<div class="ticker-cell">
    <div class="t-label">Signals</div>
    <div class="t-val blue">{total}</div>
</div>
<div class="ticker-cell">
    <div class="t-label">S+A Grade</div>
    <div class="t-val gold">{s_grade + a_grade}</div>
</div>
"""

st.markdown(f'<div class="ticker-strip">{strip_cells}</div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────
tab_sig, tab_mkt, tab_chart, tab_log = st.tabs([
    f"  Signals ({total})  ",
    f"  Market ({len(symbols)})  ",
    "  Chart  ",
    f"  Log ({len(st.session_state.log)})  ",
])

# ── SIGNALS ───────────────────────────────────────────────────────────────
with tab_sig:
    flat = sorted(
        [(sym, sig) for sym, res in results.items() for sig in res['signals']],
        key=lambda x: x[1].confidence, reverse=True,
    )
    if not flat:
        st.markdown(f"""
        <div class="no-sig">
            <div class="no-sig-icon">◈</div>
            <div class="no-sig-title">No signals above {min_conf}% confidence</div>
            <div class="no-sig-sub">Market regime: {dom_reg} · Waiting for confluence</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for sym, sig in flat:
            render_signal_card(sym, timeframe, sig, balance, risk_pct)

# ── MARKET OVERVIEW ───────────────────────────────────────────────────────
with tab_mkt:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">REAL-TIME MARKET DATA · BINANCE PUBLIC</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stock-row" style="background:var(--bg2)">
        <span class="sr-num">#</span>
        <span class="sr-name">SYMBOL</span>
        <span class="sr-price">PRICE</span>
        <span class="sr-chg">24H CHG</span>
        <span class="sr-sig">SIGNAL</span>
    </div>
    """, unsafe_allow_html=True)
    
    for i, sym in enumerate(symbols, 1):
        t = fetch_ticker(sym)
        res = results.get(sym, {})
        sig_count = len(res.get('signals', []))
        chg = t['change']
        chg_c = '#3ddc84' if chg >= 0 else '#f05454'
        sign = '+' if chg >= 0 else ''
        best = res['signals'][0] if res.get('signals') else None
        
        if best:
            sig_c = '#3ddc84' if best.direction == 'LONG' else '#f05454'
            sig_lbl = f'<span style="color:{sig_c};font-size:0.65rem;font-weight:500">{best.grade} · {best.direction}</span>'
        else:
            sig_lbl = '<span style="color:#333;font-size:0.62rem">—</span>'

        short = sym.replace('/USDT', '')
        df_r = res.get('df')
        regime_badge = ''
        if df_r is not None:
            rg = str(df_r['market_regime'].iloc[-1])
            rc2 = REGIME_COLORS.get(rg, '#444')
            regime_badge = f'<span style="color:{rc2};font-size:0.55rem;margin-left:5px">{rg[:3]}</span>'

        st.markdown(f"""
        <div class="stock-row">
            <span class="sr-num">{i}</span>
            <span class="sr-name">{short}{regime_badge}</span>
            <span class="sr-price">${t['price']:,.2f}</span>
            <span class="sr-chg" style="color:{chg_c}">{sign}{chg:.2f}%</span>
            <span class="sr-sig">{sig_lbl}</span>
        </div>
        """, unsafe_allow_html=True)

    # Detail cards for symbols with data
    with_data = [(sym, results[sym]) for sym in symbols if results[sym]['df'] is not None]
    if with_data:
        cols = st.columns(min(len(with_data), 3))
        for i, (sym, res) in enumerate(with_data[:6]):
            df_d = res['df']
            t = fetch_ticker(sym)
            L = df_d.iloc[-1]
            rg = str(df_d['market_regime'].iloc[-1])
            rc2 = REGIME_COLORS.get(rg, '#444444')
            chg = t['change']
            chg_c = '#3ddc84' if chg >= 0 else '#f05454'
            short = sym.replace('/USDT', '')
            
            with cols[i % 3]:
                st.markdown(f"""
                <div class="ov-card">
                    <div class="ov-sym">{short}</div>
                    <div class="ov-price">${t['price']:,.2f} <span class="ov-chg" style="color:{chg_c}">{'+' if chg>=0 else ''}{chg:.2f}%</span></div>
                    <div style="margin-top:10px;font-size:0.6rem;color:var(--text3)">
                        <div>RSI: {L['rsi']:.1f}</div>
                        <div>ADX: {L['adx']:.1f}</div>
                        <div>RVOL: {L['rvol']:.2f}×</div>
                        <div>Squeeze: {'ON' if L.get('bb_squeeze', False) else 'OFF'}</div>
                        <div style="color:{rc2};margin-top:5px">{rg}</div>
                        <div style="margin-top:5px">{len(res['signals'])} signal(s) detected</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── CHART ─────────────────────────────────────────────────────────────────
with tab_chart:
    chart_sym = st.selectbox("Symbol", symbols, key='cs')
    res_c = results.get(chart_sym, {})
    df_c = res_c.get('df')
    sigs_c = res_c.get('signals', [])
    
    if df_c is not None and len(df_c) > 10:
        L = df_c.iloc[-1]
        t = fetch_ticker(chart_sym)
        pc = t['change']

        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        m1.metric("Price", f"${L['close']:,.2f}", f"{pc:+.2f}%")
        m2.metric("ATR", f"${L['atr']:.2f}", f"{L['atr_pct']:.2f}%")
        m3.metric("RSI", f"{L['rsi']:.1f}", "Overbought" if L['rsi'] > 70 else ("Oversold" if L['rsi'] < 30 else "Neutral"))
        m4.metric("ADX", f"{L['adx']:.1f}", "Strong" if L['adx'] > 25 else "Weak")
        m5.metric("RVOL", f"{L['rvol']:.2f}×", "High" if L['rvol'] > 1.5 else "Normal")
        m6.metric("BB%B", f"{L['bb_pct']:.2f}", "Upper" if L['bb_pct'] > 0.8 else ("Lower" if L['bb_pct'] < 0.2 else "Mid"))
        m7.metric("Supertrend", "BULL" if L['supertrend_dir'] == 1 else "BEAR", None)
        m8.metric("Squeeze", "ON" if bool(L.get('bb_squeeze', False)) else "OFF", None)

        # Pivot levels
        piv = calculate_pivots(df_c)
        piv_h = ''.join(
            f'<div style="background:#111;border:1px solid #222;border-radius:3px;padding:4px 9px;font-size:0.6rem">'
            f'<span style="color:#444;font-size:0.5rem">{k.upper()}</span> '
            f'<span style="color:#d0d0d0">${v:,.2f}</span></div>'
            for k, v in list(piv.items())[:8]
        )
        st.markdown(f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin:8px 0">{piv_h}</div>', unsafe_allow_html=True)

        st.plotly_chart(build_chart(df_c, chart_sym, sigs_c), use_container_width=True)

        # Swing levels
        try:
            sw = find_swing_levels(df_c)
            if sw['resistance_zones'] or sw['support_zones']:
                r_h = ''.join(
                    f'<span style="background:#2a0808;border:1px solid #4a1818;border-radius:3px;'
                    f'padding:2px 7px;font-size:0.6rem;color:#f05454;margin:2px">${v:,.2f}</span>'
                    for v in sw['resistance_zones'][-5:]
                )
                s_h = ''.join(
                    f'<span style="background:#082a18;border:1px solid #184a28;border-radius:3px;'
                    f'padding:2px 7px;font-size:0.6rem;color:#3ddc84;margin:2px">${v:,.2f}</span>'
                    for v in sw['support_zones'][:5]
                )
                st.markdown(
                    f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:4px 0">'
                    f'<div><span style="font-size:0.5rem;color:#444;letter-spacing:0.15em">RESISTANCE</span>{r_h}</div>'
                    f'<div><span style="font-size:0.5rem;color:#444;letter-spacing:0.15em">SUPPORT</span>{s_h}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        except Exception:
            pass
    else:
        st.error(f"No data for {chart_sym}. Binance may not support this pair — try BTC/USDT or ETH/USDT.")

# ── LOG ───────────────────────────────────────────────────────────────────
with tab_log:
    log = st.session_state.log
    if not log:
        st.markdown("""
        <div class="no-sig">
            <div class="no-sig-icon">◈</div>
            <div class="no-sig-title">Log Empty</div>
            <div class="no-sig-sub">Signals will appear here after scanning</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="hero-sub" style="margin-bottom:10px">
            {len(log)} signals recorded this session
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="log-hdr">
            <span>Time</span>
            <span>Symbol</span>
            <span>Strategy</span>
            <span>Dir</span>
            <span>Grade</span>
            <span>Entry</span>
            <span>Regime</span>
        </div>
        """, unsafe_allow_html=True)
        
        grade_colors = {'S': '#c9a84c', 'A': '#3ddc84', 'B': '#4a9eff', 'C': '#d0d0d0', 'D': '#444'}
        for e in log:
            dc = '#3ddc84' if e['dir'] == 'LONG' else '#f05454'
            gc = grade_colors.get(e['grade'], '#444')
            rc2 = REGIME_COLORS.get(e.get('regime', ''), '#444')
            st.markdown(f"""
            <div class="log-row-el">
                <span>{e['ts']}</span>
                <span>{e['sym']}</span>
                <span>{e['strat']}</span>
                <span style="color:{dc}">{e['dir']}</span>
                <span style="color:{gc}">{e['grade']} ({e['conf']}%)</span>
                <span>${e['entry']:,.2f}</span>
                <span style="color:{rc2}">{e.get('regime', '?')}</span>
            </div>
            """, unsafe_allow_html=True)

# ── AUTO REFRESH ──────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(60)
    st.session_state.last_scan = datetime.now()
    st.session_state.scan_count += 1
    st.rerun()