"""
╔══════════════════════════════════════════════════════════════════╗
║   PRO SIGNAL TERMINAL  v2  ·  Bybit  ·  Signal Generator Only  ║
╚══════════════════════════════════════════════════════════════════╝
7 strategies · Regime-aware scoring · Ichimoku · Supertrend · ADX
Full indicator suite · Multi-symbol scanner · Signal history log
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

from indicators import enrich, calculate_pivots, find_swing_levels
from signals import scan, Signal

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SIGNAL TERMINAL", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600;700&family=Bebas+Neue&display=swap');

:root {
  --bg:       #060a0f;
  --bg1:      #0a1018;
  --bg2:      #0d1520;
  --border:   #1a2535;
  --border2:  #0f1c28;
  --green:    #00e676;
  --green2:   #00c853;
  --red:      #ff1744;
  --red2:     #d50000;
  --amber:    #ffab00;
  --blue:     #2979ff;
  --muted:    #2a3f55;
  --text:     #b0c8e0;
  --text2:    #6a8aa0;
}

html, body, [class*="css"] {
  font-family: 'IBM Plex Mono', monospace !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Header ─────────────────────── */
.t-head {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2.8rem;
  letter-spacing: 0.12em;
  color: var(--green);
  text-shadow: 0 0 40px #00e67633, 0 0 80px #00e67611;
  line-height: 1;
}
.t-sub {
  font-size: 0.6rem;
  letter-spacing: 0.4em;
  color: var(--muted);
  margin-top: 2px;
  text-transform: uppercase;
}

/* ── Regime badge ───────────────── */
.regime-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 2px;
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 0.2em;
  border: 1px solid;
}

/* ── Market strip ───────────────── */
.mstrip {
  display: flex;
  gap: 1px;
  margin: 12px 0;
  background: var(--border2);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.mstrip-cell {
  flex: 1;
  padding: 10px 14px;
  background: var(--bg1);
  text-align: center;
}
.mstrip-label { font-size: 0.5rem; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }
.mstrip-val   { font-size: 1rem; font-weight: 600; color: var(--text); margin-top: 3px; line-height: 1; }
.g { color: var(--green)  !important; }
.r { color: var(--red)    !important; }
.a { color: var(--amber)  !important; }
.b { color: var(--blue)   !important; }

/* ── Signal cards ───────────────── */
.sig {
  border: 1px solid var(--border);
  border-left: 3px solid var(--green);
  background: linear-gradient(135deg, var(--bg2) 0%, var(--bg1) 100%);
  border-radius: 3px;
  padding: 16px 20px;
  margin-bottom: 10px;
  position: relative;
}
.sig.short { border-left-color: var(--red); }
.sig-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}
.sig-name {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.08em;
  color: var(--green);
}
.sig.short .sig-name { color: var(--red); }
.sig-meta { font-size: 0.55rem; color: var(--muted); letter-spacing: 0.15em; margin-top: 2px; }
.sig-grade {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 3rem;
  line-height: 1;
  color: var(--green);
}
.sig.short .sig-grade { color: var(--red); }
.sig-rr { font-size: 0.6rem; color: var(--muted); text-align: right; margin-top: 2px; }

/* Confidence bar */
.cbar-wrap { height: 3px; background: var(--bg); border-radius: 2px; margin: 8px 0; overflow: hidden; }
.cbar-fill  { height: 100%; border-radius: 2px; background: var(--green); transition: width 0.5s; }
.sig.short .cbar-fill { background: var(--red); }

/* Level grid */
.lvl-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin: 10px 0;
}
.lvl-box {
  background: var(--bg);
  border: 1px solid var(--border2);
  border-radius: 2px;
  padding: 7px 9px;
  text-align: center;
}
.lvl-label { font-size: 0.5rem; color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase; }
.lvl-val   { font-size: 0.8rem; font-weight: 600; color: var(--text); margin-top: 2px; }
.sl  .lvl-val { color: var(--red);    }
.tp1 .lvl-val { color: #69f0ae;       }
.tp2 .lvl-val { color: var(--green);  }
.tp3 .lvl-val { color: #b9f6ca;       }

/* Reasons */
.reasons { margin: 10px 0; border-top: 1px solid var(--border2); padding-top: 8px; }
.reason-label { font-size: 0.5rem; color: var(--muted); letter-spacing: 0.2em; margin-bottom: 5px; }
.reason { font-size: 0.65rem; color: #5a8a6a; padding: 2px 0; }
.reason::before { content: "▸ "; color: #1a4a2a; }
.warn   { font-size: 0.65rem; color: #aa6622; padding: 2px 0; }
.warn::before   { content: "⚠ "; }

/* Tags */
.tag {
  display: inline-block;
  padding: 1px 6px;
  border: 1px solid var(--muted);
  border-radius: 2px;
  font-size: 0.5rem;
  color: var(--muted);
  letter-spacing: 0.1em;
  margin-right: 4px;
  text-transform: uppercase;
}

/* Risk footer */
.risk-footer {
  display: flex;
  gap: 20px;
  border-top: 1px solid var(--border2);
  padding-top: 10px;
  margin-top: 10px;
}
.rf-item .rf-label { font-size: 0.5rem; color: var(--muted); letter-spacing: 0.1em; }
.rf-item .rf-val   { font-size: 0.85rem; font-weight: 600; color: var(--amber); margin-top: 2px; }

/* ── No signal ──────────────────── */
.no-sig {
  border: 1px dashed var(--border);
  border-radius: 3px;
  padding: 50px 30px;
  text-align: center;
  color: var(--muted);
}
.no-sig-icon  { font-size: 2rem; margin-bottom: 8px; }
.no-sig-title { font-size: 0.9rem; letter-spacing: 0.2em; font-weight: 600; }
.no-sig-sub   { font-size: 0.6rem; color: #1a3050; margin-top: 6px; letter-spacing: 0.1em; }

/* ── Log ────────────────────────── */
.log-hdr, .log-row {
  display: grid;
  grid-template-columns: 90px 90px 200px 65px 55px 90px 70px;
  gap: 8px;
  padding: 7px 12px;
  font-size: 0.65rem;
  align-items: center;
  border-bottom: 1px solid var(--border2);
}
.log-hdr { color: var(--muted); font-size: 0.55rem; letter-spacing: 0.12em; text-transform: uppercase; }
.log-row:hover { background: var(--bg2); }

/* ── Metric cards ───────────────── */
div[data-testid="metric-container"] {
  background: var(--bg1) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  padding: 10px 14px !important;
}
.stMetric label  { color: var(--muted) !important; font-size: 0.55rem !important; letter-spacing: 0.15em !important; }
.stMetric [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 1rem !important; font-weight: 600 !important; }
.stMetric [data-testid="stMetricDelta"] { font-size: 0.6rem !important; }

/* ── Misc ───────────────────────── */
hr { border-color: var(--border) !important; }
.stTabs [data-baseweb="tab"] {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.6rem !important;
  letter-spacing: 0.2em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
.stTabs [aria-selected="true"] { color: var(--green) !important; border-bottom-color: var(--green) !important; }
.stButton button {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.65rem !important;
  letter-spacing: 0.15em !important;
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  color: var(--green) !important;
  border-radius: 2px !important;
  transition: all 0.15s ease !important;
}
.stButton button:hover {
  background: #0a2a18 !important;
  border-color: var(--green) !important;
  box-shadow: 0 0 12px #00e67622 !important;
}
.stSelectbox > div, .stMultiSelect > div {
  background: var(--bg1) !important;
  border-color: var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'scan_count' not in st.session_state:
    st.session_state.scan_count = 0

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.3rem;letter-spacing:0.2em;color:var(--green,#00e676);margin-bottom:14px">⚡ TERMINAL CONFIG</div>', unsafe_allow_html=True)

    def _s(key, fallback=""):
        try: return st.secrets.get("bybit", {}).get(key, fallback)
        except: return fallback

    api_key = st.text_input("Bybit API Key", type="password", value=_s("api_key"), placeholder="Optional — public data works without keys")
    secret  = st.text_input("Bybit Secret",  type="password", value=_s("secret"),  placeholder="Read-only key sufficient")
    st.caption("⚠ Read-only keys only. No trading permissions used.")

    st.divider()

    SYMBOLS_AVAIL = ["BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT",
                     "DOGE/USDT","AVAX/USDT","LINK/USDT","DOT/USDT","MATIC/USDT",
                     "ADA/USDT","ATOM/USDT","LTC/USDT","APT/USDT","ARB/USDT"]

    symbols   = st.multiselect("Symbols", SYMBOLS_AVAIL, default=["BTC/USDT","ETH/USDT","SOL/USDT"])
    timeframe = st.selectbox("Timeframe", ["5m","15m","30m","1h","2h","4h","1d"], index=3)

    st.divider()
    min_conf   = st.slider("Min Confidence", 0, 100, 50, 5)
    show_long  = st.checkbox("Long signals",  True)
    show_short = st.checkbox("Short signals", True)
    grade_filter = st.multiselect("Grades", ["S","A","B","C","D"], default=["S","A","B","C"])

    st.divider()
    balance  = st.number_input("Account Balance (USDT)", 100, 10_000_000, 10_000, 500)
    risk_pct = st.slider("Risk per Trade %", 0.1, 5.0, 1.0, 0.1)

    st.divider()
    auto_ref = st.checkbox("Auto-refresh (60s)", False)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        scan_now = st.button("⟳ SCAN", use_container_width=True)
    with col_s2:
        clear_log = st.button("✕ LOG", use_container_width=True)

    if clear_log:
        st.session_state.signal_log = []

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=55, show_spinner=False)
def fetch(symbol: str, tf: str, ak: str = "", sk: str = "") -> pd.DataFrame | None:
    try:
        cfg = {'enableRateLimit': True, 'options': {'defaultType': 'linear', 'adjustForTimeDifference': True}}
        if ak and sk:
            cfg['apiKey'] = ak
            cfg['secret'] = sk
        ex   = ccxt.bybit(cfg)
        raw  = ex.fetch_ohlcv(symbol, timeframe=tf, limit=350)
        df   = pd.DataFrame(raw, columns=['time','open','high','low','close','volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return enrich(df)
    except Exception:
        return None


def make_chart(df: pd.DataFrame, symbol: str, sigs: list[Signal]) -> go.Figure:
    PLOT_BG   = '#060a0f'
    GRID_COL  = '#0d1828'
    GREEN     = '#00e676'
    RED       = '#ff1744'
    AMBER     = '#ffab00'
    BLUE      = '#2979ff'
    PURPLE    = '#aa44cc'

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        row_heights=[0.46, 0.13, 0.13, 0.13, 0.15],
        vertical_spacing=0.015,
        subplot_titles=['', 'VOLUME', 'RSI', 'MACD', 'STOCH RSI'],
    )

    # ── Candles ──
    fig.add_trace(go.Candlestick(
        x=df['time'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_fillcolor='#1a3a2a', increasing_line_color=GREEN,
        decreasing_fillcolor='#3a1020', decreasing_line_color=RED,
    ), row=1, col=1)

    # ── Ichimoku cloud ──
    span_a = df['ichi_span_a']
    span_b = df['ichi_span_b']
    cloud_green = span_a > span_b
    for i in range(len(df)-1):
        color = 'rgba(0,200,100,0.06)' if cloud_green.iloc[i] else 'rgba(255,50,80,0.06)'
    fig.add_trace(go.Scatter(x=df['time'], y=span_a, line=dict(color='rgba(0,200,100,0.3)', width=0.8), name='Span A', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=span_b, line=dict(color='rgba(255,50,80,0.3)', width=0.8), name='Span B', fill='tonexty', fillcolor='rgba(0,200,100,0.04)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ichi_tenkan'], line=dict(color='#4488ff', width=1), name='Tenkan', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ichi_kijun'],  line=dict(color='#ff8844', width=1), name='Kijun',  showlegend=False), row=1, col=1)

    # ── Bollinger Bands ──
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_upper'], line=dict(color='rgba(41,121,255,0.4)', width=1, dash='dot'), name='BB', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['bb_lower'], line=dict(color='rgba(41,121,255,0.4)', width=1, dash='dot'), name='BB', fill='tonexty', fillcolor='rgba(41,121,255,0.03)', showlegend=False), row=1, col=1)

    # ── EMAs ──
    ema_cfg = [(20,'#2244aa',1.0),(50,'#996622',1.0),(200,'#cc4422',1.5)]
    for span, col, w in ema_cfg:
        fig.add_trace(go.Scatter(x=df['time'], y=df[f'ema_{span}'], line=dict(color=col, width=w), name=f'EMA{span}', showlegend=True), row=1, col=1)

    # ── Supertrend ──
    st_up   = df['supertrend'].where(df['supertrend_dir'] == 1)
    st_down = df['supertrend'].where(df['supertrend_dir'] == -1)
    fig.add_trace(go.Scatter(x=df['time'], y=st_up,   mode='lines', line=dict(color=GREEN, width=1.5), name='ST↑', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=st_down, mode='lines', line=dict(color=RED,   width=1.5), name='ST↓', showlegend=False), row=1, col=1)

    # ── VWAP ──
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], line=dict(color=AMBER, width=1, dash='dash'), name='VWAP', showlegend=True), row=1, col=1)

    # ── Signal markers + levels ──
    for sig in sigs:
        mc   = GREEN if sig.direction == 'LONG' else RED
        sym  = 'triangle-up' if sig.direction == 'LONG' else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[df['time'].iloc[-1]], y=[sig.entry],
            mode='markers+text',
            marker=dict(symbol=sym, size=16, color=mc, line=dict(color='white', width=1)),
            text=[f' {sig.strategy[:12]}'], textfont=dict(size=8, color=mc),
            textposition='middle right', name=sig.strategy, showlegend=False,
        ), row=1, col=1)
        for level, lc in [(sig.sl,'#ff1744'),(sig.tp1,'#69f0ae'),(sig.tp2,GREEN),(sig.tp3,'#b9f6ca')]:
            fig.add_hline(y=level, line=dict(color=lc, width=0.7, dash='dot'), row=1, col=1)

    # ── Volume ──
    vol_colors = [GREEN if c >= o else RED for c,o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df['time'], y=df['volume'], marker_color=vol_colors, opacity=0.6, showlegend=False, name='Vol'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['vol_sma'], line=dict(color=AMBER, width=1), showlegend=False, name='VolSMA'), row=2, col=1)

    # ── RSI ──
    rsi_color = [RED if v > 70 else (GREEN if v < 30 else BLUE) for v in df['rsi'].fillna(50)]
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], line=dict(color=PURPLE, width=1.5), showlegend=False, name='RSI'), row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,23,68,0.05)', line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,230,118,0.05)', line_width=0, row=3, col=1)
    fig.add_hline(y=70, line=dict(color='rgba(255,23,68,0.4)', width=0.7, dash='dot'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='rgba(0,230,118,0.4)', width=0.7, dash='dot'), row=3, col=1)
    fig.add_hline(y=50, line=dict(color='rgba(41,121,255,0.3)', width=0.5), row=3, col=1)

    # ── MACD ──
    hist_c = [GREEN if v >= 0 else RED for v in df['macd_hist'].fillna(0)]
    fig.add_trace(go.Bar(x=df['time'], y=df['macd_hist'], marker_color=hist_c, opacity=0.8, showlegend=False, name='Hist'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd'],        line=dict(color=BLUE,  width=1.2), showlegend=False, name='MACD'),   row=4, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['macd_signal'], line=dict(color=AMBER, width=1.2), showlegend=False, name='Signal'), row=4, col=1)
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.1)', width=0.5), row=4, col=1)

    # ── Stoch RSI ──
    fig.add_trace(go.Scatter(x=df['time'], y=df['stoch_k'], line=dict(color='#00bcd4', width=1.2), showlegend=False, name='%K'), row=5, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['stoch_d'], line=dict(color=AMBER,    width=1.0), showlegend=False, name='%D'), row=5, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor='rgba(255,23,68,0.05)',  line_width=0, row=5, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor='rgba(0,230,118,0.05)',  line_width=0, row=5, col=1)
    fig.add_hline(y=80, line=dict(color='rgba(255,23,68,0.4)',  width=0.6, dash='dot'), row=5, col=1)
    fig.add_hline(y=20, line=dict(color='rgba(0,230,118,0.4)',  width=0.6, dash='dot'), row=5, col=1)

    fig.update_layout(
        height=680, template='plotly_dark',
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(family='IBM Plex Mono', size=8, color='#4a6a80'),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', x=0, y=1.02, font=dict(size=8), bgcolor='rgba(0,0,0,0)', xanchor='left'),
        margin=dict(l=55, r=15, t=25, b=10),
        hovermode='x unified',
    )
    for ax in fig.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(gridcolor=GRID_COL, showgrid=True, zeroline=False)
    for r in range(1, 6):
        fig.update_yaxes(row=r, col=1, gridcolor=GRID_COL)
    return fig


REGIME_COLORS = {
    'TRENDING_UP':   ('#00e676', '#0a2a18'),
    'TRENDING_DOWN': ('#ff1744', '#2a0a10'),
    'TRENDING':      ('#2979ff', '#0a1830'),
    'RANGING':       ('#ffab00', '#2a1e00'),
    'VOLATILE':      ('#aa44cc', '#1a0a2a'),
}


def render_signal(sig: Signal, balance: float, risk: float):
    sc    = 'short' if sig.direction == 'SHORT' else ''
    ra    = balance * risk / 100
    dist  = abs(sig.entry - sig.sl)
    coins = ra / dist if dist > 0 else 0
    usdt  = coins * sig.entry
    conf  = sig.confidence
    bar   = int(conf)
    tags  = ''.join(f'<span class="tag">{t}</span>' for t in sig.tags) if sig.tags else ''
    regime_col = REGIME_COLORS.get(sig.regime, ('#6a8aa0','#0a1018'))
    reasons_html = ''.join(f'<div class="reason">{r}</div>' for r in sig.reasons)
    warns_html   = ''.join(f'<div class="warn">{w}</div>'   for w in sig.warnings)

    st.markdown(f"""
<div class="sig {sc}">
  <div class="sig-head">
    <div>
      <div class="sig-name">{sig.emoji} {sig.strategy}</div>
      <div class="sig-meta">
        {sig.direction} · {conf}% CONFIDENCE ·
        <span class="regime-badge" style="color:{regime_col[0]};border-color:{regime_col[0]};background:{regime_col[1]}">{sig.regime}</span>
      </div>
      <div style="margin-top:5px">{tags}</div>
    </div>
    <div style="text-align:right">
      <div class="sig-grade">{sig.grade}</div>
      <div class="sig-rr">R:R  {sig.risk_reward_str}</div>
    </div>
  </div>

  <div class="cbar-wrap"><div class="cbar-fill" style="width:{bar}%"></div></div>

  <div class="lvl-grid">
    <div class="lvl-box">
      <div class="lvl-label">Entry</div>
      <div class="lvl-val">${sig.entry:,.2f}</div>
    </div>
    <div class="lvl-box sl">
      <div class="lvl-label">Stop Loss</div>
      <div class="lvl-val">${sig.sl:,.2f}</div>
    </div>
    <div class="lvl-box tp1">
      <div class="lvl-label">Target 1</div>
      <div class="lvl-val">${sig.tp1:,.2f}</div>
    </div>
    <div class="lvl-box tp2">
      <div class="lvl-label">Target 2</div>
      <div class="lvl-val">${sig.tp2:,.2f}</div>
    </div>
  </div>
  <div style="display:flex;gap:6px;margin-bottom:8px">
    <div class="lvl-box tp3" style="flex:1;text-align:center">
      <span class="lvl-label">Target 3 </span>
      <span class="lvl-val" style="font-size:0.75rem">${sig.tp3:,.2f}</span>
    </div>
    <div class="lvl-box" style="flex:1;text-align:center">
      <span class="lvl-label">SL Distance </span>
      <span class="lvl-val" style="font-size:0.75rem">{sig.sl_pct:.2f}%</span>
    </div>
    <div class="lvl-box" style="flex:1;text-align:center">
      <span class="lvl-label">TP1 Distance </span>
      <span class="lvl-val" style="font-size:0.75rem">{sig.tp1_pct:.2f}%</span>
    </div>
    <div class="lvl-box" style="flex:1;text-align:center">
      <span class="lvl-label">ATR </span>
      <span class="lvl-val" style="font-size:0.75rem">${sig.atr:.2f}</span>
    </div>
  </div>

  <div class="reasons">
    <div class="reason-label">CONFLUENCE FACTORS</div>
    {reasons_html}
    {warns_html}
  </div>

  <div class="risk-footer">
    <div class="rf-item"><div class="rf-label">Risk Amount</div><div class="rf-val">${ra:,.2f} USDT</div></div>
    <div class="rf-item"><div class="rf-label">Position (USDT)</div><div class="rf-val">${usdt:,.2f}</div></div>
    <div class="rf-item"><div class="rf-label">Contracts</div><div class="rf-val">{coins:.6f}</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown('<div class="t-head">⚡ SIGNAL TERMINAL</div>', unsafe_allow_html=True)
    st.markdown('<div class="t-sub">Bybit · 7 Strategies · Signal Generator · No Auto-Trading</div>', unsafe_allow_html=True)
with h2:
    if st.session_state.last_scan:
        ts = st.session_state.last_scan.strftime('%H:%M:%S')
        st.markdown(f'<div style="text-align:right;font-size:0.55rem;color:var(--muted,#2a3f55);padding-top:10px">LAST SCAN<br><span style="color:#4a6a80;font-size:0.7rem">{ts}</span><br><span style="color:#1a3050">SCAN #{st.session_state.scan_count}</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════════════════
if not symbols:
    st.markdown('<div class="no-sig"><div class="no-sig-icon">—</div><div class="no-sig-title">NO SYMBOLS SELECTED</div><div class="no-sig-sub">Add symbols in the sidebar to begin scanning</div></div>', unsafe_allow_html=True)
    st.stop()

results: dict[str, dict] = {}
prog = st.progress(0, text="Initialising scanner...")
for idx, sym in enumerate(symbols):
    prog.progress((idx+1)/len(symbols), text=f"Scanning {sym} · {timeframe}...")
    df = fetch(sym, timeframe, api_key, secret)
    if df is not None and len(df) > 60:
        raw = scan(df, min_confidence=min_conf)
        filtered = [
            s for s in raw
            if s.grade in grade_filter
            and (show_long  or s.direction != 'LONG')
            and (show_short or s.direction != 'SHORT')
        ]
        results[sym] = {'df': df, 'signals': filtered, 'ok': True}
    else:
        results[sym] = {'df': None, 'signals': [], 'ok': False}
prog.empty()

# Update state
if scan_now or auto_ref:
    st.session_state.last_scan  = datetime.now()
    st.session_state.scan_count += 1
elif st.session_state.last_scan is None:
    st.session_state.last_scan  = datetime.now()
    st.session_state.scan_count = 1

# Log
for sym, res in results.items():
    for sig in res['signals']:
        ts = datetime.now().strftime('%H:%M:%S')
        entry = {'ts': ts, 'sym': sym, 'strat': sig.strategy,
                 'dir': sig.direction, 'grade': sig.grade,
                 'conf': sig.confidence, 'entry': sig.entry, 'regime': sig.regime}
        dup = any(l['sym']==sym and l['strat']==sig.strategy and l['ts']==ts for l in st.session_state.signal_log)
        if not dup:
            st.session_state.signal_log.insert(0, entry)
st.session_state.signal_log = st.session_state.signal_log[:200]

# ── Aggregates ──
total  = sum(len(r['signals']) for r in results.values())
longs  = sum(sum(1 for s in r['signals'] if s.direction=='LONG')  for r in results.values())
shorts = sum(sum(1 for s in r['signals'] if s.direction=='SHORT') for r in results.values())
s_grade= sum(sum(1 for s in r['signals'] if s.grade=='S')         for r in results.values())
a_grade= sum(sum(1 for s in r['signals'] if s.grade=='A')         for r in results.values())
failed = sum(1 for r in results.values() if not r['ok'])

# Regime summary
regimes = [r['df']['market_regime'].iloc[-1] for r in results.values() if r['df'] is not None]
from collections import Counter
regime_counts = Counter(regimes)
dominant_regime = regime_counts.most_common(1)[0][0] if regime_counts else 'N/A'
rc = REGIME_COLORS.get(dominant_regime, ('#6a8aa0','#0a1018'))

# ── Market strip ──
st.markdown(f"""
<div class="mstrip">
  <div class="mstrip-cell"><div class="mstrip-label">Symbols</div><div class="mstrip-val">{len(symbols)}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Signals</div><div class="mstrip-val {'g' if total>0 else ''}">{total}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Long</div><div class="mstrip-val g">{longs}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Short</div><div class="mstrip-val r">{shorts}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">S-Grade</div><div class="mstrip-val a">{s_grade}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">A-Grade</div><div class="mstrip-val b">{a_grade}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Timeframe</div><div class="mstrip-val">{timeframe}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Regime</div><div class="mstrip-val" style="color:{rc[0]};font-size:0.75rem">{dominant_regime}</div></div>
  <div class="mstrip-cell"><div class="mstrip-label">Min Conf</div><div class="mstrip-val">{min_conf}%</div></div>
  {'<div class="mstrip-cell"><div class="mstrip-label">Errors</div><div class="mstrip-val r">' + str(failed) + '</div></div>' if failed else ''}
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_sig, tab_chart, tab_regime, tab_log = st.tabs([
    f"  ⚡ SIGNALS ({total})  ",
    "  📊 CHART  ",
    "  🌐 MARKET REGIME  ",
    f"  📋 LOG ({len(st.session_state.signal_log)})  ",
])

# ── SIGNALS TAB ────────────────────────────────────────────────────────────
with tab_sig:
    flat = sorted(
        [(sym, sig) for sym, res in results.items() for sig in res['signals']],
        key=lambda x: x[1].confidence, reverse=True,
    )
    if not flat:
        st.markdown(f"""
<div class="no-sig">
  <div class="no-sig-icon">—</div>
  <div class="no-sig-title">NO SIGNALS ABOVE {min_conf}% CONFIDENCE</div>
  <div class="no-sig-sub">MARKETS ARE IN {dominant_regime} REGIME · WAITING FOR CONFLUENCE</div>
</div>
""", unsafe_allow_html=True)
    else:
        for sym, sig in flat:
            st.markdown(f'<div style="font-size:0.55rem;letter-spacing:0.25em;color:var(--muted,#2a3f55);margin-bottom:3px;margin-top:8px">{sym} · {timeframe} · {sig.regime}</div>', unsafe_allow_html=True)
            render_signal(sig, balance, risk_pct)

# ── CHART TAB ──────────────────────────────────────────────────────────────
with tab_chart:
    chart_sym = st.selectbox("Symbol", symbols, key='cs')
    res = results.get(chart_sym, {})
    df_c = res.get('df')
    sigs_c = res.get('signals', [])

    if df_c is not None:
        L = df_c.iloc[-1]
        pc = (L['close'] - df_c.iloc[-2]['close']) / df_c.iloc[-2]['close'] * 100

        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        regime_col2 = REGIME_COLORS.get(str(df_c['market_regime'].iloc[-1]), ('#6a8aa0','#0a1018'))
        c1.metric("Price",      f"${L['close']:,.2f}",     f"{pc:+.2f}%")
        c2.metric("ATR",        f"${L['atr']:.2f}",         f"{L['atr_pct']:.2f}%")
        c3.metric("RSI",        f"{L['rsi']:.1f}",          "Obought" if L['rsi']>70 else ("Osold" if L['rsi']<30 else "Neutral"))
        c4.metric("ADX",        f"{L['adx']:.1f}",          "Strong" if L['adx']>25 else "Weak")
        c5.metric("RVOL",       f"{L['rvol']:.2f}×",        "High" if L['rvol']>1.5 else "Normal")
        c6.metric("BB %B",      f"{L['bb_pct']:.2f}",       "Upper" if L['bb_pct']>0.8 else ("Lower" if L['bb_pct']<0.2 else "Mid"))
        c7.metric("Supertrend", "BULL" if L['supertrend_dir']==1 else "BEAR", None)
        c8.metric("Squeeze",    "ON" if L.get('bb_squeeze', False) else "OFF", None)

        # Pivots
        pivots = calculate_pivots(df_c)
        swings = find_swing_levels(df_c)
        pv_html = ''.join(
            f'<div style="background:#0a1018;border:1px solid #1a2535;border-radius:2px;padding:4px 9px;font-size:0.62rem"><span style="color:#2a3f55;font-size:0.5rem">{k.upper()} </span><span style="color:#b0c8e0">${v:,.2f}</span></div>'
            for k,v in list(pivots.items())[:8]
        )
        st.markdown(f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin:8px 0">{pv_html}</div>', unsafe_allow_html=True)

        fig = make_chart(df_c, chart_sym, sigs_c)
        st.plotly_chart(fig, use_container_width=True)

        # Swing levels
        if swings['resistance_zones'] or swings['support_zones']:
            rc_html = ''.join(f'<span style="background:#2a0a10;border:1px solid #5a1020;border-radius:2px;padding:2px 7px;font-size:0.6rem;color:#ff6680;margin:2px">${v:,.2f}</span>' for v in swings['resistance_zones'][-5:])
            sc_html = ''.join(f'<span style="background:#0a2a10;border:1px solid #105a20;border-radius:2px;padding:2px 7px;font-size:0.6rem;color:#66ff99;margin:2px">${v:,.2f}</span>' for v in swings['support_zones'][:5])
            st.markdown(f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin:4px 0"><div><span style="font-size:0.5rem;color:#2a3f55;letter-spacing:0.15em">RESISTANCE ZONES </span>{rc_html}</div><div><span style="font-size:0.5rem;color:#2a3f55;letter-spacing:0.15em">SUPPORT ZONES </span>{sc_html}</div></div>', unsafe_allow_html=True)
    else:
        st.error(f"Could not fetch {chart_sym}. Check API key or connection.")

# ── REGIME TAB ────────────────────────────────────────────────────────────
with tab_regime:
    st.markdown('<div style="font-size:0.6rem;letter-spacing:0.2em;color:#2a3f55;margin-bottom:12px">MARKET REGIME ANALYSIS · ALL SCANNED SYMBOLS</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(symbols), 4))
    for i, sym in enumerate(symbols):
        res = results.get(sym, {})
        df_r = res.get('df')
        with cols[i % len(cols)]:
            if df_r is not None:
                L = df_r.iloc[-1]
                regime = str(df_r['market_regime'].iloc[-1])
                rcol = REGIME_COLORS.get(regime, ('#6a8aa0','#0a1018'))
                adx  = L.get('adx', 0)
                rsi  = L['rsi']
                rvol = L.get('rvol', 1.0)
                hv   = L.get('hv_20', np.nan)
                pc   = (L['close'] - df_r.iloc[-2]['close']) / df_r.iloc[-2]['close'] * 100
                st.markdown(f"""
<div style="border:1px solid {rcol[0]}44;background:{rcol[1]};border-radius:3px;padding:14px;margin-bottom:10px">
  <div style="font-size:0.6rem;color:#2a3f55;letter-spacing:0.2em">{sym}</div>
  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;color:{rcol[0]};margin:4px 0">{regime}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px">
    <div style="font-size:0.6rem"><span style="color:#2a3f55">Price</span><br><span style="color:#b0c8e0">${L['close']:,.2f} <span style="color:{'#00e676' if pc>=0 else '#ff1744'}">{pc:+.2f}%</span></span></div>
    <div style="font-size:0.6rem"><span style="color:#2a3f55">ADX</span><br><span style="color:{'#00e676' if adx>25 else '#ffab00'}">{adx:.1f}</span></div>
    <div style="font-size:0.6rem"><span style="color:#2a3f55">RSI</span><br><span style="color:{'#ff1744' if rsi>70 else ('#00e676' if rsi<30 else '#b0c8e0')}">{rsi:.1f}</span></div>
    <div style="font-size:0.6rem"><span style="color:#2a3f55">RVOL</span><br><span style="color:{'#00e676' if rvol>1.5 else '#b0c8e0'}">{rvol:.2f}×</span></div>
    <div style="font-size:0.6rem"><span style="color:#2a3f55">Supertrend</span><br><span style="color:{'#00e676' if L.get('supertrend_dir',0)==1 else '#ff1744'}">{'BULL' if L.get('supertrend_dir',0)==1 else 'BEAR'}</span></div>
    <div style="font-size:0.6rem"><span style="color:#2a3f55">Squeeze</span><br><span style="color:{'#ffab00' if L.get('bb_squeeze',False) else '#2a3f55'}">{'ON' if L.get('bb_squeeze',False) else 'OFF'}</span></div>
  </div>
  <div style="margin-top:8px;font-size:0.55rem;color:#2a3f55">{len(res.get('signals',[]))} signals detected</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="border:1px dashed #1a2535;border-radius:3px;padding:14px;color:#1a3050;font-size:0.6rem">{sym}<br>DATA UNAVAILABLE</div>', unsafe_allow_html=True)

# ── LOG TAB ───────────────────────────────────────────────────────────────
with tab_log:
    log = st.session_state.signal_log
    if not log:
        st.markdown('<div class="no-sig"><div class="no-sig-icon">—</div><div class="no-sig-title">LOG EMPTY</div><div class="no-sig-sub">Signals will appear here after scanning</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.55rem;color:#2a3f55;margin-bottom:8px">{len(log)} signals · Session history</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="log-hdr">
  <div>Time</div><div>Symbol</div><div>Strategy</div>
  <div>Dir</div><div>Grade</div><div>Entry</div><div>Regime</div>
</div>
""", unsafe_allow_html=True)
        for e in log:
            dc = '#00e676' if e['dir'] == 'LONG' else '#ff1744'
            gc_map = {'S':'#ffab00','A':'#00e676','B':'#2979ff','C':'#b0c8e0','D':'#2a3f55'}
            gc = gc_map.get(e['grade'], '#2a3f55')
            rc_t = REGIME_COLORS.get(e.get('regime',''), ('#6a8aa0',''))[0]
            st.markdown(f"""
<div class="log-row">
  <div style="color:#2a3f55">{e['ts']}</div>
  <div style="color:#b0c8e0">{e['sym']}</div>
  <div style="color:#4a6a80">{e['strat']}</div>
  <div style="color:{dc};font-weight:600">{e['dir']}</div>
  <div style="color:{gc};font-weight:600">{e['grade']} <span style="color:#2a3f55;font-size:0.55rem">({e['conf']}%)</span></div>
  <div style="color:#b0c8e0">${e['entry']:,.2f}</div>
  <div style="color:{rc_t};font-size:0.55rem">{e.get('regime','?')}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# AUTO REFRESH
# ══════════════════════════════════════════════════════════════════════════════
if auto_ref:
    time.sleep(60)
    st.session_state.last_scan  = datetime.now()
    st.session_state.scan_count += 1
    st.rerun()