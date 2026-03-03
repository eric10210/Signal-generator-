# ⚡ PRO SIGNAL TERMINAL — v2

> Bybit · 7 Strategies · Regime-Aware Scoring · Signal Generator Only · No Auto-Trading

A professional-grade crypto signal terminal built with Streamlit. Scans multiple symbols simultaneously, detects high-probability setups across 7 strategies, scores each signal with multi-factor confluence analysis, and presents full trade plans with entry, stop loss, and three take-profit targets.

---

## What It Does

- Connects to Bybit (public data or with API key) and fetches live OHLCV candles
- Runs a full 20-indicator suite on every symbol
- Detects signals across 7 independent strategies
- Scores each signal 0–100 with regime-adjusted weighting
- Calculates entry, SL, TP1/TP2/TP3, R:R ratio, and position size
- Displays a 5-panel interactive chart per symbol
- Classifies market regime per symbol (Trending Up/Down, Ranging, Volatile)
- Logs every signal detected during the session

---

## Project Files

```
app.py              Main Streamlit application  (784 lines)
signals.py          7-strategy signal engine    (655 lines)
indicators.py       20-indicator suite          (292 lines)
requirements.txt    Python dependencies
.gitignore          Keeps secrets out of git
.streamlit/
    secrets.toml    API keys — local only, never commit
```

---

## Strategies

| # | Strategy | Best Regime | Signal Type |
|---|---|---|---|
| 1 | BB/Keltner Squeeze Bounce | Ranging | Mean Reversion |
| 2 | Volume Momentum Breakout | Trending | Continuation |
| 3 | EMA Trend Pullback | Trending Up/Down | Re-entry |
| 4 | RSI Divergence | Ranging / Volatile | Reversal |
| 5 | Ichimoku TK Cross | Trending | Trend Confirmation |
| 6 | Supertrend Flip | Any | Trend Change |
| 7 | VWAP Reversion | Ranging / Volatile | Mean Reversion |

Each strategy scores 0–100 based on how many confluence factors are present, then that score is multiplied by a regime weight (0.6–1.4×) so strategies are naturally suppressed when market conditions don't suit them.

---

## Indicator Suite

**Volatility**
ATR (14) · Historical Volatility (20) · Bollinger Bands (20, 2σ) · Keltner Channels · BB Squeeze flag

**Momentum**
RSI (14) + slope · MACD (12/26/9) + cross flags · Stochastic RSI (14/14/3/3) + cross flags · Williams %R (14) · CCI (20)

**Trend**
EMA stack (9/20/50/100/200) · Full alignment flags · ADX + DI± (14) · Supertrend (10, 3×) + flip detection · Ichimoku (Tenkan/Kijun/Span A/Span B/Chikou + cloud flags)

**Volume**
Relative Volume (20 SMA) · OBV + EMA + slope · Money Flow Index (14) · VWAP

**Structure**
Classic pivots (P/R1/R2/R3/S1/S2/S3) · Fibonacci retracements (23.6% / 38.2% / 50% / 61.8% / 78.6%) · Swing high/low clustering for S/R zones

**Regime Classifier**
Outputs: TRENDING_UP · TRENDING_DOWN · TRENDING · RANGING · VOLATILE

---

## Signal Grades

| Grade | Confidence | Meaning |
|---|---|---|
| S | 85–100% | Exceptional confluence — rare, high-quality setup |
| A | 70–84%  | Strong setup — multiple factors aligned |
| B | 55–69%  | Good setup — most factors present |
| C | 40–54%  | Marginal setup — use caution |
| D | < 40%   | Filtered out by default |

---

## Risk Calculator

Every signal includes an automatic position size calculation:

- **Risk Amount** = Account Balance × Risk %
- **Stop Loss** = Entry ± (ATR × strategy multiplier)
- **Position Size** = Risk Amount ÷ Distance to Stop Loss
- **Take Profit 1** = 2.0–2.5× ATR from entry
- **Take Profit 2** = 3.5–4.5× ATR from entry
- **Take Profit 3** = 6.0–8.0× ATR from entry

These are display calculations only. No orders are placed.

---

## Deploy to Streamlit Cloud

### Step 1 — Push to GitHub

Do NOT include `.streamlit/secrets.toml` in the commit.

```bash
git init
git add app.py signals.py indicators.py requirements.txt .gitignore
git commit -m "pro signal terminal v2"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Create app on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo, branch (`main`), and entry point (`app.py`)
5. Click **Deploy**

### Step 3 — Add API keys

In the Streamlit Cloud dashboard:
**App → ⋮ → Settings → Secrets**

```toml
[bybit]
api_key = "your_bybit_api_key"
secret  = "your_bybit_secret"
```

Click Save. The app restarts automatically.

> The terminal works without API keys using Bybit's public endpoints. Keys are only needed if Bybit restricts unauthenticated access from Streamlit's servers.

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Add keys for local use (already git-ignored)
mkdir -p .streamlit
nano .streamlit/secrets.toml

# Run
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Get a Bybit API Key

1. Log in at [bybit.com](https://www.bybit.com)
2. Go to **Account → API Management → Create New Key**
3. Set key type: **System-generated**
4. Permissions: **Read only** (nothing else needed)
5. Optionally whitelist your IP or Streamlit Cloud's IP range
6. Copy the API Key and Secret — secret is shown once only

---

## Sidebar Settings

| Setting | Description |
|---|---|
| Symbols | Up to 15 pairs available — select any combination |
| Timeframe | 5m · 15m · 30m · 1h · 2h · 4h · 1d |
| Min Confidence | Filter out signals below this threshold (default 50%) |
| Grades | Filter by signal grade S / A / B / C |
| Long / Short | Toggle direction filters independently |
| Account Balance | Used for position size calculation only |
| Risk % | Percentage of balance to risk per trade |
| Auto-refresh | Re-scans every 60 seconds automatically |

---

## Chart Panels

The 5-panel chart includes:

1. **Price** — Candlesticks · Ichimoku cloud · Bollinger Bands · EMA 20/50/200 · Supertrend · VWAP · Signal markers + SL/TP lines
2. **Volume** — Bar chart · 20-period SMA overlay
3. **RSI** — With overbought/oversold zones (70/30)
4. **MACD** — Histogram · Signal line · MACD line
5. **Stochastic RSI** — %K and %D with overbought/oversold zones (80/20)

---

## Market Regime Tab

Shows a per-symbol regime card with:
- Classified regime (TRENDING_UP / TRENDING_DOWN / RANGING / VOLATILE)
- Price change, ADX, RSI, RVOL, Supertrend direction, Squeeze status
- Number of signals detected for that symbol

---

## ⚠ Important Disclaimers

This tool is for **informational and educational purposes only.**

- It does not place, manage, or cancel any trades
- Signals are generated by technical algorithms and do not constitute financial advice
- Past signal accuracy does not guarantee future performance
- Always use a testnet account to validate behaviour before live trading
- Crypto markets are highly volatile — only risk capital you can afford to lose
- The developer assumes no responsibility for trading decisions made using this tool

---

## Test on Bybit Testnet First

Before connecting live keys, test on the Bybit paper trading environment:

1. Create a testnet account at [testnet.bybit.com](https://testnet.bybit.com)
2. Generate API keys there
3. The terminal will display signals from testnet market data

---

## License

MIT — use freely, trade responsibly.
