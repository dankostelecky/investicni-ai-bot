# ============================================================
# KLONDIKE SPOT SCANNER 4.0 — sjednocená verze
# Spot swing decision-support scanner s feedback loopem
# ============================================================

import re
import time
import logging
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from supabase import create_client

# ---------------------- LOGOVÁNÍ ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("klondike")

# ---------------------- KONFIGURACE ---------------------------
st.set_page_config(
    page_title="Klondike Spot Scanner 4.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- CSS -----------------------------------
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #f9fafb !important; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .4rem 0 0; color: #d1d5db; }
    .buyzone {
        background: linear-gradient(135deg,#ecfdf5,#f0fdf4);
        border: 1px solid #86efac; border-radius: 15px; padding: 1rem;
    }
    .waitzone {
        background: linear-gradient(135deg,#fffbeb,#fefce8);
        border: 1px solid #fde68a; border-radius: 15px; padding: 1rem;
    }
    .dangerzone {
        background: linear-gradient(135deg,#fef2f2,#fff1f2);
        border: 1px solid #fca5a5; border-radius: 15px; padding: 1rem;
    }
    .small { color:#6b7280; font-size:.85rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e5e7eb;
        padding: .7rem; border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------- TICKERY -------------------------------
DEFAULT_TICKERS = [
    "NVDA","AAPL","GOOGL","MSFT","AMZN","META","AVGO","TSLA","BRK-B",
    "WMT","LLY","MU","JPM","ORCL","XOM","V","MA","AMD","JNJ","COST",
    "HD","CRM","UNH","PG","ABBV","BAC","IBM","DIS","INTC","KO","PLTR","NKE",
    "UBER","PYPL","PFE","BABA","SPY"
]

TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")

# ---------------------- POMOCNÉ FUNKCE ------------------------

def safe_float(x, default=np.nan) -> float:
    """Vrací default i pro NaN/inf, ne jen při výjimce."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def is_valid_ticker(t: str) -> bool:
    return bool(TICKER_RE.match(t or ""))


# ---------------------- DATOVÉ FUNKCE -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Stáhne historii z Yahoo Finance s retry a ošetřením MultiIndex sloupců."""
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" not in df.columns:
                return pd.DataFrame()
            return df.dropna(subset=["Close"]).copy()
        except Exception as e:
            log.warning(f"download_history({ticker}) pokus {attempt+1} selhal: {e}")
            time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def get_spy() -> pd.DataFrame:
    return download_history("SPY", "1y", "1d")


@st.cache_data(ttl=86400, show_spinner=False)
def basic_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", ticker), info.get("sector", "N/A")
    except Exception:
        return ticker, "N/A"


@st.cache_data(ttl=300, show_spinner=False)
def premarket(ticker: str):
    try:
        obj = yf.Ticker(ticker)
        fast = obj.fast_info
        pre, prev = None, None
        try:
            pre = fast.get("pre_market_price")
            prev = fast.get("previous_close") or fast.get("regular_market_previous_close")
        except Exception:
            pass
        if pre is None or prev is None:
            info = obj.info
            pre = info.get("preMarketPrice")
            prev = info.get("previousClose")
        if pre is not None and prev:
            return float(pre), (float(pre) / float(prev) - 1) * 100
    except Exception:
        pass
    return None, None


# ---------------------- INDIKÁTORY ----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    d["EMA20"]  = c.ewm(span=20,  adjust=False).mean()
    d["EMA50"]  = c.ewm(span=50,  adjust=False).mean()
    d["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))
    d.loc[(loss < 1e-12) & (gain > 1e-12), "RSI14"] = 100
    d.loc[(loss < 1e-12) & (gain <= 1e-12), "RSI14"] = 50

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()

    macd   = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    d["MACD"]        = macd
    d["MACD_SIGNAL"] = signal
    d["MACD_HIST"]   = macd - signal

    d["VOL20"]     = v.rolling(20).mean()
    d["VOL_RATIO"] = v / d["VOL20"].replace(0, np.nan)

    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    d["BB_MID"]   = mid
    d["BB_UPPER"] = mid + 2 * std
    d["BB_LOWER"] = mid - 2 * std
    d["BB_WIDTH"] = (d["BB_UPPER"] - d["BB_LOWER"]) / mid.replace(0, np.nan)

    d["HIGH20"]  = c.rolling(20).max()
    d["LOW20"]   = c.rolling(20).min()
    d["HIGH52W"] = c.rolling(252, min_periods=60).max()
    d["LOW52W"]  = c.rolling(252, min_periods=60).min()

    return d


def detect_patterns(d: pd.DataFrame) -> dict:
    """Vzorce z druhého skeneru: průraz, propad, squeeze a potvrzení objemem."""
    close = d["Close"]
    high20 = close.rolling(20).max()
    low20 = close.rolling(20).min()
    breakout = bool(len(d) >= 21 and close.iloc[-1] >= high20.iloc[-2])
    breakdown = bool(len(d) >= 21 and close.iloc[-1] <= low20.iloc[-2])
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    bandwidth = (4 * std) / mid.replace(0, np.nan)
    baseline = bandwidth.rolling(50, min_periods=20).mean().iloc[-1]
    squeeze = bool(pd.notna(baseline) and pd.notna(bandwidth.iloc[-1])
                   and bandwidth.iloc[-1] < baseline * 0.8)
    volume_spike = bool(safe_float(d["VOL_RATIO"].iloc[-1], 0) >= 1.5)
    return {"breakout": breakout, "breakdown": breakdown,
            "squeeze": squeeze, "volume_spike": volume_spike}


# ---------------------- SUPPORT / RESISTANCE ------------------
def support_resistance(d: pd.DataFrame):
    last = float(d["Close"].iloc[-1])
    lookback = min(120, len(d))
    recent = d.iloc[-lookback:]
    lo = recent["Low"].to_numpy()
    hi = recent["High"].to_numpy()

    supports, resistances = [], []

    if len(lo) >= 5:
        mask_low = (
            (lo[2:-2] < lo[:-4]) & (lo[2:-2] < lo[1:-3]) &
            (lo[2:-2] < lo[3:-1]) & (lo[2:-2] < lo[4:])
        )
        supports.extend(lo[2:-2][mask_low].tolist())

        mask_high = (
            (hi[2:-2] > hi[:-4]) & (hi[2:-2] > hi[1:-3]) &
            (hi[2:-2] > hi[3:-1]) & (hi[2:-2] > hi[4:])
        )
        resistances.extend(hi[2:-2][mask_high].tolist())

    for col in ["EMA20", "EMA50", "EMA200", "BB_LOWER", "BB_UPPER"]:
        if col in d.columns and pd.notna(d[col].iloc[-1]):
            val = float(d[col].iloc[-1])
            if val < last:
                supports.append(val)
            elif val > last:
                resistances.append(val)

    supports    = [s for s in supports    if 0.85 * last < s < last]
    resistances = [r for r in resistances if last < r < 1.15 * last]

    support = max(supports, default=float(d["LOW20"].iloc[-1]))
    resistance = min(resistances, default=float(d["HIGH20"].iloc[-1]))

    if not np.isfinite(support) or support <= 0:
        support = last * 0.95
    if not np.isfinite(resistance) or resistance <= last:
        resistance = last * 1.05

    return float(support), float(resistance)


# ---------------------- TRŽNÍ REŽIM ---------------------------
def market_regime(spy: pd.DataFrame):
    if spy.empty or len(spy) < 60:
        return "UNKNOWN", 0
    s = add_indicators(spy)
    last = s.iloc[-1]
    score = 0
    if last["Close"]   > last["EMA50"]:  score += 1
    if last["EMA20"]   > last["EMA50"]:  score += 1
    if last["EMA50"]   > last["EMA200"]: score += 1
    if last["MACD_HIST"] > 0:            score += 1
    if last["RSI14"]   >= 50:            score += 1
    if score >= 4: return "BULLISH", score
    if score <= 1: return "BEARISH", score
    return "NEUTRAL", score


# ---------------------- SKÓROVÁNÍ -----------------------------
def calculate_scores(d: pd.DataFrame, spy: pd.DataFrame) -> dict:
    x = d.iloc[-1]
    price = float(x["Close"])

    trend = 50
    if price > x["EMA20"]:        trend += 10
    if x["EMA20"] > x["EMA50"]:   trend += 12
    if x["EMA50"] > x["EMA200"]:  trend += 13
    if x["MACD_HIST"] > 0:        trend += 10
    trend = float(np.clip(trend, 0, 100))

    rsi = safe_float(x["RSI14"], 50)
    momentum = 50
    if 50 <= rsi <= 68:    momentum += 20
    elif rsi > 68:         momentum += 5
    elif 40 <= rsi < 50:   momentum -= 5
    else:                  momentum -= 15
    if x["MACD_HIST"] > 0: momentum += 15
    if x["VOL_RATIO"] >= 1.2: momentum += 10
    momentum = float(np.clip(momentum, 0, 100))

    rs = 0.0
    if len(spy) >= 31 and len(d) >= 31:
        stock_ret = (price / float(d["Close"].iloc[-31]) - 1) * 100
        spy_ret   = (float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-31]) - 1) * 100
        rs = stock_ret - spy_ret
    rs_score = float(50 + 50 * np.tanh(rs / 10))

    atr_pct = safe_float(x["ATR14"], 0) / price * 100 if price else 99
    risk = 75
    if atr_pct > 5:   risk -= 25
    elif atr_pct > 3: risk -= 12
    if rsi > 75:      risk -= 20
    if price < x["EMA200"]: risk -= 15
    risk = float(np.clip(risk, 0, 100))

    quality = round(trend * .30 + momentum * .25 + rs_score * .25 + risk * .20, 1)

    return {
        "trend_score":    round(trend, 1),
        "momentum_score": round(momentum, 1),
        "rs_score":       round(rs_score, 1),
        "risk_score":     round(risk, 1),
        "quality_score":  quality,
        "rs_30d":         round(rs, 2),
        "atr_pct":        round(atr_pct, 2),
    }


# ---------------------- ENTRY ENGINE --------------------------
def calculate_entry_engine(d: pd.DataFrame, scores: dict) -> dict:
    x = d.iloc[-1]
    price = float(x["Close"])
    atr = safe_float(x["ATR14"], price * .03)
    support, resistance = support_resistance(d)

    candidates = [
        support,
        safe_float(x["EMA20"]),
        safe_float(x["EMA50"]),
        safe_float(x["BB_LOWER"]),
        price - atr,
        price - 1.5 * atr,
    ]
    candidates = [v for v in candidates if np.isfinite(v) and 0 < v < price]

    preferred = max(candidates, default=price - atr)
    preferred = max(preferred, price - 2.0 * atr)

    aggressive    = min(price - 0.5 * atr, price)
    conservative  = max(preferred - 0.8 * atr, price - 2.5 * atr)

    zone_low  = max(0.01, preferred - 0.35 * atr)
    zone_high = min(price, preferred + 0.35 * atr)

    structural_stop = support - 0.25 * atr
    atr_stop        = preferred - 1.5 * atr
    stop = max(0.01, min(structural_stop, atr_stop))

    risk_per_share = preferred - stop
    if risk_per_share <= 0:
        risk_per_share = 1.5 * atr
        stop = max(0.01, preferred - risk_per_share)

    target1 = max(resistance, preferred + 1.5 * risk_per_share)
    target2 = max(preferred + 2.5 * risk_per_share, target1 + 0.5 * atr)

    rr1 = (target1 - preferred) / risk_per_share if risk_per_share > 0 else 0
    rr2 = (target2 - preferred) / risk_per_share if risk_per_share > 0 else 0

    distance_to_zone = (price - zone_high) / price * 100 if price else 999

    entry_score = 50
    if price <= zone_high:               entry_score += 20
    if price <= float(x["EMA20"]):       entry_score += 10
    if 45 <= safe_float(x["RSI14"], 50) <= 65: entry_score += 10
    if scores["rs_30d"] > 0:             entry_score += 10
    if rr1 >= 2:                         entry_score += 10
    if safe_float(x["RSI14"], 50) > 75:  entry_score -= 20
    entry_score = float(np.clip(entry_score, 0, 100))

    if price <= zone_high and entry_score >= 65 and scores["quality_score"] >= 55:
        signal = "NÁKUPNÍ ZÓNA"
    elif price > zone_high and scores["quality_score"] >= 70:
        signal = "ČEKAT NA KOREKCI"
    elif scores["quality_score"] < 45:
        signal = "VYHNOUT SE / SLABÉ"
    else:
        signal = "SLEDOVAT"

    return {
        "support":             round(support, 2),
        "resistance":          round(resistance, 2),
        "aggressive_entry":    round(aggressive, 2),
        "preferred_entry":     round(preferred, 2),
        "conservative_entry":  round(conservative, 2),
        "zone_low":            round(zone_low, 2),
        "zone_high":           round(zone_high, 2),
        "stop":                round(stop, 2),
        "target1":             round(target1, 2),
        "target2":             round(target2, 2),
        "risk_per_share":      round(risk_per_share, 2),
        "rr1":                 round(rr1, 2),
        "rr2":                 round(rr2, 2),
        "entry_score":         round(entry_score, 1),
        "distance_to_zone":    round(distance_to_zone, 2),
        "signal":              signal,
    }


# ---------------------- POSITION SIZE -------------------------
def position_size(capital, risk_pct, entry, stop, max_position_pct):
    values = (capital, risk_pct, entry, stop, max_position_pct)
    if (not all(np.isfinite(v) for v in values)
            or capital <= 0 or entry <= 0 or stop <= 0 or stop >= entry):
        return 0, 0.0, 0.0, 0.0
    max_risk = capital * risk_pct / 100
    risk_per_share = max(entry - stop, 0.0001)
    shares_by_risk = int(max_risk / risk_per_share)

    max_position_value = capital * max_position_pct / 100
    shares_by_capital = int(max_position_value / entry) if entry > 0 else 0

    shares = max(0, min(shares_by_risk, shares_by_capital))
    return shares, shares * entry, max_risk, risk_per_share


# ---------------------- SUPABASE ------------------------------
@st.cache_resource(show_spinner=False)
def get_supabase():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError:
        log.warning("SUPABASE_URL/KEY chybí v secrets.toml – historie vypnuta.")
        return None
    except Exception as e:
        log.warning(f"Supabase není připojeno: {e}")
        return None


def build_signal_payload(r: dict) -> dict:
    return {
        "ticker":           r["ticker"],
        "signal_date":      datetime.now(timezone.utc).date().isoformat(),
        "price":            r["price"],
        "quality_score":    r["quality_score"],
        "entry_score":      r["entry_score"],
        "trend_score":      r["trend_score"],
        "momentum_score":   r["momentum_score"],
        "rs_score":         r["rs_score"],
        "risk_score":       r["risk_score"],
        "preferred_entry":  r["preferred_entry"],
        "zone_low":         r["zone_low"],
        "zone_high":        r["zone_high"],
        "stop_price":       r["stop"],
        "target1":          r["target1"],
        "target2":          r["target2"],
        "rr1":              r["rr1"],
        "rr2":              r["rr2"],
        "support":          r["support"],
        "resistance":       r["resistance"],
        "rsi":              r["rsi"],
        "atr":              r["atr"],
        "atr_pct":          r["atr_pct"],
        "volume_ratio":     r["volume_ratio"],
        "rs_30d":           r["rs_30d"],
        "signal":           r["signal"],
    }


def save_signals_bulk(sb, results: list):
    if sb is None or not results:
        return
    try:
        payloads = [build_signal_payload(r) for r in results]
        sb.table("scanner_signals").upsert(
            payloads, on_conflict="ticker,signal_date"
        ).execute()
        log.info(f"Uloženo {len(payloads)} signálů.")
        return len(payloads)
    except Exception as e:
        log.error(f"save_signals_bulk selhalo: {e}", exc_info=True)
        return 0


def load_learning_stats(sb, days: int = 90):
    if sb is None:
        return pd.DataFrame()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        resp = sb.table("scanner_signals").select(
            "ticker,signal_date,quality_score,entry_score,signal,"
            "price,preferred_entry,stop_price,target1,target2"
        ).gte("signal_date", cutoff).execute()
        df = pd.DataFrame(resp.data or [])
        if df.empty:
            return df

        tickers = df["ticker"].unique().tolist()
        price_map = {}
        for t in tickers:
            hist = download_history(t, "1y", "1d")
            if not hist.empty:
                price_map[t] = hist

        n = len(df)
        fwd_returns = np.full(n, np.nan, dtype=object)
        fwd_outcomes = np.full(n, "unknown", dtype=object)

        for i, (_, row) in enumerate(df.iterrows()):
            t = row["ticker"]
            if t not in price_map:
                continue
            s = price_map[t]
            try:
                target_day = pd.Timestamp(row["signal_date"]).date()
                dates = pd.Index([ts.date() for ts in s.index])
                matches = np.flatnonzero(dates == target_day)
                if len(matches) == 0:
                    continue
                start_idx = int(matches[0])
            except Exception:
                continue

            try:
                entry = float(row["preferred_entry"])
                stop  = float(row["stop_price"])
                tp1   = float(row["target1"])
            except Exception:
                continue

            window = s.iloc[start_idx + 1:start_idx + 21]
            if window.empty:
                fwd_outcomes[i] = "pending"
                continue

            outcome = "OPEN"
            for day_idx in range(start_idx + 1, min(start_idx + 21, len(s))):
                day = s.iloc[day_idx]
                hit_tp = float(day["High"]) >= tp1
                hit_sl = float(day["Low"]) <= stop
                if hit_tp and hit_sl:
                    outcome = "AMBIGUOUS"
                    break
                if hit_tp:
                    outcome = "TP1"
                    break
                if hit_sl:
                    outcome = "SL"
                    break

            final_ret = (float(window["Close"].iloc[-1]) / entry - 1) * 100
            fwd_returns[i] = round(final_ret, 2)
            fwd_outcomes[i] = outcome

        df["forward_return_20d"] = fwd_returns
        df["outcome"] = fwd_outcomes
        return df
    except Exception as e:
        log.error(f"load_learning_stats selhalo: {e}", exc_info=True)
        return pd.DataFrame()


# ---------------------- SCAN TICKERU --------------------------
def scan_ticker(ticker: str, spy: pd.DataFrame):
    data = download_history(ticker)
    if data.empty or len(data) < 60:
        return None

    d = add_indicators(data)
    x = d.iloc[-1]

    required = ["EMA20", "EMA50", "EMA200", "RSI14", "ATR14", "VOL_RATIO"]
    if any(pd.isna(x[k]) for k in required):
        return None

    scores = calculate_scores(d, spy)
    entry  = calculate_entry_engine(d, scores)
    patterns = detect_patterns(d)

    return {
        "ticker":       ticker,
        "name":         ticker,
        "sector":       "N/A",
        "data":         d,
        "price":        float(x["Close"]),
        "rsi":          float(x["RSI14"]),
        "atr":          float(x["ATR14"]),
        "volume_ratio": float(x["VOL_RATIO"]),
        "ema20":        float(x["EMA20"]),
        "ema50":        float(x["EMA50"]),
        "ema200":       float(x["EMA200"]),
        "macd_hist":    float(x["MACD_HIST"]),
        **patterns,
        "high52":       float(x["HIGH52W"]) if pd.notna(x["HIGH52W"]) else np.nan,
        "low52":        float(x["LOW52W"])  if pd.notna(x["LOW52W"])  else np.nan,
        **scores,
        **entry,
    }


# ---------------------- PARALELNÍ SCAN ------------------------
def parallel_scan(tickers: list, spy: pd.DataFrame, max_workers: int = 4, progress_cb=None):
    results = []
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_ticker, t, spy): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                log.warning(f"Scan {t} selhal: {e}")
            done += 1
            if progress_cb:
                progress_cb(done, total)
    return results


# ============================================================
# =========================== UI =============================
# ============================================================

st.markdown("""
<div class="hero">
    <h1>📈 Klondike Spot Scanner 4.0</h1>
    <p>Technická analýza • Nákupní zóny • Risk management • Učící se historie signálů</p>
</div>
""", unsafe_allow_html=True)

sb = get_supabase()

# ---------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("⚙️ Nastavení")

    custom = st.text_input("Přidat ticker", "").upper().strip()
    tickers = DEFAULT_TICKERS.copy()
    if custom:
        if is_valid_ticker(custom):
            if custom not in tickers:
                tickers.insert(0, custom)
        else:
            st.warning("Neplatný ticker (A–Z, 0–9, tečka, pomlčka; max 10 znaků).")

    st.markdown("---")
    st.subheader("💰 Risk management")
    capital          = st.number_input("Kapitál ($)", min_value=100.0, value=5000.0, step=500.0)
    risk_pct         = st.slider("Riziko na obchod (%)", 0.25, 3.0, 1.0, 0.25)
    max_position_pct = st.slider("Max. velikost pozice (%)", 5, 100, 25, 5)

    st.markdown("---")
    st.subheader("🔎 Filtry")
    min_quality        = st.slider("Min. Quality Score", 0, 100, 60)
    only_buy_zone      = st.checkbox("Pouze NÁKUPNÍ ZÓNA", False)
    only_positive_rs   = st.checkbox("Pouze RS > S&P 500", False)
    exclude_overbought = st.checkbox("Vyloučit RSI > 75", True)
    only_breakout = st.checkbox("Pouze cenové průrazy", False)
    only_squeeze = st.checkbox("Pouze BB squeeze", False)
    only_volume_spike = st.checkbox("Pouze zvýšený objem", False)
    max_workers        = st.slider("Paralelní vlákna", 2, 12, 4)

    st.markdown("---")
    st.caption("Zdroj dat: Yahoo Finance")
    st.caption("Analytická pomůcka – ne automatický obchodní systém.")

# ---------------------- TABS --------------------------------
tab_scan, tab_detail, tab_history, tab_learning = st.tabs([
    "🔎 Scanner", "📊 Detail", "🗄️ Historie signálů", "🧠 Učící se přehled"
])

if "results" not in st.session_state:
    st.session_state.results = []
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "regime" not in st.session_state:
    st.session_state.regime = None
if "regime_score" not in st.session_state:
    st.session_state.regime_score = 0

# ============================================================
# TAB 1 – SCANNER
# ============================================================
with tab_scan:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader("Tržní sken")
    with col_b:
        run = st.button("🚀 SPUSTIT SKEN", type="primary", use_container_width=True)

    if run:
        progress_bar = st.progress(0.0, text="Analyzuji trh…")

        def update_progress(done, total):
            progress_bar.progress(done / max(total, 1),
                                  text=f"Analyzuji {done}/{total}")

        spy = get_spy()
        regime, regime_score = market_regime(spy)
        st.session_state.regime = regime
        st.session_state.regime_score = regime_score

        raw = parallel_scan(tickers, spy, max_workers=max_workers, progress_cb=update_progress)
        progress_bar.empty()

        results = []
        for r in raw:
            if r["quality_score"] < min_quality: continue
            if only_buy_zone and r["signal"] != "NÁKUPNÍ ZÓNA": continue
            if only_positive_rs and r["rs_30d"] <= 0: continue
            if exclude_overbought and r["rsi"] > 75: continue
            if only_breakout and not r["breakout"]: continue
            if only_squeeze and not r["squeeze"]: continue
            if only_volume_spike and not r["volume_spike"]: continue
            results.append(r)

        results.sort(key=lambda z: (z["entry_score"], z["quality_score"]), reverse=True)
        st.session_state.results = results

        saved = save_signals_bulk(sb, results)
        if saved:
            st.toast(f"Uloženo {saved} signálů do Supabase.", icon="💾")

    # Zobraz režim trhu (přetrvává mezi reruny)
    if st.session_state.regime:
        color = {"BULLISH": "🟢", "BEARISH": "🔴",
                 "NEUTRAL": "🟡", "UNKNOWN": "⚪"}.get(st.session_state.regime, "⚪")
        st.info(f"{color} **Tržní režim (SPY):** {st.session_state.regime} — "
                f"skóre {st.session_state.regime_score}/5")

    results = st.session_state.results

    if results:
        buy_count   = sum(r["signal"] == "NÁKUPNÍ ZÓNA" for r in results)
        avg_quality = np.mean([r["quality_score"] for r in results])
        avg_entry   = np.mean([r["entry_score"]   for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzovaných",   len(results))
        c2.metric("NÁKUPNÍ ZÓNA",    buy_count)
        c3.metric("Průměr Quality",  f"{avg_quality:.1f}")
        c4.metric("Průměr Entry",    f"{avg_entry:.1f}")

        rows = []
        for r in results:
            rows.append({
                "Ticker":   r["ticker"],
                "Cena":     round(r["price"], 2),
                "Signál":   r["signal"],
                "Quality":  r["quality_score"],
                "Entry":    r["entry_score"],
                "ZÓNA OD":  round(r["zone_low"], 2),
                "ZÓNA DO":  round(r["zone_high"], 2),
                "Vstup":    round(r["preferred_entry"], 2),
                "Stop":     round(r["stop"], 2),
                "TP1":      round(r["target1"], 2),
                "R:R":      round(r["rr1"], 2),
                "RS 30d":   r["rs_30d"],
                "RSI":      round(r["rsi"], 1),
                "Objem":    round(r["volume_ratio"], 2),
                "Průraz":   r["breakout"],
                "Squeeze":  r["squeeze"],
                "Objemový spike": r["volume_spike"],
            })
        df_show = pd.DataFrame(rows)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Stáhnout CSV",
            df_show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"klondike_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.markdown("### 📌 Kandidáti")
        for r in results:
            box_class = (
                "buyzone"    if r["signal"] == "NÁKUPNÍ ZÓNA"
                else "dangerzone" if r["signal"] == "VYHNOUT SE / SLABÉ"
                else "waitzone"
            )

            with st.expander(
                f"{r['ticker']}  |  ${r['price']:.2f}  |  {r['signal']}  |  "
                f"Quality {r['quality_score']:.0f}  |  Entry {r['entry_score']:.0f}"
            ):
                st.markdown(f"""
                <div class="{box_class}">
                    <b>{r['ticker']} — {r['signal']}</b><br>
                    Cena: ${r['price']:.2f} | Vstupní zóna: ${r['zone_low']:.2f}–${r['zone_high']:.2f}<br>
                    Preferovaný vstup: ${r['preferred_entry']:.2f} | Stop: ${r['stop']:.2f}<br>
                    Cíl 1: ${r['target1']:.2f} (R:R {r['rr1']:.2f}) | Cíl 2: ${r['target2']:.2f}
                </div>
                """, unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("Trend", f"{r['trend_score']:.0f}/100")
                c2.metric("Momentum", f"{r['momentum_score']:.0f}/100")
                c3.metric("RS vs SPY (30d)", f"{r['rs_30d']:+.2f}%")
                st.caption(f"RSI {r['rsi']:.1f} · ATR {r['atr_pct']:.2f}% · "
                           f"Objem {r['volume_ratio']:.2f}× · R:R {r['rr1']:.2f}")
                labels = []
                if r["breakout"]: labels.append("cenový průraz")
                if r["breakdown"]: labels.append("riziko propadu")
                if r["squeeze"]: labels.append("BB squeeze")
                if r["volume_spike"]: labels.append("zvýšený objem")
                st.caption("Vzorce: " + (", ".join(labels) if labels else "bez výrazného vzorce"))
                if st.button("Zobrazit detail", key=f"detail_{r['ticker']}"):
                    st.session_state.selected_ticker = r["ticker"]
                    st.session_state.show_inline_detail = True

        # Zobraz detail hned zde, aby kliknutí nemuselo hledat jinou záložku.
        if st.session_state.get("show_inline_detail", False):
            detail_item = next(
                (item for item in results
                 if item["ticker"] == st.session_state.get("selected_ticker")),
                None,
            )
            if detail_item is not None:
                st.markdown("---")
                st.subheader(f"📊 Detail: {detail_item['ticker']}")
                meta_name, meta_sector = basic_info(detail_item["ticker"])
                pre_price, pre_change = premarket(detail_item["ticker"])
                st.caption(f"{meta_name} · Sektor: {meta_sector}")
                if pre_price is not None:
                    st.caption(f"Pre-market: ${pre_price:.2f} ({pre_change:+.2f} %)")

                detail_data = detail_item["data"].tail(180)
                detail_fig = go.Figure()
                detail_fig.add_trace(go.Candlestick(
                    x=detail_data.index, open=detail_data["Open"],
                    high=detail_data["High"], low=detail_data["Low"],
                    close=detail_data["Close"], name="Cena"
                ))
                for col, color in [("EMA20", "#2563eb"), ("EMA50", "#f59e0b"),
                                   ("EMA200", "#7c3aed")]:
                    detail_fig.add_trace(go.Scatter(
                        x=detail_data.index, y=detail_data[col], name=col,
                        line={"color": color, "width": 1.2}
                    ))
                for value, label, color in [
                    (detail_item["preferred_entry"], "Vstup", "#16a34a"),
                    (detail_item["stop"], "Stop", "#dc2626"),
                    (detail_item["target1"], "TP1", "#0891b2"),
                ]:
                    detail_fig.add_hline(y=value, line_dash="dash",
                                         annotation_text=label, line_color=color)
                detail_fig.update_layout(
                    height=500, xaxis_rangeslider_visible=False,
                    margin={"l": 10, "r": 10, "t": 30, "b": 10},
                    legend={"orientation": "h", "y": 1.05},
                )
                st.plotly_chart(detail_fig, use_container_width=True,
                                key=f"inline_detail_chart_{detail_item['ticker']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Quality", f"{detail_item['quality_score']:.1f}")
                m2.metric("Entry score", f"{detail_item['entry_score']:.1f}")
                m3.metric("RSI", f"{detail_item['rsi']:.1f}")
                m4.metric("R:R", f"1 : {detail_item['rr1']:.2f}")
                if st.button("Skrýt detail", key="hide_inline_detail"):
                    st.session_state.show_inline_detail = False
                    st.rerun()

# ============================================================
# TAB 2 – DETAIL
# ============================================================
with tab_detail:
    available = [r["ticker"] for r in st.session_state.results]
    if not available:
        st.info("Nejdříve spusťte sken; detail pak nabídne analyzované tickery.")
    else:
        default = st.session_state.selected_ticker
        index = available.index(default) if default in available else 0
        ticker = st.selectbox("Ticker", available, index=index, key="detail_ticker")
        item = next(r for r in st.session_state.results if r["ticker"] == ticker)

        # Lazy metadata
        name, sector = basic_info(ticker)
        pre_price, pre_change = premarket(ticker)

        st.subheader(f"{ticker} — {name}")
        st.caption(f"Sektor: {sector}")
        if pre_price is not None:
            st.caption(f"Pre-market: ${pre_price:.2f} ({pre_change:+.2f} %)")

        d = item["data"].tail(180)
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Candlestick(
            x=d.index, open=d["Open"], high=d["High"],
            low=d["Low"], close=d["Close"], name="Cena"
        ), secondary_y=False)

        for col, color in [("EMA20", "#2563eb"), ("EMA50", "#f59e0b"), ("EMA200", "#7c3aed")]:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[col], name=col,
                line={"color": color, "width": 1.2}
            ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_UPPER"], name="BB Upper",
            line={"color": "#94a3b8", "width": 0.8, "dash": "dot"}
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_LOWER"], name="BB Lower",
            line={"color": "#94a3b8", "width": 0.8, "dash": "dot"},
            fill="tonexty", fillcolor="rgba(148,163,184,0.08)"
        ), secondary_y=False)

        fig.add_hline(y=item["preferred_entry"], line_dash="dot",
                      annotation_text="Vstup", line_color="#16a34a")
        fig.add_hline(y=item["stop"], line_dash="dash",
                      annotation_text="Stop", line_color="#dc2626")
        fig.add_hline(y=item["target1"], line_dash="dash",
                      annotation_text="TP1", line_color="#0891b2")
        fig.add_hline(y=item["target2"], line_dash="dash",
                      annotation_text="TP2", line_color="#0e7490")

        fig.update_layout(
            height=600, xaxis_rangeslider_visible=False,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            legend={"orientation": "h", "y": 1.05},
        )
        st.plotly_chart(fig, use_container_width=True)

        # RSI + MACD
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.05,
                             row_heights=[0.5, 0.5])
        fig2.add_trace(go.Scatter(x=d.index, y=d["RSI14"], name="RSI",
                                  line={"color": "#7c3aed"}), row=1, col=1)
        fig2.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=1, col=1)
        fig2.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=1, col=1)
        fig2.update_yaxes(range=[0, 100], row=1, col=1)

        fig2.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], name="MACD Hist",
                              marker_color=np.where(d["MACD_HIST"] >= 0, "#16a34a", "#dc2626")),
                       row=2, col=1)
        fig2.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD",
                                  line={"color": "#2563eb"}), row=2, col=1)
        fig2.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"], name="Signal",
                                  line={"color": "#f59e0b"}), row=2, col=1)

        fig2.update_layout(height=400, margin={"l": 10, "r": 10, "t": 20, "b": 10},
                           showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

        # Pozicování
        st.markdown("### 💰 Pozicování")
        shares, value, max_risk, per_share = position_size(
            capital, risk_pct, item["preferred_entry"], item["stop"], max_position_pct
        )
        a, b, c, dcol = st.columns(4)
        a.metric("Počet akcií", shares)
        b.metric("Hodnota pozice", f"${value:,.2f}")
        c.metric("Riziko / akcie", f"${per_share:,.2f}")
        dcol.metric("Max. riziko", f"${shares * per_share:,.2f}")

        # Klíčové metriky
        st.markdown("### 📊 Klíčové metriky")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Quality", f"{item['quality_score']:.1f}")
        m2.metric("Entry", f"{item['entry_score']:.1f}")
        m3.metric("Trend", f"{item['trend_score']:.1f}")
        m4.metric("Momentum", f"{item['momentum_score']:.1f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("RS vs SPY (30d)", f"{item['rs_30d']:+.2f} %")
        m6.metric("RSI", f"{item['rsi']:.1f}")
        m7.metric("ATR %", f"{item['atr_pct']:.2f} %")
        m8.metric("Objem", f"{item['volume_ratio']:.2f}×")

        st.caption("Výpočet nezahrnuje poplatky, skluz, měnové riziko ani gapy. "
                   "Nejde o investiční doporučení.")

# ============================================================
# TAB 3 – HISTORIE SIGNÁLŮ
# ============================================================
with tab_history:
    st.subheader("Uložené signály")
    if sb is None:
        st.info("Historie není dostupná: není nakonfigurováno připojení Supabase.")
    else:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            limit = st.number_input("Max. řádků", 50, 2000, 500, 50)
        with col_f2:
            only_today = st.checkbox("Pouze dnešní", False)

        if st.button("🔄 Načíst historii", key="load_history"):
            try:
                q = sb.table("scanner_signals").select("*").order("signal_date", desc=True)
                if only_today:
                    today = datetime.now(timezone.utc).date().isoformat()
                    q = q.eq("signal_date", today)
                resp = q.limit(int(limit)).execute()
                hist = pd.DataFrame(resp.data or [])
                st.session_state.history_df = hist
            except Exception as e:
                log.exception("Načtení historie selhalo")
                st.error(f"Historii se nepodařilo načíst: {e}")

        hist = st.session_state.get("history_df")
        if hist is not None:
            if hist.empty:
                st.info("V databázi zatím nejsou uložené signály.")
            else:
                st.dataframe(hist, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Stáhnout historii CSV",
                    hist.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_historie.csv", "text/csv"
                )

# ============================================================
# TAB 4 – UČÍCÍ SE PŘEHLED
# ============================================================
with tab_learning:
    st.subheader("Vyhodnocení historických signálů")
    st.caption("Orientační zpětné vyhodnocení. Denní OHLC data nemusí určit "
               "pořadí zásahu stopu a cíle v rámci stejného dne.")

    if sb is None:
        st.info("Učící přehled vyžaduje připojení Supabase.")
    else:
        days = st.slider("Historie (dny)", 30, 365, 90, key="learning_days")

        if st.button("🧠 Vyhodnotit signály", key="run_learning"):
            with st.spinner("Vyhodnocuji historické signály…"):
                learning = load_learning_stats(sb, days=days)
            st.session_state.learning_df = learning

        learning = st.session_state.get("learning_df")
        if learning is not None:
            if learning.empty:
                st.info("Nejsou dostupná data k vyhodnocení.")
            else:
                # Statistiky
                counts = learning["outcome"].value_counts()
                tp1 = int(counts.get("TP1", 0))
                sl  = int(counts.get("SL", 0))
                amb = int(counts.get("AMBIGUOUS", 0))
                opn = int(counts.get("OPEN", 0))
                pend = int(counts.get("pending", 0))
                unk = int(counts.get("unknown", 0))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TP1", tp1)
                c2.metric("SL", sl)
                c3.metric("Ambiguous", amb)
                c4.metric("Open", opn)

                resolved = tp1 + sl
                if resolved > 0:
                    win_rate = tp1 / resolved * 100
                    st.metric("Win rate (TP1 vs SL)",
                              f"{win_rate:.1f} %",
                              help=f"Vyhodnoceno: {resolved} obchodů")

                # Průměrný forward return podle outcome
                valid = learning.dropna(subset=["forward_return_20d"])
                if not valid.empty:
                    st.markdown("#### Průměrný 20d forward return podle výsledku")
                    agg = valid.groupby("outcome")["forward_return_20d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg, use_container_width=True)

                # Průměrný return podle signálu
                if not valid.empty:
                    st.markdown("#### Průměrný 20d forward return podle typu signálu")
                    agg2 = valid.groupby("signal")["forward_return_20d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg2, use_container_width=True)

                st.markdown("#### Detailní data")
                st.dataframe(learning, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Stáhnout vyhodnocení CSV",
                    learning.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_learning.csv", "text/csv"
                )
