# ============================================================
# KLONDIKE SPOT SCANNER 4.4 — sjednocená a revidovaná verze
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
    page_title="Spot Scanner 4.4",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- CSS -----------------------------------
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f5f7fb; color: #111827; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: #ffffff !important; }
    [data-testid="stSidebar"] * { color: #111827 !important; }
    header[data-testid="stHeader"] { background: rgba(255,255,255,.96); }
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #dbeafe;
        border-radius: 18px;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        color: #111827;
        margin-bottom: 1rem;
        box-shadow: 0 4px 18px rgba(15,23,42,.06);
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #111827; }
    .hero p { margin: .4rem 0 0; color: #4b5563; }
    div[data-testid="stExpander"] { background: #ffffff; border-radius: 12px; }
    div[data-testid="stMetric"] { background: #ffffff; }
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
    "VMI","CCJ","AMBA","CRSP","NVDA","AAPL","GOOGL","MSFT","AMZN","META","TSLA","BRK-B",
    "WMT","MU","JPM","ORCL","XOM","AMD","JNJ","COST",
    "HD","UNH","BAC","IBM","INTC","BABA"
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
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                log.warning("download_history(%s): chybí sloupce %s", ticker,
                            sorted(required - set(df.columns)))
                return pd.DataFrame()
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            valid = ((df["Close"] > 0) & (df["Low"] > 0)
                     & (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1))
                     & (df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)))
            df = df.loc[valid].copy()
            return completed_daily_bars(df) if interval == "1d" else df
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
        pre = safe_float(pre, None)
        prev = safe_float(prev, None)
        if pre is not None and prev is not None and pre > 0 and prev > 0:
            return pre, (pre / prev - 1) * 100
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
    """Určí průraz, propad, Bollinger squeeze a neobvykle vysoký objem."""
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


def analyze_market_structure(d: pd.DataFrame, wing: int = 2) -> dict:
    """Klasifikuje swingovou strukturu a poslední potvrzený průraz."""
    empty = {
        "structure_trend": "NEDOSTATEK DAT",
        "structure_event": "Bez potvrzeného průrazu",
        "swing_high": np.nan,
        "swing_low": np.nan,
    }
    if len(d) < max(20, wing * 2 + 5):
        return empty

    highs = d["High"].to_numpy(dtype=float)
    lows = d["Low"].to_numpy(dtype=float)
    high_points, low_points = [], []
    # Swing se potvrdí až po wing následujících svíčkách.
    for i in range(wing, len(d) - wing):
        h_window = highs[i-wing:i+wing+1]
        l_window = lows[i-wing:i+wing+1]
        if highs[i] == np.max(h_window) and np.count_nonzero(h_window == highs[i]) == 1:
            high_points.append((i, highs[i]))
        if lows[i] == np.min(l_window) and np.count_nonzero(l_window == lows[i]) == 1:
            low_points.append((i, lows[i]))

    if len(high_points) < 2 or len(low_points) < 2:
        return empty

    previous_high, last_high = high_points[-2][1], high_points[-1][1]
    previous_low, last_low = low_points[-2][1], low_points[-1][1]
    higher_high = last_high > previous_high
    higher_low = last_low > previous_low
    lower_high = last_high < previous_high
    lower_low = last_low < previous_low

    if higher_high and higher_low:
        trend = "BÝČÍ (HH + HL)"
        trend_code = "BULLISH"
    elif lower_high and lower_low:
        trend = "MEDVĚDÍ (LH + LL)"
        trend_code = "BEARISH"
    else:
        trend = "SMÍŠENÁ / BOČNÍ"
        trend_code = "MIXED"

    close = d["Close"]
    last_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    swing_high = float(last_high)
    swing_low = float(last_low)
    broke_up = last_close > swing_high and previous_close <= swing_high
    broke_down = last_close < swing_low and previous_close >= swing_low

    event = "Bez potvrzeného průrazu"
    if broke_up:
        event = "CHoCH nahoru" if trend_code == "BEARISH" else "BOS nahoru"
    elif broke_down:
        event = "CHoCH dolů" if trend_code == "BULLISH" else "BOS dolů"

    return {
        "structure_trend": trend,
        "structure_event": event,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_news_context(ticker: str) -> dict:
    """Načte několik titulků a spočítá pouze orientační slovníkový sentiment."""
    positive_terms = (
        "beat", "beats", "upgrade", "upgraded", "buy rating", "growth", "record",
        "surge", "rally", "profit", "strong", "outperform", "raises guidance",
        "revenue up", "bullish", "wins contract", "approves", "approval",
    )
    negative_terms = (
        "miss", "misses", "downgrade", "downgraded", "lawsuit", "fine", "probe",
        "decline", "plunge", "crash", "loss", "weak", "underperform", "cuts guidance",
        "revenue down", "bearish", "layoff", "recall", "investigation", "fraud",
    )
    articles = []
    try:
        obj = yf.Ticker(ticker)
        try:
            raw_news = obj.get_news(count=5, tab="news")
        except Exception:
            raw_news = getattr(obj, "news", [])

        for raw in raw_news or []:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
            title = str(content.get("title") or raw.get("title") or "").strip()
            if not title:
                continue
            summary = str(content.get("summary") or content.get("description") or "").strip()
            provider = content.get("provider") or raw.get("publisher") or {}
            source = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
            published = content.get("pubDate") or content.get("displayTime") or raw.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            elif published:
                published = str(published)
            else:
                published = "Datum neuvedeno"
            url_data = content.get("canonicalUrl") or content.get("clickThroughUrl") or raw.get("link") or ""
            url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
            text = f"{title} {summary}".lower()
            score = sum(text.count(term) for term in positive_terms) - sum(text.count(term) for term in negative_terms)
            articles.append({
                "title": title[:300], "summary": summary[:700], "source": source,
                "published": published, "url": url, "score": score,
            })
            if len(articles) >= 5:
                break

        total_score = sum(a["score"] for a in articles)
        if total_score > 0:
            sentiment = "Převážně pozitivní titulky"
        elif total_score < 0:
            sentiment = "Převážně negativní titulky"
        else:
            sentiment = "Smíšené nebo neutrální titulky"
        return {"articles": articles, "sentiment": sentiment, "score": total_score}
    except Exception as e:
        log.warning("Načtení zpráv pro %s selhalo: %s", ticker, e)
        return {"articles": [], "sentiment": "Zprávy nejsou dostupné", "score": 0}


# ---------------------- SUPPORT / RESISTANCE ------------------
# Nastavení strategie 4.4: výchozí hypotézy, nikoli optimalizované parametry.
BUY_SIGNAL = "NÁKUPNÍ ZÓNA · 4.4"
STRATEGY = {
    "lookback": 90, "wing": 2, "min_touches": 2,
    "touch_spacing": 5, "cluster_atr": 0.5,
    "max_support_age": 40, "zone_below_atr": 0.15,
    "zone_above_atr": 0.5, "max_distance_pct": 1.5,
    "stop_buffer_atr": 0.3, "min_stop_atr": 0.8,
    "max_stop_atr": 2.0, "min_rr": 1.5,
    "target1_atr": 2.0, "target2_atr": 3.0,
    "min_quality": 65, "max_rsi": 68,
    "holding_days": 5,
}


def completed_daily_bars(df, now=None):
    """Konzervativně vynechá celý aktuální den v timezone burzovního indexu.

    Dnešní svíčka se použije až další kalendářní den, i po konci seance.
    U indexu bez timezone vynechá dnešek podle UTC.
    """
    if df.empty:
        return df.copy()
    stamp = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    idx = pd.DatetimeIndex(df.index)
    today = stamp.tz_convert(idx.tz or "UTC").date()
    return df.loc[idx.date < today].sort_index().copy()


def confirmed_pivots(d, column, wing=2):
    """Pivot musí být potvrzen už PŘED poslední (signální) svíčkou."""
    values = d[column].to_numpy(dtype=float)
    points = []
    for i in range(wing, len(values) - wing - 1):
        window = values[i-wing:i+wing+1]
        extremum = np.min(window) if column == "Low" else np.max(window)
        if values[i] == extremum and np.count_nonzero(window == extremum) == 1:
            points.append((i, float(values[i])))
    return points


def structural_levels(d):
    """Opakovaně testovaný support z cenových minim; bez EMA a náhradních úrovní."""
    recent = d.tail(STRATEGY["lookback"]).reset_index(drop=True)
    atr = safe_float(d["ATR14"].iloc[-1])
    price = safe_float(d["Close"].iloc[-1])
    empty = {"support": np.nan, "support_floor": np.nan,
             "support_touches": 0, "support_age": np.nan, "resistances": []}
    if not np.isfinite(atr) or atr <= 0 or len(recent) < 15:
        return empty
    lows = confirmed_pivots(recent, "Low", STRATEGY["wing"])
    candidates = []
    # Shluk má omezenou CELKOVOU šířku, nikoli řetězení vzdálených minim.
    for _, anchor in lows:
        cluster = [(i, v) for i, v in lows
                   if anchor <= v <= anchor + STRATEGY["cluster_atr"] * atr]
        separated = []
        for point in reversed(cluster):
            if not separated or separated[-1][0] - point[0] >= STRATEGY["touch_spacing"]:
                separated.append(point)
        if len(separated) < STRATEGY["min_touches"]:
            continue
        latest = max(i for i, _ in separated)
        age = len(recent) - 1 - latest
        if age > STRATEGY["max_support_age"]:
            continue
        level = float(np.median([v for _, v in separated]))
        floor = min(v for _, v in separated)
        # Proražený support není automaticky znovu platný po návratu ceny.
        after = recent["Close"].iloc[min(i for i, _ in separated)+1:]
        if (after < floor - 0.2 * atr).any() or level > price:
            continue
        candidates.append((level, floor, len(separated), age))
    if not candidates:
        return empty
    support, floor, touches, age = max(candidates, key=lambda a: a[0])
    highs = confirmed_pivots(recent, "High", STRATEGY["wing"])
    resistances = []
    for i, level in highs:
        if level <= support:
            continue
        # Rezistence překonaná uzavírací cenou již není překážkou nad supportem.
        if (recent["Close"].iloc[i+1:] > level + 0.2 * atr).any():
            continue
        resistances.append(level)
    return {"support": support, "support_floor": floor,
            "support_touches": touches, "support_age": age,
            "resistances": sorted(set(resistances))}


def support_resistance(d):
    levels = structural_levels(d)
    return levels["support"], next(iter(levels["resistances"]), np.nan)


def calculate_entry_engine(d: pd.DataFrame, scores: dict,
                           regime: str = "UNKNOWN", data_ok: bool = True) -> dict:
    x = d.iloc[-1]
    price, atr = float(x["Close"]), safe_float(x["ATR14"])
    levels = structural_levels(d)
    support = levels["support"]
    resistance = next(iter(levels["resistances"]), np.nan)
    valid = bool(np.isfinite(support) and np.isfinite(atr) and atr > 0)
    zone_low = support - STRATEGY["zone_below_atr"] * atr if valid else np.nan
    zone_high = (support + min(STRATEGY["zone_above_atr"] * atr,
                              support * STRATEGY["max_distance_pct"] / 100)
                 if valid else np.nan)
    in_zone = bool(valid and support <= price <= zone_high)
    # Close musí být nad supportem; průnik knotem pod něj může být odmítnutí ceny.
    span = float(x["High"] - x["Low"])
    bounce = bool(valid and float(x["Low"]) <= zone_high
                  and float(x["High"]) >= zone_low
                  and float(x["Low"]) >= levels["support_floor"] - 0.5 * atr
                  and price > float(x["Open"]) and price >= float(d["Close"].iloc[-2])
                  and span > 0 and (price - float(x["Low"])) / span >= 0.6)
    # Pro aktuální signál vždy R:R ze signálního close; plán čekání z horní hrany zóny.
    entry = price if in_zone else zone_high
    stop = (min(levels["support_floor"] - STRATEGY["stop_buffer_atr"] * atr,
                entry - STRATEGY["min_stop_atr"] * atr) if valid else np.nan)
    risk = entry - stop
    target1 = (min(resistance - 0.1 * atr, entry + STRATEGY["target1_atr"] * atr)
               if valid and np.isfinite(resistance) else np.nan)
    target2 = np.nan
    if valid and len(levels["resistances"]) > 1:
        # Druhý cíl je jen scénář po překonání první rezistence.
        next_resistance = next((r for r in levels["resistances"]
                                if r > resistance + 0.5 * atr), np.nan)
        if np.isfinite(next_resistance):
            candidate = min(next_resistance - 0.1 * atr, entry + STRATEGY["target2_atr"] * atr)
            if candidate > target1:
                target2 = candidate
    rr1 = (target1 - entry) / risk if risk > 0 else np.nan
    rr2 = (target2 - entry) / risk if risk > 0 else np.nan
    structure = analyze_market_structure(d)
    trend_ok = bool(price > x["EMA50"] and x["EMA20"] > x["EMA50"]
                    and x["EMA50"] >= d["EMA50"].iloc[-6]
                    and not structure["structure_trend"].startswith("MEDVĚDÍ"))
    checks = {
        "Aktuální uzavřená data": bool(data_ok),
        "Potvrzený opakovaně testovaný support": valid,
        "Cena u supportu": in_zone,
        "Potvrzení odrazu": bounce,
        "Rostoucí trend": trend_ok,
        "SPY není medvědí a je dostupný": regime in {"BULLISH", "NEUTRAL"},
        "Quality alespoň 65": scores["quality_score"] >= STRATEGY["min_quality"],
        "RSI pod limitem": safe_float(x["RSI14"], 100) < STRATEGY["max_rsi"],
        "Strukturální cíl a R:R alespoň 1,5": bool(np.isfinite(rr1) and rr1 >= STRATEGY["min_rr"]),
        "Stop v povoleném rozsahu": bool(valid and stop > 0 and
                0 < risk <= STRATEGY["max_stop_atr"] * atr),
    }
    eligible = all(checks.values())
    if eligible:
        signal = BUY_SIGNAL
    elif not valid:
        signal = "BEZ POTVRZENÉHO SUPPORTU"
    elif price > zone_high:
        signal = "ČEKAT NA KOREKCI"
    elif not bounce:
        signal = "ČEKAT NA ODRAZ"
    else:
        signal = "NEVSTUPOVAT"
    reason = ("Všechny vstupní podmínky splněny na posledním uzavřeném dni."
              if eligible else "Nesplněno: " + "; ".join(k for k, v in checks.items() if not v))
    return {
        "support": support, "resistance": resistance,
        "support_touches": levels["support_touches"], "support_age": levels["support_age"],
        "aggressive_entry": zone_high, "preferred_entry": entry,
        "conservative_entry": support, "zone_low": zone_low, "zone_high": zone_high,
        "stop": stop, "target1": target1, "target2": target2, "risk_per_share": risk,
        "rr1": rr1, "rr2": rr2,
        "entry_score": round(100 * sum(checks.values()) / len(checks), 1),
        "distance_to_zone": (price - zone_high) / price * 100 if valid and price > zone_high else 0.0,
        "support_distance_atr": (price - support) / atr if valid else np.nan,
        "support_distance_pct": (price / support - 1) * 100 if valid else np.nan,
        "signal": signal, "buy_allowed": eligible, "entry_checks": checks,
        "entry_reason": reason, "trend_ok": trend_ok,
    }


def assess_trade_action(item: dict) -> dict:
    """Jediný zdroj rozhodnutí. Scanner nezná otevřené pozice uživatele."""
    return {
        "trend_label": "BÝČÍ" if item["trend_ok"] else "NEPOTVRZENÝ / SLABÝ",
        "action_label": "NAKUPOVAT" if item["buy_allowed"] else "ČEKAT / NEVSTUPOVAT",
        "action_reason": item["entry_reason"],
    }


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
        "signal_date":      pd.Timestamp(r["data"].index[-1]).date().isoformat(),
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
        payloads = [{k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                     for k, v in payload.items()} for payload in payloads]
        sb.table("scanner_signals").upsert(
            payloads, on_conflict="ticker,signal_date"
        ).execute()
        log.info(f"Uloženo {len(payloads)} signálů.")
        return len(payloads)
    except Exception as e:
        log.error(f"save_signals_bulk selhalo: {e}", exc_info=True)
        return 0


def load_learning_stats(sb, days: int = 90):
    """Orientační simulace pouze signálů 4.4, vstup na příštím open, max. 5 dnů.

    Není to walk-forward backtest ani učení parametrů. Gap mimo zónu = bez vstupu.
    Denní OHLC nezná pořadí SL/TP; takový obchod se nezařadí mezi výhry/prohry.
    """
    if sb is None:
        return pd.DataFrame()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        response = sb.table("scanner_signals").select("*").gte(
            "signal_date", cutoff).eq("signal", BUY_SIGNAL).execute()
        df = pd.DataFrame(response.data or [])
        if df.empty:
            return df
        prices = {t: download_history(t) for t in df["ticker"].unique()}
        returns, outcomes = [], []
        for _, row in df.iterrows():
            outcome, result = "unknown", np.nan
            hist = prices[row["ticker"]]
            if hist.empty:
                returns.append(result); outcomes.append(outcome); continue
            day = pd.Timestamp(row["signal_date"]).date()
            matches = np.flatnonzero(pd.Index(hist.index.date) == day)
            if not len(matches):
                returns.append(result); outcomes.append(outcome); continue
            i = int(matches[0])
            # auto_adjust může po dividendě/splitu přepočítat historii.
            saved_close = safe_float(row.get("price"))
            ratio = float(hist["Close"].iloc[i]) / saved_close if saved_close > 0 else np.nan
            if not np.isfinite(ratio) or abs(ratio - 1) > 0.002:
                returns.append(result); outcomes.append("DATA_CHANGED"); continue
            window = hist.iloc[i+1:i+1+STRATEGY["holding_days"]]
            if window.empty:
                returns.append(result); outcomes.append("pending"); continue
            stop, tp = safe_float(row.get("stop_price")), safe_float(row.get("target1"))
            entry = float(window["Open"].iloc[0])
            support = safe_float(row.get("support"))
            upper = safe_float(row.get("zone_high"))
            if not (np.isfinite(stop) and np.isfinite(tp) and stop < entry < tp
                    and support <= entry <= upper
                    and (tp-entry)/(entry-stop) >= STRATEGY["min_rr"]):
                returns.append(result); outcomes.append("NOT_FILLED"); continue
            outcome = "OPEN"
            for _, bar in window.iterrows():
                opening = float(bar["Open"])
                if opening <= stop:
                    outcome, result = "SL", (opening / entry - 1) * 100
                    break
                if opening >= tp:
                    outcome, result = "TP1", (tp / entry - 1) * 100
                    break
                hit_sl, hit_tp = bar["Low"] <= stop, bar["High"] >= tp
                if hit_sl and hit_tp:
                    outcome = "AMBIGUOUS"
                    break
                if hit_sl or hit_tp:
                    outcome = "SL" if hit_sl else "TP1"
                    result = ((stop if hit_sl else tp) / entry - 1) * 100
                    break
            if outcome == "OPEN":
                if len(window) < STRATEGY["holding_days"]:
                    outcome = "pending"
                else:
                    outcome = "TIME_EXIT"
                    result = (float(window["Close"].iloc[-1]) / entry - 1) * 100
            returns.append(result); outcomes.append(outcome)
        df["forward_return_5d"] = returns
        df["outcome"] = outcomes
        return df
    except Exception as e:
        log.error("load_learning_stats selhalo: %s", e, exc_info=True)
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
    regime, _ = market_regime(spy)
    last_day = pd.Timestamp(d.index[-1]).date()
    today = datetime.now(timezone.utc).date()
    data_ok = 0 <= (today - last_day).days <= 4
    if spy.empty or pd.Timestamp(spy.index[-1]).date() != last_day:
        data_ok = False
    entry = calculate_entry_engine(d, scores, regime, data_ok)
    patterns = detect_patterns(d)
    structure = analyze_market_structure(d)

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
        **structure,
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
    <h3>📈 Spot Scanner 4.4</h3>
    <p>Technická analýza • Struktura trhu • Zprávy • Nákupní zóny • Risk management</p>
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
    st.caption("Zdroj dat: Yahoo Finance; pouze předchozí dokončené denní svíčky.")
    st.caption("Support: ≥2 oddělené testy. Vstup nejvýše 0,5 ATR a 1,5 % nad ním. "
               "R:R ≥1,5; horizont 5 obchodních dnů. Parametry v STRATEGY.")
    st.caption("Analytická pomůcka – ne automatický obchodní systém.")

# ---------------------- TABS --------------------------------
tab_scan, tab_history, tab_learning = st.tabs([
    "🔎 Scanner", "🗄️ Historie signálů", "🧠 Učící se přehled"
])

if "results" not in st.session_state:
    st.session_state.results = []
elif st.session_state.results and "buy_allowed" not in st.session_state.results[0]:
    st.session_state.results = []
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
        run = st.button("🚀 SPUSTIT SKEN PŘEDNASTAVENÝCH", type="primary", use_container_width=True)

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
            if only_buy_zone and r["signal"] != BUY_SIGNAL: continue
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
        buy_count   = sum(r["signal"] == BUY_SIGNAL for r in results)
        avg_quality = np.mean([r["quality_score"] for r in results])
        avg_entry   = np.mean([r["entry_score"]   for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzovaných",   len(results))
        c2.metric("NÁKUPNÍ ZÓNA",    buy_count)
        c3.metric("Průměr Quality",  f"{avg_quality:.1f}")
        c4.metric("Průměr Entry",    f"{avg_entry:.1f}")

        rows = []
        for r in results:
            assessment = assess_trade_action(r)
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
                "Support": r["support"],
                "Od supportu (ATR)": round(r["support_distance_atr"], 2),
                "Testy supportu": r["support_touches"],
                "Důvod": r["entry_reason"],
                "Datum dat": str(r["data"].index[-1].date()),
                "RS 30d":   r["rs_30d"],
                "RSI":      round(r["rsi"], 1),
                "Objem":    round(r["volume_ratio"], 2),
                "Průraz":   r["breakout"],
                "Squeeze":  r["squeeze"],
                "Objemový spike": r["volume_spike"],
                "Struktura trhu": r["structure_trend"],
                "Průlom struktury": r["structure_event"],
                "Trend": assessment["trend_label"],
                "Orientační akce": assessment["action_label"],
            })
        df_show = pd.DataFrame(rows)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Stáhnout CSV",
            df_show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"klondike_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.markdown("### 📊 Podrobnosti jednotlivých instrumentů")
        st.caption("Rozbalte instrument. Každý panel obsahuje jen jeho vlastní metriky, signál a grafy.")

        for item in results:
            ticker = item["ticker"]
            assessment = assess_trade_action(item)
            with st.expander(
                f"{ticker} · {assessment['action_label']} · trend {assessment['trend_label']} · "
                f"{item['signal']}",
                expanded=False,
            ):
                st.subheader(f"{ticker} · {item['signal']}")
                action_message = (
                    f"**Orientační akce: {assessment['action_label']}** — "
                    f"{assessment['action_reason']}"
                )
                if assessment["action_label"] in {"NAKUPOVAT", "POMALU DOKUPOVAT"}:
                    st.success(action_message)
                elif assessment["action_label"] == "ČEKAT / NEVSTUPOVAT":
                    st.info(action_message)
                elif assessment["action_label"] == "POMALU ODPRODÁVAT":
                    st.warning(action_message)
                else:
                    st.error(action_message)
                st.info(f"**Trend: {assessment['trend_label']}** · {item['structure_event']}")
                st.caption("Automatické technické vyhodnocení; zohledněte vlastní strategii a riziko.")
                st.caption(
                    f"Cena ${item['price']:.2f} · Vstupní zóna ${item['zone_low']:.2f}–"
                    f"${item['zone_high']:.2f} · Preferovaný vstup ${item['preferred_entry']:.2f}"
                )

                st.caption(f"Signální den: {item['data'].index[-1].date()}. "
                           "Dnešní denní svíčka je vynechána. Cena není živá kotace.")
                st.write(f"Support: {item['support_touches']} potvrzených testů; "
                         f"vzdálenost {item['support_distance_atr']:.2f} ATR "
                         f"({item['support_distance_pct']:.2f} %).")
                st.dataframe(pd.DataFrame([
                    {"Podmínka": label, "Splněno": ok}
                    for label, ok in item["entry_checks"].items()
                ]), hide_index=True, use_container_width=True)
                st.markdown("#### 🧭 Struktura trhu")
                st.write(f"**{item['structure_trend']}** · {item['structure_event']}")
                st.caption(
                    f"Poslední potvrzené swingové úrovně: maximum ${item['swing_high']:.2f} · "
                    f"minimum ${item['swing_low']:.2f}. Swingy se potvrzují až po dalších svíčkách."
                )

                # Síťové doplňky se načítají jen po kliknutí v konkrétní ticker záložce.
                if st.button("Načíst název, pre-market a zprávy", key=f"load_context_{ticker}"):
                    st.session_state[f"context_loaded_{ticker}"] = True
                if st.session_state.get(f"context_loaded_{ticker}", False):
                    name, sector = basic_info(ticker)
                    pre_price, pre_change = premarket(ticker)
                    st.markdown(f"**{name}** · Sektor: {sector}")
                    if pre_price is not None:
                        st.caption(f"Pre-market: ${pre_price:.2f} ({pre_change:+.2f} %)")
                    news = get_news_context(ticker)
                    st.markdown("#### 📰 Kontext zpráv")
                    st.caption(
                        f"{news['sentiment']} (orientační skóre titulků: {news['score']:+d}). "
                        "Jde o jednoduché klíčové fráze, ne o porozumění významu článku."
                    )
                    if news["articles"]:
                        for article_index, article in enumerate(news["articles"]):
                            st.write(f"**{article['title']}**")
                            meta = " · ".join(
                                part for part in [article["source"], article["published"]] if part
                            )
                            if meta:
                                st.caption(meta)
                            if article["summary"]:
                                st.write(article["summary"])
                            if article["url"].startswith(("https://", "http://")):
                                st.link_button(
                                    "Otevřít článek", article["url"],
                                    key=f"news_{ticker}_{article_index}"
                                )
                    else:
                        st.info("Pro tento ticker nejsou dostupné zprávy.")

                d = item["data"].tail(180)
                st.markdown("#### 📊 Klíčové metriky")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Quality", f"{item['quality_score']:.1f}/100")
                k2.metric("Entry", f"{item['entry_score']:.1f}/100")
                k3.metric("Trend", f"{item['trend_score']:.1f}/100")
                k4.metric("Momentum", f"{item['momentum_score']:.1f}/100")
                k5, k6, k7, k8 = st.columns(4)
                k5.metric("RS vs SPY (30d)", f"{item['rs_30d']:+.2f} %")
                k6.metric("RSI (14)", f"{item['rsi']:.1f}")
                k7.metric("ATR", f"${item['atr']:.2f} ({item['atr_pct']:.2f} %)")
                k8.metric("Objem vs průměr", f"{item['volume_ratio']:.2f}×")

                shares, value, _, per_share = position_size(
                    capital, risk_pct, item["preferred_entry"], item["stop"], max_position_pct
                )
                if not item["buy_allowed"]:
                    shares, value, per_share = 0, 0.0, 0.0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Počet akcií", shares)
                p2.metric("Hodnota pozice", f"${value:,.2f}")
                p3.metric("Riziko na akcii", f"${per_share:,.2f}")
                p4.metric("Riziko pozice", f"${shares * per_share:,.2f}")
                st.caption(
                    "Výpočty nezahrnují poplatky, skluz, měnové riziko ani cenové gapy. "
                    "Nejde o investiční doporučení."
                )

                st.markdown("#### 📈 Cena, klouzavé průměry a obchodní úrovně")
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Candlestick(
                    x=d.index, open=d["Open"], high=d["High"],
                    low=d["Low"], close=d["Close"], name="Cena"
                ), secondary_y=False)
                for col, color in [("EMA20", "#2563eb"), ("EMA50", "#f59e0b"),
                                   ("EMA200", "#7c3aed")]:
                    fig.add_trace(go.Scatter(
                        x=d.index, y=d[col], name=col,
                        line={"color": color, "width": 1.2}
                    ), secondary_y=False)
                for col, label in [("BB_UPPER", "BB Upper"), ("BB_LOWER", "BB Lower")]:
                    fig.add_trace(go.Scatter(
                        x=d.index, y=d[col], name=label,
                        line={"color": "#94a3b8", "width": 0.8, "dash": "dot"}
                    ), secondary_y=False)
                for value, label, dash, color in [
                    (item["preferred_entry"], "Vstup", "dot", "#16a34a"),
                    (item["stop"], "Stop", "dash", "#dc2626"),
                    (item["target1"], "TP1", "dash", "#0891b2"),
                    (item["target2"], "TP2", "dash", "#0e7490"),
                    (item["support"], "Podpora", "dot", "#64748b"),
                    (item["resistance"], "Rezistence", "dot", "#a16207"),
                ]:
                    if np.isfinite(value):
                        fig.add_hline(y=value, line_dash=dash,
                                      annotation_text=label, line_color=color)
                if np.isfinite(item["zone_low"]):
                    fig.add_hrect(y0=item["zone_low"], y1=item["zone_high"],
                                  fillcolor="green", opacity=0.12, line_width=0,
                                  annotation_text="Oblast supportu")
                fig.update_layout(
                    height=520, xaxis_rangeslider_visible=False,
                    margin={"l": 10, "r": 10, "t": 30, "b": 10},
                    legend={"orientation": "h", "y": 1.05},
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{ticker}")

                st.markdown("#### 📉 RSI a MACD")
                fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05, row_heights=[0.5, 0.5])
                fig2.add_trace(go.Scatter(
                    x=d.index, y=d["RSI14"], name="RSI",
                    line={"color": "#7c3aed"}
                ), row=1, col=1)
                fig2.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=1, col=1)
                fig2.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=1, col=1)
                fig2.update_yaxes(range=[0, 100], row=1, col=1)
                fig2.add_trace(go.Bar(
                    x=d.index, y=d["MACD_HIST"], name="MACD histogram",
                    marker_color=np.where(d["MACD_HIST"] >= 0, "#16a34a", "#dc2626")
                ), row=2, col=1)
                fig2.add_trace(go.Scatter(
                    x=d.index, y=d["MACD"], name="MACD",
                    line={"color": "#2563eb"}
                ), row=2, col=1)
                fig2.add_trace(go.Scatter(
                    x=d.index, y=d["MACD_SIGNAL"], name="Signál",
                    line={"color": "#f59e0b"}
                ), row=2, col=1)
                fig2.update_layout(
                    height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    showlegend=True, template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True, key=f"indicators_chart_{ticker}")



# ============================================================
# TAB 2 – HISTORIE SIGNÁLŮ
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
# TAB 3 – UČÍCÍ SE PŘEHLED
# ============================================================
with tab_learning:
    st.subheader("Vyhodnocení historických signálů")
    st.caption("Pouze nákupní signály 4.4. Model: příští open v zóně, kontrola R:R, výstup do 5 dnů. "
               "Jde o simulaci bez nákladů, nikoli validaci ziskovosti. Denní OHLC data nemusí určit "
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
                opn = int(counts.get("TIME_EXIT", 0))
                pend = int(counts.get("pending", 0))
                not_filled = int(counts.get("NOT_FILLED", 0))

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("TP1", tp1)
                c2.metric("SL", sl)
                c3.metric("Ambiguous", amb)
                c4.metric("Výstup 5. den", opn)
                c5.metric("Vstup nevyplněn", not_filled)
                if pend:
                    st.caption(f"Čeká na dokončení 5 obchodních dnů: {pend} signálů.")

                resolved = tp1 + sl
                if resolved > 0:
                    win_rate = tp1 / resolved * 100
                    st.metric("TP1 / (TP1 + SL); bez časových a nejasných výstupů",
                              f"{win_rate:.1f} %",
                              help=f"Vyhodnoceno: {resolved} obchodů")

                # Průměrný forward return podle outcome
                valid = learning.dropna(subset=["forward_return_5d"])
                if not valid.empty:
                    st.markdown("#### Průměrný výnos (výstup nejpozději 5. den) podle výsledku")
                    agg = valid.groupby("outcome")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg, use_container_width=True)

                # Průměrný return podle signálu
                if not valid.empty:
                    st.markdown("#### Průměrný výnos (výstup nejpozději 5. den) podle typu signálu")
                    agg2 = valid.groupby("signal")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg2, use_container_width=True)

                st.markdown("#### Detailní data")
                st.dataframe(learning, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Stáhnout vyhodnocení CSV",
                    learning.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_learning.csv", "text/csv"
                )
