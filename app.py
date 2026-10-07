# ============================================================
# KLONDIKE SPOT SCANNER 4.8 — sjednocená a revidovaná verze
# Spot swing decision-support scanner s feedback loopem
# 4.8: přepínač Akcie / Zemědělské komodity, počasí a volitelný horizont plodin.
# Zachováno: opční rizikový filtr; dividendy/výsledky; správa long pozice.
# Instalace: pip install streamlit yfinance numpy pandas plotly supabase tzdata
# Spuštění: streamlit run app.py
# Supabase: původní schéma scanner_signals se nemění. Detail doplňků exportujte do CSV.
# Žádná data nepředstíráme: příští částka dividendy není odvozena z dividendRate.
# OI neodhaluje identitu velkých hráčů ani jejich nákupy/prodeje.
# ============================================================

import re
import json
import urllib.request
import urllib.parse
from calendar import monthrange
import time
import logging
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
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
    page_title="Spot Scanner 4.8",
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

# Londýnské USD kotace, ověřené podle seznamu emitenta WisdomTree.
# Jde o ETC navázaná na futures, nikoli o přímou cenu tuny plodiny.
CROP_PRODUCTS = {
    "WEAT.L": {"crop": "wheat", "name": "Pšenice", "product": "WisdomTree Wheat", "currency": "USD",
               "source": "https://www.wisdomtree.com/ie/products/commodities/wisdomtree-wheat"},
    "CORN.L": {"crop": "corn", "name": "Kukuřice", "product": "WisdomTree Corn", "currency": "USD",
               "source": "https://www.wisdomtree.com/gb/products/commodities/wisdomtree-corn"},
    "SOYB.L": {"crop": "soy", "name": "Sója", "product": "WisdomTree Soybeans", "currency": "USD",
               "source": "https://www.wisdomtree.com/gb/products/commodities/wisdomtree-soybeans"},
}
CROP_NAMES = {"wheat": "Pšenice", "corn": "Kukuřice", "soy": "Sója"}

# Hrubé měsíční kalendáře citlivosti a MODELOVÉ váhy sledovaných bodů.
# Váhy nejsou podíly na světové produkci. Nejde o kompletní mapu pěstitelských ploch.
CROP_CALENDARS = {
    "corn_us": {4: .3, 5: .4, 6: .7, 7: 1.0, 8: .9, 9: .5, 10: .2},
    "corn_br": {1: .8, 2: .6, 3: .7, 4: 1.0, 5: .9, 6: .6, 7: .3, 9: .3, 10: .4, 11: .7, 12: 1.0},
    "corn_ar": {1: 1.0, 2: .9, 3: .7, 4: .5, 5: .3, 9: .3, 10: .5, 11: .7, 12: 1.0},
    "soy_us": {4: .3, 5: .4, 6: .6, 7: .8, 8: 1.0, 9: .7, 10: .3},
    "soy_br": {1: 1.0, 2: .8, 3: .5, 4: .3, 9: .3, 10: .4, 11: .7, 12: .9},
    "soy_ar": {1: .9, 2: 1.0, 3: .8, 4: .5, 5: .3, 10: .3, 11: .4, 12: .7},
    "wheat_winter": {3: .4, 4: .7, 5: 1.0, 6: .9, 7: .5, 9: .3, 10: .4, 11: .2},
    "wheat_au": {5: .3, 6: .4, 7: .5, 8: .7, 9: 1.0, 10: .9, 11: .5, 12: .3},
}
CROP_REGIONS = {
    "corn": [
        {"name": "USA — Iowa", "lat": 42.03, "lon": -93.62, "weight": .25, "calendar": "corn_us"},
        {"name": "USA — Illinois", "lat": 40.11, "lon": -88.23, "weight": .25, "calendar": "corn_us"},
        {"name": "Brazílie — Mato Grosso", "lat": -12.54, "lon": -55.72, "weight": .30, "calendar": "corn_br"},
        {"name": "Argentina — Córdoba", "lat": -32.41, "lon": -63.24, "weight": .20, "calendar": "corn_ar"},
    ],
    "soy": [
        {"name": "USA — Iowa", "lat": 42.03, "lon": -93.62, "weight": .20, "calendar": "soy_us"},
        {"name": "USA — Illinois", "lat": 40.11, "lon": -88.23, "weight": .20, "calendar": "soy_us"},
        {"name": "Brazílie — Mato Grosso", "lat": -12.54, "lon": -55.72, "weight": .40, "calendar": "soy_br"},
        {"name": "Argentina — Córdoba", "lat": -32.41, "lon": -63.24, "weight": .20, "calendar": "soy_ar"},
    ],
    "wheat": [
        {"name": "USA — Illinois", "lat": 38.53, "lon": -89.12, "weight": .30, "calendar": "wheat_winter"},
        {"name": "Francie — Centre", "lat": 48.45, "lon": 1.49, "weight": .25, "calendar": "wheat_winter"},
        {"name": "Ukrajina — střed", "lat": 48.51, "lon": 32.26, "weight": .25, "calendar": "wheat_winter"},
        {"name": "Austrálie — New South Wales", "lat": -32.25, "lon": 148.61, "weight": .20, "calendar": "wheat_au"},
    ],
}
WEATHER_RULES = {
    "forecast_days": 14, "min_members": 15, "min_coverage": .80,
    "max_age_hours": 6, "max_reference_age_hours": 72,
    "min_common_days": 5, "revision_block": -.20,
    "heat_c": 35.0, "frost_c": -2.0, "dry_day_mm": 1.0, "heavy_rain_mm": 40.0,
}

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
# Nastavení strategie 4.8: výchozí hypotézy, nikoli optimalizované parametry.
BUY_SIGNAL = "NÁKUPNÍ ZÓNA · 4.8"
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
    "min_dollar_volume": 10_000_000, "max_gap_atr": 1.0,
    "max_signal_range_atr": 2.5,
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


def execution_bounds(support, zone_high, stop, target, atr):
    """Povolený interval při NEMĚNNÉM stopu a cíli, bez poplatků/slippage."""
    if not all(np.isfinite(v) for v in (support, zone_high, stop, target, atr)) or atr <= 0:
        return np.nan, np.nan
    lower = max(support, stop + STRATEGY["min_stop_atr"] * atr)
    upper = min(zone_high,
                (target + STRATEGY["min_rr"] * stop) / (1 + STRATEGY["min_rr"]),
                stop + STRATEGY["max_stop_atr"] * atr)
    if stop <= 0 or target <= lower or upper < lower - 1e-9:
        return np.nan, np.nan
    return lower, upper


def trader_filters(d, result, mode="stocks"):
    """Likvidita a výjimečný pohyb jsou blokace; konfluence/objem jen kontext."""
    r = dict(result)
    x = d.iloc[-1]
    atr = safe_float(x["ATR14"])
    prior_atr = safe_float(d["ATR14"].iloc[-2])
    previous_close = safe_float(d["Close"].iloc[-2])
    prior = d.iloc[-21:-1]
    volume = pd.to_numeric(d.get("Volume", pd.Series(index=d.index, dtype=float)), errors="coerce")
    prior_volume = volume.iloc[-21:-1]
    valid_volume = prior_volume.where(prior_volume > 0).replace([np.inf, -np.inf], np.nan)
    turnover = (prior["Close"] * valid_volume).dropna()
    dollar_volume = safe_float(turnover.median()) if not turnover.empty else np.nan
    enough_volume = len(valid_volume) == 20 and valid_volume.notna().all()
    volume_baseline = safe_float(valid_volume.mean())
    volume_ratio = safe_float(volume.iloc[-1]) / volume_baseline if volume_baseline > 0 else np.nan
    gap_atr = (abs(float(x["Open"]) - previous_close) / prior_atr
               if prior_atr > 0 else np.nan)
    true_range = max(float(x["High"] - x["Low"]),
                     abs(float(x["High"]) - previous_close),
                     abs(float(x["Low"]) - previous_close))
    range_atr = true_range / prior_atr if prior_atr > 0 else np.nan
    minimum_turnover = 1_000_000 if mode == "crops" else STRATEGY["min_dollar_volume"]
    extra = {
        f"Likvidita: medián obratu 20 dnů alespoň {minimum_turnover/1e6:g} mil.": bool(enough_volume and
            dollar_volume >= minimum_turnover and safe_float(volume.iloc[-1], 0) > 0),
        "Cenová mezera signálního dne nejvýše 1 ATR": bool(np.isfinite(gap_atr) and gap_atr <= STRATEGY["max_gap_atr"]),
        "Skutečné cenové rozpětí signálního dne nejvýše 2,5 ATR": bool(np.isfinite(range_atr) and range_atr <= STRATEGY["max_signal_range_atr"]),
    }
    checks = {**r["entry_checks"], **extra}
    eligible = all(checks.values())
    confluence = [col for col in ("EMA20", "EMA50")
                  if np.isfinite(r["support"]) and atr > 0
                  and abs(safe_float(x.get(col)) - r["support"]) <= 0.3 * atr]
    # Menší objem posledních 3 dnů korekce vs předchozích 17; pouze kontext.
    older = safe_float(valid_volume.iloc[:-3].mean())
    recent = safe_float(valid_volume.iloc[-3:].mean())
    quiet_pullback = bool(older > 0 and recent < older * 0.8
                          and len(d) >= 5 and d["Close"].iloc[-2] < d["Close"].iloc[-5])
    lower, upper = execution_bounds(r["support"], r["zone_high"], r["stop"], r["target1"], atr)
    r.update({
        "entry_checks": checks, "buy_allowed": eligible,
        "entry_score": round(100 * sum(checks.values()) / len(checks), 1),
        "entry_reason": ("Všechny vstupní podmínky jsou splněny podle poslední dokončené denní svíčky."
                         if eligible else "Nesplněno: " + "; ".join(k for k, ok in checks.items() if not ok)),
        "signal": BUY_SIGNAL if eligible else ("NEVSTUPOVAT" if r["buy_allowed"] else r["signal"]),
        "median_dollar_volume": dollar_volume, "signal_gap_atr": gap_atr,
        "signal_range_atr": range_atr, "bounce_volume_ratio": volume_ratio,
        "quiet_pullback": quiet_pullback, "confluence": confluence,
        "min_execution_price": lower, "max_execution_price": upper,
        "one_r_price": r["preferred_entry"] + r["risk_per_share"],
        "strategy_version": "4.8",
    })
    return r


def check_execution_price(item, proposed):
    """Pouze ručně zadaná cena vs uložený plán; není nový živý signál."""
    lower, upper = item["min_execution_price"], item["max_execution_price"]
    risk = proposed - item["stop"]
    rr = (item["target1"] - proposed) / risk if risk > 0 else np.nan
    context = item.get("options_context", {})
    context_fresh = not context.get("usable") or options_are_current(context)
    if item.get("mode") == "crops":
        context_fresh = weather_is_current(item.get("weather_context", {}))
    ok = bool(item["buy_allowed"] and context_fresh and np.isfinite(proposed)
              and lower - 1e-9 <= proposed <= upper + 1e-9
              and np.isfinite(rr) and rr >= STRATEGY["min_rr"] - 1e-9)
    return ok, rr


def calculate_entry_engine(d: pd.DataFrame, scores: dict,
                           regime: str = "UNKNOWN", data_ok: bool = True, mode="stocks") -> dict:
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
        "Aktuální data z dokončených denních svíček": bool(data_ok),
        "Potvrzený, opakovaně testovaný support": valid,
        "Cena u supportu": in_zone,
        "Potvrzení odrazu": bounce,
        "Rostoucí trend": trend_ok,
        "Skóre kvality alespoň 65": scores["quality_score"] >= STRATEGY["min_quality"],
        "RSI pod limitem": safe_float(x["RSI14"], 100) < STRATEGY["max_rsi"],
        "Strukturální cíl a R:R alespoň 1,5": bool(np.isfinite(rr1) and rr1 >= STRATEGY["min_rr"]),
        "Stop v povoleném rozsahu": bool(valid and stop > 0 and
                0 < risk <= STRATEGY["max_stop_atr"] * atr),
    }
    if mode == "stocks":
        checks["SPY není medvědí a je dostupný"] = regime in {"BULLISH", "NEUTRAL"}
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
    reason = ("Všechny vstupní podmínky jsou splněny podle poslední dokončené denní svíčky."
              if eligible else "Nesplněno: " + "; ".join(k for k, v in checks.items() if not v))
    result = {
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
    return trader_filters(d, result, mode)


# ---------------------- SEZÓNNOST KAŽDÉHO TICKERU --------------
# Pravidla jsou konzervativní výchozí hypotézy, nikoli validovaný model.
# Každý rok přispívá jedním vzorkem; aktuální rok do odhadu nevstupuje.
SEASONALITY_RULES = {
    "lookback_years": 10, "min_years": 5,
    "min_median_pct": 0.25, "positive_hit_rate_pct": 60.0,
    "negative_hit_rate_pct": 40.0, "recent_years": 3,
    "max_anchor_shift_days": 4, "max_gap_days": 7,
    "min_month_bars": 15,
}
MONTH_NAMES = ["Leden", "Únor", "Březen", "Duben", "Květen", "Červen",
               "Červenec", "Srpen", "Září", "Říjen", "Listopad", "Prosinec"]


def empty_seasonality(as_of_day, holding_days, reason):
    return {
        "usable": False, "status": "NEDOSTATEK DAT", "entry_block": False,
        "as_of_date": str(as_of_day), "holding_days": int(holding_days),
        "sample_years": 0, "mean_pct": np.nan, "median_pct": np.nan,
        "positive_pct": np.nan, "recent_median_pct": np.nan,
        "worst_pct": np.nan, "best_pct": np.nan,
        "samples": [], "monthly": [], "reason": reason,
    }


def seasonal_stats(values):
    """Podíl růstových období není pravděpodobností ziskového obchodu."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"sample_years": 0, "mean_pct": np.nan, "median_pct": np.nan,
                "positive_pct": np.nan, "worst_pct": np.nan, "best_pct": np.nan}
    return {"sample_years": int(len(a)), "mean_pct": float(np.mean(a)),
            "median_pct": float(np.median(a)),
            "positive_pct": float(100 * np.mean(a > 0)),
            "worst_pct": float(np.min(a)), "best_pct": float(np.max(a))}


def calculate_seasonality(history, as_of_day, holding_days=5):
    """Stejné datum v minulých letech, vstup při otevření a výstup při uzavření trhu.

    Víkend/svátek: historický referenční den posuneme na první dostupnou
    seanci v rámci nejvýše 4 kalendářních dnů. V nepřestupném roce nahradíme
    29. únor 28. únorem.
    Měsíční statistika je pouze přehled; neřídí pětidenní vstupní filtr.
    Výnosy upravených cen zahrnují vliv dividend; bez SL/TP a nákladů.
    """
    reference = pd.Timestamp(as_of_day).date()
    horizon = int(holding_days)
    context = empty_seasonality(reference.isoformat(), horizon,
                                "Historie cen není dostupná.")
    if horizon < 1:
        raise ValueError("Horizont sezónnosti musí být alespoň jeden obchodní den.")
    if history.empty or not {"Open", "Close"}.issubset(history.columns):
        return context
    h = history[["Open", "Close"]].copy()
    idx = pd.DatetimeIndex(h.index)
    # Zachováme místní kalendářní datum burzy, nikoli UTC datum.
    h.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    for col in ("Open", "Close"):
        h[col] = pd.to_numeric(h[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    first_year = reference.year - SEASONALITY_RULES["lookback_years"]
    # Nečteme žádnou cenu z aktuálního roku ani z budoucnosti.
    h = h.loc[(h.index >= pd.Timestamp(first_year - 1, 12, 1)) &
              (h.index < pd.Timestamp(reference.year, 1, 1))]
    if h.empty:
        return context
    samples = []
    for year in range(first_year, reference.year):
        anchor = pd.Timestamp(year, reference.month,
                              min(reference.day, monthrange(year, reference.month)[1]))
        i = int(h.index.searchsorted(anchor, side="left"))
        if i >= len(h) or (h.index[i] - anchor).days > SEASONALITY_RULES["max_anchor_shift_days"]:
            continue
        # Signální svíčka končí; vstup následuje až na open další seance.
        window = h.iloc[i + 1:i + 1 + horizon]
        if len(window) != horizon:
            continue
        sample_dates = h.index[i:i + 1 + horizon]
        gaps = sample_dates.to_series().diff().dt.days.dropna()
        if (gaps > SEASONALITY_RULES["max_gap_days"]).any():
            continue
        if not (window.notna().all().all() and (window > 0).all().all()):
            continue
        value = (float(window["Close"].iloc[-1]) / float(window["Open"].iloc[0]) - 1) * 100
        samples.append({"year": year, "reference_date": h.index[i].date().isoformat(),
                        "entry_date": window.index[0].date().isoformat(),
                        "exit_date": window.index[-1].date().isoformat(),
                        "return_pct": value})
    stats = seasonal_stats([s["return_pct"] for s in samples])
    recent = samples[-SEASONALITY_RULES["recent_years"]:]
    recent_median = float(np.median([s["return_pct"] for s in recent])) if recent else np.nan
    context.update({**stats, "samples": samples, "recent_median_pct": recent_median})

    # Výnos měsíce: poslední close měsíce / poslední close předchozího měsíce.
    # Nevytváříme výnos přes chybějící měsíc ani přes neúplný měsíc IPO.
    groups = {p: g for p, g in h.groupby(h.index.to_period("M"))}
    monthly_values = {m: [] for m in range(1, 13)}
    def complete_month(group, period):
        return (group is not None and len(group) >= SEASONALITY_RULES["min_month_bars"]
                and group["Close"].notna().all() and (group["Close"] > 0).all()
                and group.index[0].day <= 7
                and group.index[-1].day >= period.days_in_month - 7
                and not (group.index.to_series().diff().dt.days.dropna() >
                         SEASONALITY_RULES["max_gap_days"]).any())
    for period, group in groups.items():
        if not first_year <= period.year < reference.year:
            continue
        previous = groups.get(period - 1)
        if complete_month(group, period) and complete_month(previous, period - 1):
            value = (float(group["Close"].iloc[-1]) / float(previous["Close"].iloc[-1]) - 1) * 100
            monthly_values[period.month].append(value)
    context["monthly"] = [
        {"month": m, "name": MONTH_NAMES[m-1], **seasonal_stats(monthly_values[m])}
        for m in range(1, 13)
    ]
    if stats["sample_years"] < SEASONALITY_RULES["min_years"]:
        context["reason"] = (f"Počet dostupných ročních vzorků: {stats['sample_years']}; minimum je "
                             f"{SEASONALITY_RULES['min_years']}. Sezónní filtr se nepoužil.")
        return context

    values = np.asarray([s["return_pct"] for s in samples])
    # Jediný extrémní rok nesmí určovat znaménko mediánu.
    loo = [float(np.median(np.delete(values, j))) for j in range(len(values))]
    threshold = SEASONALITY_RULES["min_median_pct"]
    positive = (stats["median_pct"] >= threshold and
                stats["positive_pct"] >= SEASONALITY_RULES["positive_hit_rate_pct"] and
                recent_median > 0 and all(v > 0 for v in loo))
    negative = (stats["median_pct"] <= -threshold and
                stats["positive_pct"] <= SEASONALITY_RULES["negative_hit_rate_pct"] and
                recent_median < 0 and all(v < 0 for v in loo))
    context.update({"usable": True, "entry_block": bool(negative),
                    "status": "PŘÍZNIVÁ" if positive else ("NEPŘÍZNIVÁ" if negative else "SMÍŠENÁ"),
                    "reason": (f"Stejné období; počet obchodních dnů: {horizon}. Medián pohybu: "
                               f"{stats['median_pct']:+.2f} %, podíl růstových období: "
                               f"{stats['positive_pct']:.0f} %, počet sledovaných let: {stats['sample_years']}. "
                               f"Medián nejnovějších ročních vzorků (počet: {len(recent)}): {recent_median:+.2f} %. ") +
                              ("Opakovaně nepříznivé období blokuje nový nákup." if negative else
                               "Historie podporuje vstup při splnění ostatních podmínek." if positive else
                               "Výsledky jsou smíšené; sezónnost nový nákup neblokuje.")})
    return context


@st.cache_data(ttl=86400, show_spinner=False)
def get_seasonality_context(ticker, as_of_day, holding_days=5):
    """Denní cache na ticker, signální datum a obchodní horizont."""
    reference = pd.Timestamp(as_of_day).date()
    start = date(reference.year - SEASONALITY_RULES["lookback_years"] - 1, 12, 1)
    for attempt in range(2):
        try:
            history = yf.Ticker(ticker).history(
                start=start.isoformat(), end=(reference + timedelta(days=1)).isoformat(),
                interval="1d", auto_adjust=True, timeout=15,
            )
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)
            return calculate_seasonality(history, as_of_day, holding_days)
        except Exception as e:
            log.warning("Sezónní historie %s, pokus %s: %s", ticker, attempt + 1, e)
            if attempt == 0:
                time.sleep(0.5)
    return empty_seasonality(as_of_day, holding_days,
                             "Sezónní historii se nepodařilo načíst; filtr se nepoužil.")


def apply_seasonality_filter(entry, context):
    """Filtr smí vstup pouze omezit, nikdy obejít support nebo opční filtr."""
    r = dict(entry)
    r["pre_seasonality_buy_allowed"] = bool(entry["buy_allowed"])
    r["seasonality_context"] = context
    checks = dict(entry["entry_checks"])
    if context.get("usable"):
        checks["Sezónnost: bez opakovaně nepříznivého období"] = not context["entry_block"]
    r["entry_checks"] = checks
    r["buy_allowed"] = bool(entry["buy_allowed"] and all(checks.values()))
    r["entry_score"] = round(100 * sum(checks.values()) / len(checks), 1)
    if entry["buy_allowed"] and not r["buy_allowed"]:
        r["signal"] = "NEVSTUPOVAT · SEZÓNNOST"
        r["entry_reason"] = "Technické a opční podmínky jsou splněny, ale nákup blokuje sezónnost."
    else:
        r["entry_reason"] = entry["entry_reason"]
    r["entry_reason"] += " Sezónnost: " + context["reason"]
    r["strategy_version"] = "4.8"
    return r


def render_seasonality_panel(ticker, context):
    st.markdown("#### 📅 Sezónnost tohoto instrumentu")
    message = f"**{context['status']}** — {context['reason']}"
    if context["entry_block"]:
        st.warning(message)
    elif context["status"] == "PŘÍZNIVÁ":
        st.success(message)
    else:
        st.info(message)
    st.caption(f"Referenční datum: {context['as_of_date']} · počet obchodních dnů: "
               f"{context['holding_days']} · nejvýše {SEASONALITY_RULES['lookback_years']} "
               "posledních dokončených kalendářních let.")
    if context["samples"]:
        def pct(value):
            return f"{value:+.2f} %" if np.isfinite(value) else "—"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Medián pohybu", pct(context["median_pct"]))
        c2.metric("Průměr pohybu", pct(context["mean_pct"]))
        c3.metric("Podíl růstových období", f"{context['positive_pct']:.0f} %")
        c4.metric("Počet sledovaných let", str(context["sample_years"]))
        samples = pd.DataFrame(context["samples"]).rename(columns={
            "year": "Rok", "reference_date": "Referenční den",
            "entry_date": "Vstup při otevření trhu", "exit_date": "Výstup při uzavření trhu",
            "return_pct": "Pohyb (%)"})
        st.dataframe(samples, hide_index=True, use_container_width=True)
        st.download_button("Stáhnout sezónní údaje podle let (CSV)",
                           samples.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_sezonnost_{context['as_of_date']}.csv",
                           mime="text/csv", key=f"seasonal_samples_{ticker}")
    if context["monthly"]:
        monthly = pd.DataFrame(context["monthly"])
        st.markdown("**Přehled kalendářních měsíců**")
        fig = go.Figure(go.Bar(x=monthly["name"], y=monthly["median_pct"],
                              marker_color=["#16a34a" if v >= 0 else "#dc2626"
                                            for v in monthly["median_pct"]],
                              customdata=monthly[["sample_years", "positive_pct"]].to_numpy(),
                              hovertemplate="%{x}<br>Medián: %{y:.2f} %<br>Vzorky: %{customdata[0]}"
                                            "<br>Kladné měsíce: %{customdata[1]:.0f} %<extra></extra>"))
        fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
        fig.update_layout(height=300, yaxis_title="Medián měsíčního pohybu (%)",
                          margin=dict(l=20, r=20, t=20, b=30))
        st.plotly_chart(fig, use_container_width=True, key=f"seasonality_{ticker}")
        st.dataframe(monthly[["name", "sample_years", "mean_pct", "median_pct", "positive_pct"]]
                     .rename(columns={"name": "Měsíc", "sample_years": "Počet sledovaných let",
                                      "mean_pct": "Průměr (%)", "median_pct": "Medián (%)",
                                      "positive_pct": "Podíl růstových měsíců (%)"}),
                     hide_index=True, use_container_width=True)
    st.caption("Filtr hodnotí konkrétní datum a dobu držení; měsíční graf poskytuje širší kontext. "
               "Výpočet předpokládá vstup při otevření trhu po historickém referenčním dni "
               "a výstup při uzavření trhu poslední den sledovaného období. Kvůli svátku nebo víkendu "
               "lze referenční den posunout nejvýše o 4 kalendářní dny. Upravené ceny zohledňují "
               "dividendy a štěpení akcií. Výpočet nezahrnuje stop, cílovou cenu, poplatky ani skluz. "
               "Podíl růstových období není pravděpodobností zisku strategie. "
               "Při méně než 5 ročních vzorcích se filtr nepoužije. Příznivé hodnocení vyžaduje "
               "medián alespoň +0,25 % a nejméně 60 % růstových období. Nepříznivé hodnocení vyžaduje "
               "medián nejvýše −0,25 % a nejvýše 40 % růstových období. V obou případech musí mít "
               "stejné znaménko medián posledních 3 vzorků i celkový medián po vynechání kteréhokoli "
               "jednoho roku. Prahy jsou výchozí předpoklady; účinnost nebyla ověřena zpětným testem.")


# ---------------------- OPCE A FIREMNÍ UDÁLOSTI -----------------
# Pravidla jsou explicitní hypotézy; nejsou optimalizovaná ani backtestovaná.
# OI neidentifikuje instituce, směr obchodů, změnu OI ani dealer gamma exposure.
OPTION_RULES = {
    "max_expiries": 3, "max_dte": 60, "strike_band_pct": 20.0,
    "min_total_oi": 5000, "min_strike_oi": 1000,
    "min_strike_share": 0.20, "near_strike_atr": 0.5,
    "expiry_risk_days": 7, "put_call_attention": 1.5,
    "max_context_age_hours": 24,
}


def local_today():
    return datetime.now(ZoneInfo("Europe/Prague")).date()


def event_date(value, epoch=False):
    """Epoch kalendáře představuje UTC den; earnings index zachová lokální datum."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not epoch or not np.isfinite(value) or value <= 0:
                return None
            stamp = pd.to_datetime(value, unit="s", utc=True)
        else:
            stamp = pd.Timestamp(value)
        return None if pd.isna(stamp) else stamp.date()
    except (ValueError, TypeError, OverflowError):
        return None


def date_values(value):
    if isinstance(value, (list, tuple, pd.Series, pd.Index, np.ndarray)):
        return list(value)
    return [value]


def days_until(value, today=None):
    day = event_date(value)
    if day is None:
        return None
    delta = (day - (today or local_today())).days
    return delta if delta >= 0 else None


def calendar_dict(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        if "Earnings Date" in raw.index or "Dividend Date" in raw.index:
            return raw.iloc[:, 0].to_dict()
        return raw.iloc[0].to_dict()
    return {}


def normalize_events(info, calendar, earnings_index=(), today=None):
    """Nezaměňuje roční dividendRate s částkou jedné dividendy.

    Yahoo neposkytuje spolehlivou částku budoucí jednorázové výplaty.
    lastDividendValue je pouze historická reference, nikoli schválená příští částka.
    Dvě data Earnings Date zachováme jako rozpětí, ne jako dvě oddělené události.
    """
    today = today or local_today()
    info, calendar = info or {}, calendar_dict(calendar)
    def upcoming(*values):
        dates = [event_date(v, epoch=True) for v in values]
        return next((d.isoformat() for d in dates if d and d >= today), None)
    payment = upcoming(calendar.get("Dividend Date"), info.get("dividendDate"))
    ex_date = upcoming(calendar.get("Ex-Dividend Date"), info.get("exDividendDate"))
    earnings = sorted({d for v in date_values(calendar.get("Earnings Date"))
                       if (d := event_date(v, epoch=True)) is not None})
    # Zachováme i počátek rozpětí v minulosti, pokud jeho konec ještě nenastal.
    earnings_kind = "kalendář Yahoo; datum ověřte u emitenta"
    if not earnings or earnings[-1] < today:
        earnings = sorted({d for v in earnings_index
                           if (d := event_date(v)) is not None and d >= today})[:1]
        earnings_kind = "earnings_dates Yahoo; datum ověřte u emitenta"
    last_date = event_date(info.get("lastDividendDate"), epoch=True)
    last_amount = safe_float(info.get("lastDividendValue"), None)
    if last_amount is not None and (last_amount <= 0 or last_date is None or last_date > today):
        last_amount = None
    return {
        "currency": info.get("currency") or "měna neuvedena",
        "payment_date": payment, "ex_dividend_date": ex_date,
        "next_dividend_amount": None,
        "last_dividend_amount": last_amount,
        "last_dividend_ex_date": last_date.isoformat() if last_amount is not None else None,
        "earnings_start": earnings[0].isoformat() if earnings else None,
        "earnings_end": earnings[-1].isoformat() if earnings else None,
        "earnings_kind": earnings_kind,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_corporate_events(ticker):
    obj = yf.Ticker(ticker)
    errors, info, calendar, earnings_index = [], {}, {}, []
    try:
        info = obj.info or {}
    except Exception as e:
        log.warning("Info událostí %s: %s", ticker, e)
        errors.append("Základní údaje se nepodařilo načíst.")
    try:
        calendar = calendar_dict(obj.calendar)
    except Exception as e:
        log.warning("Kalendář %s: %s", ticker, e)
        errors.append("Firemní kalendář se nepodařilo načíst.")
    dates = [event_date(v, epoch=True) for v in date_values(calendar.get("Earnings Date"))]
    if not any(d and d >= local_today() for d in dates):
        try:
            frame = obj.get_earnings_dates(limit=12)
            if isinstance(frame, pd.DataFrame):
                earnings_index = frame.index
        except Exception as e:
            log.warning("Výsledková data %s: %s", ticker, e)
            errors.append("Náhradní výsledkový kalendář není dostupný.")
    result = normalize_events(info, calendar, earnings_index)
    result.update(errors=errors, fetched_at=datetime.now(timezone.utc).isoformat())
    return result


def clean_option_side(frame, side, expiry, price):
    """Neznámé OI ponechá NaN; chybějící data nejsou nula."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or "strike" not in frame:
        return pd.DataFrame()
    out = frame.copy()
    for col in ("strike", "openInterest", "volume"):
        values = out[col] if col in out else pd.Series(np.nan, index=out.index)
        out[col] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out.loc[out[col] < 0, col] = np.nan
    if "contractSize" in out:
        out = out.loc[out["contractSize"].eq("REGULAR")].copy()
    band = OPTION_RULES["strike_band_pct"] / 100
    out = out.loc[out["strike"].between(price * (1 - band), price * (1 + band))].copy()
    out["side"], out["expiry"] = side, expiry
    return out[["side", "expiry", "strike", "openInterest", "volume"]]


def summarize_options(rows, price, atr, expected_expiries=(), today=None):
    """Analýza nejvýše 3 expirací; koncentrace se měří po jednotlivých expiracích."""
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    empty = {
        "usable": False, "coverage_complete": False, "call_oi": None, "put_oi": None,
        "call_volume": None, "put_volume": None, "put_call_oi": None,
        "put_call_volume": None, "entry_block": False, "put_attention": False,
        "expiry_risk": False, "concentration": [], "top_strikes": [],
        "expiries": list(expected_expiries), "status": "Opční data nejsou dostupná.",
    }
    if rows.empty or not np.isfinite(price) or price <= 0:
        return empty
    calls, puts = rows.loc[rows["side"] == "CALL"], rows.loc[rows["side"] == "PUT"]
    call_oi = safe_float(calls["openInterest"].sum(min_count=1), None)
    put_oi = safe_float(puts["openInterest"].sum(min_count=1), None)
    call_vol = safe_float(calls["volume"].sum(min_count=1), None)
    put_vol = safe_float(puts["volume"].sum(min_count=1), None)
    complete = bool(expected_expiries) and rows["openInterest"].notna().all()
    for expiry in expected_expiries:
        subset = rows.loc[rows["expiry"] == expiry]
        complete = complete and set(subset["side"]) == {"CALL", "PUT"}
    total = (call_oi or 0) + (put_oi or 0)
    usable = bool(complete and total >= OPTION_RULES["min_total_oi"])
    # Výrazná koncentrace OI v nejbližších 7 dnech těsně nad cenou může omezit vstup.
    # Nejde o prokázanou rezistenci ani předpověď ceny či směru hedgingu.
    grouped = rows.groupby(["expiry", "side", "strike"], as_index=False).agg(
        openInterest=("openInterest", lambda x: x.sum(min_count=1)),
        volume=("volume", lambda x: x.sum(min_count=1)),
    )
    crowded = []
    for expiry in expected_expiries:
        expiry_day = event_date(expiry)
        dte = (expiry_day - today).days if expiry_day else -1
        series = grouped.loc[(grouped["expiry"] == expiry) & (grouped["side"] == "CALL")]
        series_total = safe_float(series["openInterest"].sum(min_count=1), 0)
        if not 0 <= dte <= OPTION_RULES["expiry_risk_days"] or series_total <= 0 or not atr > 0:
            continue
        near = series.loc[(series["strike"] >= price) &
                          ((series["strike"] - price) <= OPTION_RULES["near_strike_atr"] * atr) &
                          (series["openInterest"] >= OPTION_RULES["min_strike_oi"]) &
                          (series["openInterest"] / series_total >= OPTION_RULES["min_strike_share"])]
        for _, row in near.iterrows():
            crowded.append({"expiry": expiry, "strike": float(row["strike"]),
                            "oi": int(row["openInterest"]), "share": float(row["openInterest"] / series_total)})
    pcr = put_oi / call_oi if call_oi is not None and call_oi > 0 and put_oi is not None else None
    vcr = put_vol / call_vol if call_vol is not None and call_vol > 0 and put_vol is not None else None
    top = grouped.sort_values("openInterest", ascending=False).head(12)
    status = ("Dostatečný vzorek OI pro doplňkový filtr." if usable else
              "Neúplné údaje OI nebo příliš malý vzorek; filtr není použit.")
    return {**empty, "usable": usable, "coverage_complete": bool(complete),
            "call_oi": call_oi, "put_oi": put_oi, "call_volume": call_vol, "put_volume": put_vol,
            "put_call_oi": pcr, "put_call_volume": vcr, "entry_block": bool(usable and crowded),
            "put_attention": bool(usable and pcr is not None and pcr >= OPTION_RULES["put_call_attention"]),
            "expiry_risk": bool(crowded), "concentration": crowded,
            "top_strikes": top.where(pd.notna(top), None).to_dict("records"), "status": status}


@st.cache_data(ttl=900, show_spinner=False)
def get_options_context(ticker, price, atr):
    obj, frames, errors, selected = yf.Ticker(ticker), [], [], []
    today = datetime.now(ZoneInfo("America/New_York")).date()
    try:
        expiries = sorted(obj.options or ())
        selected = [e for e in expiries if event_date(e) is not None and
                    0 <= (event_date(e) - today).days <= OPTION_RULES["max_dte"]][:OPTION_RULES["max_expiries"]]
        for expiry in selected:
            try:
                chain = obj.option_chain(expiry)
                frames.extend([clean_option_side(chain.calls, "CALL", expiry, price),
                               clean_option_side(chain.puts, "PUT", expiry, price)])
            except Exception as e:
                log.warning("Opce %s %s: %s", ticker, expiry, e)
                errors.append(f"Expirace {expiry} se nepodařila načíst.")
    except Exception as e:
        log.warning("Opční expirace %s: %s", ticker, e)
        errors.append("Seznam opčních expirací se nepodařilo načíst.")
    frames = [f for f in frames if not f.empty]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = summarize_options(rows, price, atr, selected, today)
    if not selected and not errors:
        result["status"] = "Yahoo neposkytuje expirace v rozsahu 0–60 dnů; filtr není použit."
    if errors:
        result.update(usable=False, entry_block=False, put_attention=False,
                      status="Část opčních dat není dostupná; filtr není použit.")
    result.update(errors=errors, fetched_at=datetime.now(timezone.utc).isoformat())
    return result


def options_are_current(context, now=None):
    """Kontroluje stáří načtení, nikoli neznámé datum OI od poskytovatele."""
    try:
        fetched = pd.Timestamp(context.get("fetched_at"))
        now = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
        if pd.isna(fetched) or fetched.tzinfo is None:
            return False
        age = (now - fetched).total_seconds() / 3600
        return 0 <= age <= OPTION_RULES["max_context_age_hours"]
    except (TypeError, ValueError):
        return False


def apply_options_filter(entry, context):
    """Opce mohou vstup omezit; nikdy nepovolí vstup, který odmítla technická analýza."""
    r = dict(entry)
    r["technical_buy_allowed"] = r["buy_allowed"]
    r["options_context"] = context
    checks = dict(r["entry_checks"])
    if context.get("usable") and options_are_current(context):
        checks["Bez blízké koncentrace CALL OI před expirací"] = not context["entry_block"]
    r["entry_checks"] = checks
    r["buy_allowed"] = all(checks.values())
    r["entry_score"] = round(100 * sum(checks.values()) / len(checks), 1)
    if not r["buy_allowed"]:
        if r["technical_buy_allowed"]:
            r["signal"] = "NEVSTUPOVAT · OPČNÍ KONCENTRACE"
        r["entry_reason"] = "Nesplněno: " + "; ".join(k for k, ok in checks.items() if not ok)
    if not context.get("usable"):
        r["entry_reason"] += " Opční údaje chybí nebo nejsou úplné; opční filtr se nepoužil."
    r["strategy_version"] = "4.8"
    return r


def assess_trade_action(item: dict, held=False, position_stop=None, position_target=None) -> dict:
    """Odděleně nový vstup a správa existující long pozice, vždy z uzavřeného dne.

    Vlastní stop/cíl má přednost. Nezaměňujeme nový modelový stop s plánem držitele.
    Samotné OI nikdy nevyvolá prodej; vyžadujeme současně cenové oslabení.
    """
    result = {"trend_label": "BÝČÍ" if item["trend_ok"] else "NEPOTVRZENÝ / SLABÝ"}
    if not held:
        if item.get("mode") == "crops" and not weather_is_current(item.get("weather_context", {})):
            return {**result, "action_label": "ČEKAT / NEVSTUPOVAT",
                    "action_reason": "Počasí chybí, není úplné nebo je snímek starší než 6 hodin. Spusťte nový sken."}
        opts = item.get("options_context", {})
        if opts.get("usable") and not options_are_current(opts):
            return {**result, "action_label": "ČEKAT / NEVSTUPOVAT",
                    "action_reason": "Záznam opčních dat je starší než 24 hodin. Spusťte nový sken."}
        return {**result, "action_label": "NAKUPOVAT" if item["buy_allowed"] else "ČEKAT / NEVSTUPOVAT",
                "action_reason": item["entry_reason"]}
    price = item["price"]
    stop, target = safe_float(position_stop, 0), safe_float(position_target, 0)
    if stop > 0 and target > 0 and target <= stop:
        return {**result, "action_label": "OVĚŘIT PLÁN", "action_reason": "U nákupní pozice musí být cílová cena nad úrovní stopu."}
    if stop > 0 and price <= stop:
        label, reason = "PRODAT", "Závěrečná cena je na úrovni vašeho stopu nebo pod ní."
    elif target > 0 and price >= target:
        label, reason = "PRODAT", "Závěrečná cena dosáhla vašeho cíle nebo jej překonala."
    elif str(item.get("structure_event", "")).endswith("dolů"):
        label, reason = "PRODAT / ZVÁŽIT REDUKCI", "Potvrzený průlom swingové struktury dolů."
    else:
        opts = item.get("options_context", {})
        weak = price < item["ema50"] and item["macd_hist"] < 0 and not item["trend_ok"]
        option_risk = opts.get("usable") and options_are_current(opts) and (
            opts.get("entry_block") or opts.get("put_attention"))
        if weak and option_risk:
            label = "PRODAT / ZVÁŽIT REDUKCI"
            reason = ("Cena pod EMA50 a záporný MACD potvrzují oslabení; současně je přítomna "
                      "opční koncentrace nebo převaha PUT OI. Jde o konzervativní heuristiku, "
                      "nikoli důkaz prodejů institucí.")
        else:
            label = "DRŽET / SLEDOVAT"
            reason = "Podle poslední dokončené denní svíčky není splněna žádná podmínka pro výstup."
            if weak:
                reason += " Technické ukazatele slábnou; zkontrolujte vlastní stop."
            if option_risk:
                reason += " Opční data vyžadují zvýšenou pozornost, sama však nejsou důvodem k prodeji."
    if not (stop > 0 or target > 0):
        reason += " Nebyla zadána vlastní úroveň stopu ani cílová cena."
    return {**result, "action_label": label, "action_reason": reason}


def render_events_panel(ticker, events):
    st.markdown("#### 🗓️ Dividendy a firemní výsledky")
    def countdown(day):
        n = days_until(day)
        if n is None:
            return "Datum není dostupné"
        if n == 0:
            return "Dnes"
        unit = "den" if n == 1 else ("dny" if 2 <= n <= 4 else "dnů")
        return f"Za {n} {unit}"
    c1, c2, c3 = st.columns(3)
    c1.metric("Do výplaty dividendy", countdown(events.get("payment_date")))
    c1.caption(events.get("payment_date") or "Budoucí datum výplaty není dostupné.")
    c2.metric("Do ex-dividendového dne", countdown(events.get("ex_dividend_date")))
    c2.caption(events.get("ex_dividend_date") or "Budoucí ex-dividendové datum není dostupné.")
    first, last = events.get("earnings_start"), events.get("earnings_end")
    if first and last and first != last:
        n1, n2 = days_until(first), days_until(last)
        value = f"{n1}–{n2}" if n1 is not None else ("V odhadovaném termínu" if n2 is not None else "Datum není dostupné")
        c3.metric("Počet dnů do výsledků (rozmezí)", value)
        c3.caption(f"{first} až {last}")
    else:
        c3.metric("Do vyhlášení výsledků", countdown(first))
        c3.caption(first or "Budoucí datum výsledků není dostupné.")
    st.caption("Počty zbývajících dnů vycházejí z kalendáře a pražského časového pásma. Datum výsledků může být odhadem poskytovatele Yahoo Finance; "
               "čas zveřejnění zde není potvrzen. Výplata dividendy a ex-dividendový den jsou různé události.")
    c4, c5 = st.columns(2)
    currency = events.get("currency", "měna neuvedena")
    amount = events.get("last_dividend_amount")
    c4.metric("Poslední známá dividenda na akcii", f"{amount:.4f} {currency}" if amount is not None else "Nedostupné")
    c4.caption(f"Historický ex-dividendový den: {events.get('last_dividend_ex_date') or 'neuveden'}. "
               "Částka není potvrzením příští výplaty ani roční dividendou.")
    confirmed = st.checkbox("Znám potvrzenou částku příští výplaty z oznámení emitenta", key=f"div_confirmed_{ticker}")
    if confirmed:
        manual = st.number_input("Potvrzená dividenda na akcii za jednu výplatu", min_value=0.0,
                                 value=0.0, step=0.01, format="%.4f", key=f"div_amount_{ticker}")
        c5.metric("Příští dividenda na akcii — ručně zadaná", f"{manual:.4f} {currency}")
    else:
        c5.metric("Příští dividenda na akcii", "Částka nepotvrzena")
    n = days_until(first)
    if (n is not None and n <= 7) or (first and last and event_date(first) <= local_today() <= event_date(last)):
        st.warning("Vyhlášení výsledků se blíží nebo již spadá do odhadovaného termínu. Hrozí cenová mezera; jde o upozornění, nikoli automatické zablokování vstupu.")
    for error in events.get("errors", []):
        st.caption(error)
    st.caption(f"Kalendář načten: {events.get('fetched_at', 'neuvedeno')} (UTC). Pro aktuální data spusťte sken.")


def render_options_panel(context):
    st.markdown("#### 🐋 Opční zájem — otevřené kontrakty a jejich koncentrace")
    st.write(context.get("status", "Opční data nejsou dostupná."))
    if not context.get("usable"):
        st.warning("Opční filtr nebyl použit. Případný nákupní signál vychází z technických podmínek a dostupného sezónního hodnocení.")
    def number(key):
        v = context.get(key)
        return f"{v:,.0f}" if v is not None else "Nedostupné"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Otevřené CALL kontrakty", number("call_oi"))
    c2.metric("Otevřené PUT kontrakty", number("put_oi"))
    pcr, vcr = context.get("put_call_oi"), context.get("put_call_volume")
    c3.metric("PUT / CALL — OI", f"{pcr:.2f}" if pcr is not None else "Nedostupné")
    c4.metric("PUT / CALL — objem", f"{vcr:.2f}" if vcr is not None else "Nedostupné")
    st.caption(f"CALL objem: {number('call_volume')} · PUT objem: {number('put_volume')}. "
               "OI = počet otevřených kontraktů, objem = zobchodované kontrakty; objem/OI neprokazuje nové pozice.")
    st.caption("Vzorek: nejvýše 3 nejbližší expirace do 60 dnů, standardní kontrakty, "
               "realizační ceny v rozmezí ±20 % od signální ceny. Nejde o celý opční trh. "
               "PUT/CALL neodhaluje nákup/prodej, účastníka ani jeho záměr; zahrnuje i zajištění a spready.")
    if context.get("concentration"):
        st.write("**Blízké koncentrace CALL OI před expirací:**")
        for level in context["concentration"]:
            st.write(f"Realizační cena {level['strike']:.2f} · expirace {level['expiry']} · "
                     f"OI {level['oi']:,} · {level['share']:.0%} CALL OI této expirace ve vzorku.")
    if context.get("top_strikes"):
        frame = pd.DataFrame(context["top_strikes"]).rename(columns={
            "expiry": "Expirace", "side": "Typ", "strike": "Realizační cena", "openInterest": "OI", "volume": "Objem"})
        st.dataframe(frame, hide_index=True, use_container_width=True)
    if context.get("entry_block"):
        st.warning("Opční filtr omezuje nový vstup kvůli blízké koncentraci před expirací. "
                   "Koncentrace není prokázaná cenová rezistence.")
    if context.get("put_attention"):
        st.info("PUT/CALL OI ≥1,5: upozornění na složení pozic; samo o sobě není medvědí signál.")
    for error in context.get("errors", []):
        st.caption(error)
    st.caption(f"Načteno: {context.get('fetched_at', 'neuvedeno')} (UTC). "
               "Čas načtení není datem samotného OI; jeho přesné stáří Yahoo nezaručuje. "
               "Záznam opčních dat a denní signál nemusejí pocházet ze stejného okamžiku.")
    if not options_are_current(context):
        st.warning("Načtený opční kontext je starší než 24 hodin nebo nemá platný čas. Spusťte nový sken.")


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
        # Nové sloupce do existující Supabase tabulky nepřidáváme bez migrace.
        # Detail opcí, sezónnosti, kalendář a rozhodnutí držitele jsou v UI/CSV, ne v DB.
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
    """Simulace nákupů 4.8 aktuální části a horizontu; vstup na příštím open.

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
            stored_atr = safe_float(row.get("atr"))
            lower, upper = execution_bounds(support, upper, stop, tp, stored_atr)
            if not (np.isfinite(stop) and np.isfinite(tp) and stop < entry < tp
                    and lower - 1e-9 <= entry <= upper + 1e-9
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
        df[f"forward_return_{STRATEGY['holding_days']}d"] = returns
        df["outcome"] = outcomes
        return df
    except Exception as e:
        log.error("load_learning_stats selhalo: %s", e, exc_info=True)
        return pd.DataFrame()


# ---------------------- PLODINY A POČASÍ ----------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_crop_benchmark():
    """Rovnoměrný modelový koš tří ETC; všechny složky na stejných datech."""
    components = []
    for ticker in CROP_PRODUCTS:
        hist = download_history(ticker)
        if hist.empty:
            return pd.DataFrame()
        s = hist["Close"].copy()
        s.index = pd.Index(pd.DatetimeIndex(s.index).date)
        components.append(s.rename(ticker))
    common = pd.concat(components, axis=1, join="inner").dropna().sort_index()
    if len(common) < 60 or (common <= 0).any().any():
        return pd.DataFrame()
    basket = common.div(common.iloc[0]).mean(axis=1) * 100
    basket.index = pd.to_datetime(basket.index)
    return pd.DataFrame({"Open": basket, "High": basket, "Low": basket,
                         "Close": basket, "Volume": 1.0})


def parse_ensemble(payload):
    """Členy páruje pro všechny proměnné; chybějící hodnoty nejsou nuly."""
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    variables = ("temperature_2m_max", "temperature_2m_min", "precipitation_sum")
    units = payload.get("daily_units", {})
    if any(units.get(v) != ("mm" if v == "precipitation_sum" else "°C") for v in variables):
        raise ValueError("Neočekávané jednotky předpovědi.")
    members = set.intersection(*[
        {k[len(v):] for k in daily if k == v or k.startswith(v + "_member")}
        for v in variables])
    records = []
    for i, day in enumerate(dates):
        rows = []
        for suffix in sorted(members):
            arrays = [daily[v + suffix] for v in variables]
            if any(len(a) <= i for a in arrays):
                continue
            values = [safe_float(a[i]) for a in arrays]
            if (all(np.isfinite(x) for x in values) and values[0] >= values[1]
                    and values[2] >= 0):
                rows.append(values)
        if len(rows) < WEATHER_RULES["min_members"]:
            continue
        a = np.array(rows)
        records.append({"date": pd.Timestamp(day).date().isoformat(), "members": len(rows),
                        "max_c": float(np.median(a[:, 0])), "min_c": float(np.median(a[:, 1])),
                        "rain_mm": float(np.median(a[:, 2])),
                        "heat": float(np.mean(a[:, 0] >= WEATHER_RULES["heat_c"])),
                        "frost": float(np.mean(a[:, 1] <= WEATHER_RULES["frost_c"])),
                        "low_rain": float(np.mean(a[:, 2] < WEATHER_RULES["dry_day_mm"])),
                        "heavy_rain": float(np.mean(a[:, 2] >= WEATHER_RULES["heavy_rain_mm"]))})
    return records


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_region_weather(lat, lon):
    params = {"latitude": lat, "longitude": lon, "models": "gfs_seamless",
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
              "timezone": "auto", "forecast_days": WEATHER_RULES["forecast_days"],
              "temperature_unit": "celsius", "precipitation_unit": "mm"}
    url = "https://ensemble-api.open-meteo.com/v1/ensemble?" + urllib.parse.urlencode(params)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CropScanner/4.8"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        local_today = datetime.now(timezone.utc) + timedelta(seconds=payload.get("utc_offset_seconds", 0))
        days = [d for d in parse_ensemble(payload) if d["date"] >= local_today.date().isoformat()]
        return {"days": days, "fetched_at": datetime.now(timezone.utc).isoformat(),
                "error": None, "timezone": payload.get("timezone")}
    except Exception as exc:
        return {"days": [], "fetched_at": None, "error": str(exc), "timezone": None}


def weather_is_current(context, now=None):
    if not context.get("usable"):
        return False
    try:
        now = pd.Timestamp(now or datetime.now(timezone.utc))
        fetched = pd.Timestamp(context["fetched_at"])
        age = (now - fetched).total_seconds() / 3600
        return 0 <= age <= WEATHER_RULES["max_age_hours"]
    except (KeyError, ValueError, TypeError):
        return False


def weather_stress(day, calendar):
    sensitivity = CROP_CALENDARS[calendar].get(pd.Timestamp(day["date"]).month, 0.0)
    # Málo srážek není měření sucha. Maximum brání sčítání souvisejících rizik.
    return sensitivity * max(day["heat"], day["frost"], .5 * day["low_rain"],
                             .7 * day["heavy_rain"])


def calculate_crop_weather(crop, regions, previous=None, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    configured = CROP_REGIONS[crop]
    valid, summary, timestamps = {}, [], []
    for region in configured:
        data = regions.get(region["name"], {})
        days = data.get("days", [])
        current = weather_is_current({"usable": True, "fetched_at": data.get("fetched_at")}, now)
        if not current or len(days) < WEATHER_RULES["forecast_days"]:
            summary.append({"Oblast": region["name"], "Stav": "Nedostatek aktuálních dat",
                            "Detail": data.get("error") or "Neúplná předpověď"})
            continue
        timestamps.append(data["fetched_at"])
        valid[region["name"]] = {d["date"]: weather_stress(d, region["calendar"]) for d in days}
        summary.append({"Oblast": region["name"], "Stav": "Dostupné", "Modelová váha": region["weight"],
                        "Od": days[0]["date"], "Do": days[-1]["date"],
                        "Nejméně členů": min(d["members"] for d in days),
                        "Nejvyšší medián Tmax (°C)": max(d["max_c"] for d in days),
                        "Nejnižší medián Tmin (°C)": min(d["min_c"] for d in days),
                        "Součet denních mediánů srážek (mm)": sum(d["rain_mm"] for d in days),
                        "Index stresu": float(np.mean(list(valid[region["name"]].values())))})
    coverage = sum(r["weight"] for r in configured if r["name"] in valid)
    usable = coverage + 1e-9 >= WEATHER_RULES["min_coverage"]
    fetched_at = min(timestamps) if timestamps else None
    stress = (sum(r["weight"] * np.mean(list(valid[r["name"]].values()))
                  for r in configured if r["name"] in valid) / coverage if coverage else None)
    snapshot = {"crop": crop, "fetched_at": fetched_at, "regions": valid, "usable": usable}
    revision, common_days = None, 0
    if usable and previous and previous.get("usable") and previous.get("crop") == crop:
        try:
            age = (now - pd.Timestamp(previous["fetched_at"])).total_seconds() / 3600
            newer = fetched_at > previous["fetched_at"]
            names = [r for r in configured if r["name"] in valid and r["name"] in previous["regions"]]
            joint_weight = sum(r["weight"] for r in names)
            if names and joint_weight + 1e-9 >= WEATHER_RULES["min_coverage"] and newer and 0 <= age <= WEATHER_RULES["max_reference_age_hours"]:
                dates = set.intersection(*[set(valid[r["name"]]) & set(previous["regions"][r["name"]]) for r in names])
                common_days = len(dates)
                if common_days >= WEATHER_RULES["min_common_days"]:
                    revision = float(sum(r["weight"] * np.mean([
                        valid[r["name"]][day] - previous["regions"][r["name"]][day]
                        for day in sorted(dates)]) for r in names) / joint_weight)
        except (KeyError, ValueError, TypeError):
            pass
    block = bool(revision is not None and revision <= WEATHER_RULES["revision_block"])
    if not usable:
        reason = "Nedostatek aktuálních předpovědí pro sledované oblasti; nový nákup je blokován."
    elif revision is None:
        reason = "Počasí je dostupné. Chybí srovnatelná starší předpověď; směr změny není potvrzen."
    elif block:
        reason = "Modelový stres plodin výrazně klesl. Možný tlak na cenu; konzervativní filtr blokuje nákup."
    else:
        reason = ("Modelový stres plodin roste; možné omezení nabídky." if revision > 0 else
                  "Změna počasí nepřekročila práh blokace.")
    return {"usable": usable, "fetched_at": fetched_at, "coverage": coverage, "stress": stress,
            "revision": revision, "common_days": common_days, "entry_block": block or not usable,
            "reason": reason, "summary": summary, "snapshot": snapshot, "raw_regions": regions}


def prepare_crop_weather(crop, previous=None):
    configured = CROP_REGIONS[crop]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {r["name"]: pool.submit(fetch_region_weather, r["lat"], r["lon"]) for r in configured}
        regions = {name: future.result() for name, future in futures.items()}
    return calculate_crop_weather(crop, regions, previous)


def apply_weather_filter(entry, context):
    r = dict(entry)
    checks = dict(entry["entry_checks"])
    checks["Počasí: aktuální a dostatečně úplná data"] = weather_is_current(context)
    if context.get("revision") is not None:
        checks["Počasí: bez výrazného zlepšení podmínek plodin"] = not context["entry_block"]
    r.update(weather_context=context, entry_checks=checks,
             buy_allowed=bool(entry["buy_allowed"] and all(checks.values())),
             entry_score=round(100 * sum(checks.values()) / len(checks), 1))
    if entry["buy_allowed"] and not r["buy_allowed"]:
        r["signal"] = "NEVSTUPOVAT · POČASÍ"
    r["entry_reason"] = entry["entry_reason"] + " Počasí: " + context.get("reason", "Chybí předpověď; nákup blokován.")
    return r


def render_crop_panel(item):
    product = CROP_PRODUCTS[item["ticker"]]
    st.markdown(f"#### 🌾 {product['name']} · {product['product']}")
    st.caption("Cena jedné jednotky londýnského ETC v USD. Výnos ovlivňují futures, rolování a náklady produktu.")
    st.link_button("Informace emitenta", product["source"])
    context = item["weather_context"]
    st.markdown("#### 🌦️ Počasí v pěstitelských oblastech")
    st.write(context["reason"])
    st.caption(f"Načteno: {context['fetched_at'] or 'nedostupné'} · "
               f"pokrytí vah sledovaných bodů {context['coverage']:.0%} · model GFS ensemble, 14 kalendářních dní.")
    st.caption("Předpověď nepokrývá vždy celou zvolenou dobu držení. Počasí i cenový plán je potřeba kontrolovat průběžně.")
    c1, c2 = st.columns(2)
    c1.metric("Modelový index stresu plodin", f"{context['stress']:.2f}" if context["stress"] is not None else "—")
    c2.metric("Změna na shodných dnech", f"{context['revision']:+.2f}" if context["revision"] is not None else "—")
    st.dataframe(pd.DataFrame(context["summary"]), hide_index=True, use_container_width=True)
    st.caption("Index 0–1 není pravděpodobnost ztráty úrody ani růstu ceny. Body, váhy, měsíční citlivost a prahy "
               "jsou neověřené modelové předpoklady. Nízké srážky nejsou měření sucha ani půdní vlhkosti. "
               "Změna se porovnává jen na shodných budoucích dnech, alespoň 5, se snímkem do 72 hodin. "
               "Starší snímky se uchovávají pouze v aktuální relaci. Počasí nemůže povolit technicky zamítnutý vstup.")
    st.download_button("Stáhnout snímek počasí (JSON)", json.dumps(context, ensure_ascii=False, indent=2),
                       file_name=f"pocasi_{item['ticker']}.json", mime="application/json", key=f"weather_{item['ticker']}")
    st.markdown("**Další metriky k doplnění**")
    st.markdown("- [USDA WASDE](https://www.usda.gov/oce/commodity/wasde): zásoby/spotřeba, výnosy a změny odhadů.\n"
                "- [Crop Progress](https://www.nass.usda.gov/Publications/): kondice porostů a postup setí/sklizně.\n"
                "- [Export Sales](https://apps.fas.usda.gov/esrquery/): poptávka a vývozní závazky.\n"
                "- [CFTC COT](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm): pozice fondů.\n"
                "- Futures křivka a náklady rolování; spread a odchylka ceny ETC od hodnoty produktu.")
    st.caption("Tyto další metriky se zatím automaticky nestahují a nevstupují do skóre.")


# ---------------------- SCAN TICKERU --------------------------
def scan_ticker(ticker: str, spy: pd.DataFrame, mode="stocks", weather_context=None):
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
    entry = calculate_entry_engine(d, scores, regime, data_ok, mode)
    if mode == "stocks":
        entry = apply_options_filter(entry, get_options_context(ticker, float(x["Close"]), float(x["ATR14"])))
    else:
        entry.update(technical_buy_allowed=entry["buy_allowed"], options_context={
            "usable": False, "entry_block": False, "put_call_oi": None, "fetched_at": None})
    entry = apply_seasonality_filter(entry, get_seasonality_context(
        ticker, last_day.isoformat(), STRATEGY["holding_days"]))
    if mode == "crops":
        entry = apply_weather_filter(entry, weather_context or {})
        if not entry["signal"].startswith("PLODINY ·"):
            entry["signal"] = f"PLODINY · {STRATEGY['holding_days']}D · " + entry["signal"]
    events = get_corporate_events(ticker) if mode == "stocks" else {}
    if events.get("last_dividend_amount") is None and "Dividends" in d:
        dividends = d.loc[d["Dividends"] > 0, "Dividends"]
        if not dividends.empty:
            events = {**events, "last_dividend_amount": float(dividends.iloc[-1]),
                      "last_dividend_ex_date": pd.Timestamp(dividends.index[-1]).date().isoformat()}
    patterns = detect_patterns(d)
    structure = analyze_market_structure(d)

    return {
        "ticker":       ticker,
        "mode": mode, "holding_days": STRATEGY["holding_days"],
        "corporate_events": events,
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
def parallel_scan(tickers: list, spy: pd.DataFrame, max_workers: int = 4, progress_cb=None,
                  mode="stocks", weather_contexts=None):
    results = []
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_ticker, t, spy, mode,
                   (weather_contexts or {}).get(CROP_PRODUCTS[t]["crop"]) if mode == "crops" else None): t for t in tickers}
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

mode_label = st.radio("Část aplikace", ["Akcie", "Zemědělské komodity"], horizontal=True, key="asset_mode")
mode = "crops" if mode_label == "Zemědělské komodity" else "stocks"
if mode == "crops":
    horizon = st.selectbox("Maximální doba držení (obchodní dny)", [5, 10, 20], index=1, key="crop_horizon")
else:
    horizon = 5
STRATEGY["holding_days"] = horizon
BUY_SIGNAL = f"PLODINY · NÁKUPNÍ ZÓNA · 4.8 · {horizon}D" if mode == "crops" else "NÁKUPNÍ ZÓNA · 4.8"
return_column = f"forward_return_{horizon}d"
benchmark_label = "modelový koš pšenice/kukuřice/sóji" if mode == "crops" else "SPY"
# Výsledky a historie každého režimu a horizontu mají vlastní stav relace.
profile_key = f"{mode}_{horizon}"
profiles = st.session_state.setdefault("scan_profiles", {})
old_profile = st.session_state.get("active_profile")
if old_profile != profile_key:
    if old_profile:
        profiles[old_profile] = {key: st.session_state.get(key) for key in
                                 ("results", "regime", "regime_score", "history_df", "learning_df")}
    restored = profiles.get(profile_key, {})
    for key, default in {"results": [], "regime": None, "regime_score": 0,
                         "history_df": None, "learning_df": None}.items():
        st.session_state[key] = restored.get(key, default)
    st.session_state.active_profile = profile_key

hero_detail = ("Pšenice • Kukuřice • Sója • Sezónnost ETC • Počasí • Řízení rizika" if mode == "crops" else
               "Technická analýza • Sezónnost každé akcie • Otevřené opční pozice • Dividendy • Výsledky • Řízení rizika")
st.markdown(f"""
<div class="hero">
    <h3>📈 Spot Scanner 4.8</h3>
    <p>{hero_detail}</p>
</div>
""", unsafe_allow_html=True)
st.caption(f"Signály z D1; maximální doba držení {horizon} obchodních dní od vstupu. "
           "Pracovní postup: W1 pro širší trend, D1 pro signál, H4 pro zpřesnění vstupu. "
           "W1/H4 aplikace automaticky nevyhodnocuje. Nejlepší horizont musí potvrdit testy po nákladech.")

sb = get_supabase()

# ---------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("⚙️ Nastavení")

    if mode == "stocks":
        custom = st.text_input("Přidat burzovní symbol (ticker)", "").upper().strip()
        tickers = DEFAULT_TICKERS.copy()
        if custom:
            if is_valid_ticker(custom):
                if custom not in tickers:
                    tickers.insert(0, custom)
            else:
                st.warning("Neplatný ticker (A–Z, 0–9, tečka, pomlčka; nejvýše 10 znaků).")
    else:
        tickers = st.multiselect("Plodiny – londýnská ETC v USD", list(CROP_PRODUCTS), default=list(CROP_PRODUCTS),
                                format_func=lambda t: f"{CROP_PRODUCTS[t]['name']} · {t}")
        st.caption("Cenová data patří ETC, nikoli fyzické plodině. Vybraný ticker musí být dostupný u vašeho brokera.")

    st.markdown("---")
    st.subheader("💰 Řízení rizika")
    capital          = st.number_input("Kapitál ($)", min_value=100.0, value=5000.0, step=500.0)
    risk_pct         = st.slider("Riziko na obchod (%)", 0.25, 3.0, 1.0, 0.25)
    max_position_pct = st.slider("Maximální velikost pozice (%)", 5, 100, 25, 5)

    st.markdown("---")
    st.subheader("🔎 Filtry")
    min_quality        = st.slider("Minimální skóre kvality", 0, 100, 60)
    only_buy_zone      = st.checkbox("Pouze NÁKUPNÍ ZÓNA", False)
    only_positive_rs   = st.checkbox("Pouze RS > srovnávací koš" if mode == "crops" else "Pouze RS > S&P 500", False)
    exclude_overbought = st.checkbox("Vyloučit RSI > 75", True)
    only_breakout = st.checkbox("Pouze cenové průrazy", False)
    only_squeeze = st.checkbox("Pouze zúžení Bollingerových pásem", False)
    only_volume_spike = st.checkbox("Pouze zvýšený objem", False)
    max_workers        = st.slider("Paralelní vlákna", 2, 12, 4)

    st.markdown("---")
    st.caption("Zdroj dat: Yahoo Finance; pouze předchozí dokončené denní svíčky.")
    st.caption("Support: ≥2 oddělené testy. Vstup nejvýše 0,5 ATR a zároveň nejvýše 1,5 % nad ním. "
               f"R:R ≥1,5; horizont {horizon} obchodních dnů.")
    if mode == "stocks":
        st.caption("Opční filtr: vzorek ≥5 000 OI; blízká koncentrace CALL OI před expirací může blokovat nákup. "
                   "Jde o neověřené rizikové pravidlo. Chybějící opce nejsou potvrzení.")
    else:
        st.caption("Počasí: Open-Meteo / GFS ensemble; 14 kalendářních dní, aktualizace mezipaměti za hodinu. "
                   "Neúplné nebo staré počasí blokuje nový nákup. Výrazné snížení stresu na shodných dnech také. "
                   "Modelové prahy nejsou ověřené obchodní parametry.")
        st.caption("Relativní síla se porovnává s rovnoměrným košem všech 3 ETC. Kandidát je třetinou koše; "
                   "nejde o nezávislý index. Režim koše je kontext, nikoli filtr SPY. "
                   "Výchozí minimální medián obratu je 1 mil. USD denně.")
    st.caption("Sezónnost: stejné datum v posledních 10 uzavřených letech, horizont podle STRATEGY "
               f"({horizon} obchodních dnů), nejméně 5 ročních vzorků. Opakovaně nepříznivé období "
               "blokuje nový nákup. Příznivé období podporuje technicky povolený vstup. "
               "Chybějící historie se označí; sezónní filtr se tehdy nepoužije.")
    st.caption("Sezónní historie se uchovává v mezipaměti po dobu 24 hodin; první sken může trvat déle.")
    if mode == "stocks":
        st.caption("Dividendy a výsledky se načítají při skenu. Více požadavků může sken zpomalit.")
    st.caption("Analytická pomůcka – ne automatický obchodní systém.")

# ---------------------- TABS --------------------------------
tab_scan, tab_history, tab_learning = st.tabs([
    "🌾 Přehled plodin" if mode == "crops" else "🔎 Přehled akcií", "🗄️ Historie signálů", "🧠 Vyhodnocení signálů"
])

if "results" not in st.session_state:
    st.session_state.results = []
elif st.session_state.results and st.session_state.results[0].get("strategy_version") != "4.8":
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
        run = st.button("🌾 SPUSTIT SKEN PLODIN" if mode == "crops" else "🚀 SPUSTIT SKEN AKCIÍ",
                        type="primary", use_container_width=True, disabled=not tickers)

    if run:
        progress_bar = st.progress(0.0, text="Načítám ceny a počasí…" if mode == "crops" else "Analyzuji akcie…")

        def update_progress(done, total):
            progress_bar.progress(done / max(total, 1),
                                  text=f"Analyzuji {done}/{total}")

        spy = get_crop_benchmark() if mode == "crops" else get_spy()
        weather_contexts = {}
        if mode == "crops":
            weather_records = st.session_state.setdefault("weather_records", {})
            for crop in sorted({CROP_PRODUCTS[t]["crop"] for t in tickers}):
                saved_weather = weather_records.get(crop, {})
                latest = saved_weather.get("latest")
                ctx = prepare_crop_weather(crop, latest)
                if latest and ctx["fetched_at"] == latest.get("fetched_at"):
                    ctx = calculate_crop_weather(crop, ctx["raw_regions"], saved_weather.get("reference"))
                elif ctx["usable"]:
                    weather_records[crop] = {"latest": ctx["snapshot"], "reference": latest}
                weather_contexts[crop] = ctx
        regime, regime_score = market_regime(spy)
        st.session_state.regime = regime
        st.session_state.regime_score = regime_score

        raw = parallel_scan(tickers, spy, max_workers=max_workers, progress_cb=update_progress,
                            mode=mode, weather_contexts=weather_contexts)
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
        skipped = len(tickers) - len(raw)
        if skipped:
            st.warning(f"Počet instrumentů bez dostatečných cenových dat nebo s chybou skenu: {skipped}.")
        if spy.empty:
            st.warning("Srovnávací data nejsou dostupná; nový vstup je blokován.")

        saved = save_signals_bulk(sb, results)
        if saved:
            st.toast(f"Počet signálů uložených do databáze Supabase: {saved}.", icon="💾")

    # Zobraz režim trhu (přetrvává mezi reruny)
    if st.session_state.regime:
        color = {"BULLISH": "🟢", "BEARISH": "🔴",
                 "NEUTRAL": "🟡", "UNKNOWN": "⚪"}.get(st.session_state.regime, "⚪")
        st.info(f"{color} **Tržní režim ({benchmark_label}):** {st.session_state.regime} — "
                f"skóre {st.session_state.regime_score}/5")

    results = st.session_state.results

    if results:
        buy_count   = sum(r["signal"] == BUY_SIGNAL for r in results)
        avg_quality = np.mean([r["quality_score"] for r in results])
        avg_entry   = np.mean([r["entry_score"]   for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzované plodiny" if mode == "crops" else "Analyzované akcie", len(results))
        c2.metric("NÁKUPNÍ ZÓNA",    buy_count)
        c3.metric("Průměrné skóre kvality",  f"{avg_quality:.1f}")
        c4.metric("Průměrné vstupní skóre",    f"{avg_entry:.1f}")

        rows = []
        for r in results:
            assessment = assess_trade_action(r, held=st.session_state.get(f"held_{mode}_{r['ticker']}", False),
                                            position_stop=st.session_state.get(f"held_stop_{mode}_{r['ticker']}"),
                                            position_target=st.session_state.get(f"held_target_{mode}_{r['ticker']}"))
            rows.append({
                "Ticker":   r["ticker"],
                "Cena":     round(r["price"], 2),
                "Signál":   r["signal"],
                "Kvalita":  r["quality_score"],
                "Vstupní skóre":    r["entry_score"],
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
                "Nejvyšší vstupní cena podle plánu": r["max_execution_price"],
                "Medián obratu za 20 dnů": r["median_dollar_volume"],
                "Cenová mezera / ATR": r["signal_gap_atr"],
                "Souběh ukazatelů": ", ".join(r["confluence"]),
                "Datum dat": str(r["data"].index[-1].date()),
                "Relativní síla za 30 dnů":   r["rs_30d"],
                "RSI":      round(r["rsi"], 1),
                "Objem":    round(r["volume_ratio"], 2),
                "Průraz":   r["breakout"],
                "Zúžení Bollingerových pásem":  r["squeeze"],
                "Výrazně zvýšený objem": r["volume_spike"],
                "Struktura trhu": r["structure_trend"],
                "Průlom struktury": r["structure_event"],
                "Trend": assessment["trend_label"],
                "Orientační akce": assessment["action_label"],
                "Důvod akce": assessment["action_reason"],
                "Technický vstup před opčním filtrem": r["technical_buy_allowed"],
                "Opční filtr blokuje": r["options_context"]["entry_block"],
                "PUT/CALL OI": r["options_context"]["put_call_oi"],
                "Opční data načtena": r["options_context"]["fetched_at"],
                "Vstup před sezónním filtrem": r["pre_seasonality_buy_allowed"],
                "Sezónnost": r["seasonality_context"]["status"],
                "Sezónní filtr použit": r["seasonality_context"]["usable"],
                "Sezónní filtr blokuje": r["seasonality_context"]["entry_block"],
                "Sezónní horizont (obchodní dny)": r["seasonality_context"]["holding_days"],
                "Počet let pro sezónní výpočet": r["seasonality_context"]["sample_years"],
                "Sezónní medián (%)": r["seasonality_context"]["median_pct"],
                "Sezónní průměr (%)": r["seasonality_context"]["mean_pct"],
                "Podíl růstových období (%)": r["seasonality_context"]["positive_pct"],
                "Sezónní medián posledních vzorků (%)": r["seasonality_context"]["recent_median_pct"],
                "Sezónní referenční datum": r["seasonality_context"]["as_of_date"],
                "Zdůvodnění sezónního hodnocení": r["seasonality_context"]["reason"],
                "Výplata dividendy": r["corporate_events"].get("payment_date"),
                "Dny do výplaty dividendy": days_until(r["corporate_events"].get("payment_date")),
                "Poslední dividenda na akcii (historická)": r["corporate_events"].get("last_dividend_amount"),
                "Příští dividenda na akcii (ručně potvrzená)": (
                    st.session_state.get(f"div_amount_{r['ticker']}")
                    if st.session_state.get(f"div_confirmed_{r['ticker']}", False) else None),
                "Měna dividendy": r["corporate_events"].get("currency"),
                "Výsledky od": r["corporate_events"].get("earnings_start"),
                "Výsledky do": r["corporate_events"].get("earnings_end"),
                "Dny do nejbližšího termínu výsledků": days_until(r["corporate_events"].get("earnings_start")),
            })
            if mode == "crops":
                rows[-1].update({"Plodina": CROP_PRODUCTS[r["ticker"]]["name"],
                    "Počasí použitelné": r["weather_context"]["usable"],
                    "Počasí blokuje": r["weather_context"]["entry_block"],
                    "Index stresu plodin": r["weather_context"]["stress"],
                    "Změna stresu na shodných dnech": r["weather_context"]["revision"],
                    "Počasí načteno": r["weather_context"]["fetched_at"],
                    "Pokrytí modelových vah počasí": r["weather_context"]["coverage"]})
        df_show = pd.DataFrame(rows)
        if mode == "crops":
            df_show = df_show.drop(columns=[c for c in df_show if any(fragment in c for fragment in
                ("opční", "Opční", "PUT/CALL", "dividenda", "dividendy", "dividend", "Výsledky", "výsledků"))])
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
            assessment = assess_trade_action(item, held=st.session_state.get(f"held_{mode}_{ticker}", False),
                                             position_stop=st.session_state.get(f"held_stop_{mode}_{ticker}"),
                                             position_target=st.session_state.get(f"held_target_{mode}_{ticker}"))
            with st.expander(
                f"{ticker} · {assessment['action_label']} · trend {assessment['trend_label']} · "
                f"{item['signal']}",
                expanded=False,
            ):
                st.subheader(f"{ticker} · {item['signal']}")
                held = st.checkbox("Tento instrument již držím — vyhodnotit stávající nákupní pozici", key=f"held_{mode}_{ticker}")
                own_stop, own_target = None, None
                if held:
                    ps, pt = st.columns(2)
                    own_stop = ps.number_input("Vlastní stop pozice (0 = nezadán)", min_value=0.0,
                                               value=0.0, step=0.01, key=f"held_stop_{mode}_{ticker}")
                    own_target = pt.number_input("Vlastní cíl pozice (0 = nezadán)", min_value=0.0,
                                                 value=0.0, step=0.01, key=f"held_target_{mode}_{ticker}")
                    st.caption("Stop/cíl zadejte ve stejné měně a cenovém základu jako zobrazená cena. "
                               "Hodnocení porovnává závěrečnou cenu, nikoli dosažení úrovně pokynu během dne. "
                               "Nastavení pozice a ručně zadané údaje platí po dobu aktuální relace aplikace.")
                assessment = assess_trade_action(item, held, own_stop, own_target)
                action_message = (
                    f"**Orientační akce: {assessment['action_label']}** — "
                    f"{assessment['action_reason']}"
                )
                if assessment["action_label"] in {"NAKUPOVAT", "POMALU DOKUPOVAT"}:
                    st.success(action_message)
                elif assessment["action_label"] in {"ČEKAT / NEVSTUPOVAT", "DRŽET / SLEDOVAT"}:
                    st.info(action_message)
                elif assessment["action_label"] in {"PRODAT / ZVÁŽIT REDUKCI", "OVĚŘIT PLÁN"}:
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
                    {"Podmínka": label, "Splněna": ok}
                    for label, ok in item["entry_checks"].items()
                ]), hide_index=True, use_container_width=True)
                if mode == "crops":
                    render_crop_panel(item)
                else:
                    render_events_panel(ticker, item["corporate_events"])
                    render_options_panel(item["options_context"])
                render_seasonality_panel(ticker, item["seasonality_context"])
                st.markdown("#### 🧭 Struktura trhu")
                st.write(f"**{item['structure_trend']}** · {item['structure_event']}")
                st.caption(
                    f"Poslední potvrzené swingové úrovně: maximum ${item['swing_high']:.2f} · "
                    f"minimum ${item['swing_low']:.2f}. Swingy se potvrzují až po dalších svíčkách."
                )

                # Síťové doplňky se načítají jen po kliknutí v konkrétní ticker záložce.
                if mode == "stocks" and st.button("Načíst název, cenu před otevřením trhu a zprávy", key=f"load_context_{ticker}"):
                    st.session_state[f"context_loaded_{ticker}"] = True
                if mode == "stocks" and st.session_state.get(f"context_loaded_{ticker}", False):
                    name, sector = basic_info(ticker)
                    pre_price, pre_change = premarket(ticker)
                    st.markdown(f"**{name}** · Sektor: {sector}")
                    if pre_price is not None:
                        st.caption(f"Cena před otevřením trhu: ${pre_price:.2f} ({pre_change:+.2f} %)")
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
                        st.info("Pro tuto akcii nejsou dostupné zprávy.")

                d = item["data"].tail(180)
                st.markdown("#### 📊 Klíčové metriky")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Kvalita", f"{item['quality_score']:.1f}/100")
                k2.metric("Vstupní skóre", f"{item['entry_score']:.1f}/100")
                k3.metric("Trend", f"{item['trend_score']:.1f}/100")
                k4.metric("Síla pohybu", f"{item['momentum_score']:.1f}/100")
                k5, k6, k7, k8 = st.columns(4)
                k5.metric("Relativní síla vůči koši (30 dnů)" if mode == "crops" else "Relativní síla vůči SPY (30 dnů)", f"{item['rs_30d']:+.2f} %")
                k6.metric("RSI (14)", f"{item['rsi']:.1f}")
                k7.metric("ATR", f"${item['atr']:.2f} ({item['atr_pct']:.2f} %)")
                k8.metric("Objem vůči průměru", f"{item['volume_ratio']:.2f}×")

                st.markdown("#### 🧰 Obchodní plán")
                t1, t2, t3 = st.columns(3)
                t1.metric("Nejvyšší vstupní cena podle plánu", f"${item['max_execution_price']:.4f}")
                t2.metric("Medián obratu za 20 dnů", f"{item['median_dollar_volume']/1e6:.1f} mil.")
                t3.metric("Objem odrazu / předchozích 20 dnů", f"{item['bounce_volume_ratio']:.2f}×")
                st.caption("Obrat = upravená cena × objem, orientačně v měně kotace (u US tickerů USD). "
                           "Není to měření spreadu ani hloubky trhu. Maximum vstupu není pokyn k nákupu.")
                st.write("**Souběh se supportem:** " + (", ".join(item["confluence"]) or "bez blízké EMA20/EMA50"))
                st.write("**Korekce na klesajícím objemu:** " + ("ano" if item["quiet_pullback"] else "nepotvrzena"))
                st.caption("Souběh ukazatelů a objem jsou doplňkové informace; nenahrazují povinné podmínky.")
                st.write(f"**Plán od signálního vstupu:** stop ${item['stop']:.2f}; "
                         f"TP1 ${item['target1']:.2f}; úroveň +1R ${item['one_r_price']:.2f}; "
                         f"časový výstup nejpozději na konci {STRATEGY['holding_days']}. obchodního dne včetně vstupního.")
                st.caption("+1R je orientační milník, ne automatický přesun stopu. "
                           "Před objednávkou ověřte aktuálnost dat, zprávy, spread a graf. Aplikace pokyny neprovádí.")
                proposed = st.number_input("Zkušební vstupní cena (ručně, není živá kotace)",
                                           min_value=0.01, value=max(0.01, float(item["price"])),
                                           step=0.01, format="%.4f", key=f"proposed_{mode}_{ticker}")
                price_ok, proposed_rr = check_execution_price(item, proposed)
                st.write(f"R:R při této ceně, se stejným stopem a TP1: {proposed_rr:.2f}")
                if price_ok:
                    st.success("Zadaná cena vyhovuje uloženému plánu. Aktuální tržní situace nebyla znovu ověřena.")
                else:
                    st.warning("Zadaná cena nebo původní signál nesplňuje plán: nevstupovat podle tohoto výpočtu.")
                st.caption("Pro nový signální den spusťte nový sken. Simulace ceny níže nemění uložený signál.")

                shares, value, _, per_share = position_size(
                    capital, risk_pct, proposed, item["stop"], max_position_pct
                )
                if not price_ok:
                    shares, value, per_share = 0, 0.0, 0.0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Počet jednotek ETC – zkušební cena" if mode == "crops" else "Počet akcií – zkušební cena", shares)
                p2.metric("Hodnota pozice", f"${value:,.2f}")
                p3.metric("Riziko na jednotku", f"${per_share:,.2f}")
                p4.metric("Riziko pozice", f"${shares * per_share:,.2f}")
                st.caption(
                    "Výpočty nezahrnují poplatky, skluz, měnové riziko ani cenové mezery. "
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
        st.info("Historie není dostupná: není nastaveno připojení k databázi Supabase.")
    else:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            limit = st.number_input("Maximální počet řádků", 50, 2000, 500, 50)
        with col_f2:
            only_today = st.checkbox("Pouze dnešní", False)

        if st.button("🔄 Načíst historii", key="load_history"):
            try:
                q = sb.table("scanner_signals").select("*").order("signal_date", desc=True)
                if mode == "crops":
                    q = q.like("signal", "PLODINY ·%").like("signal", f"%{horizon}D%")
                else:
                    q = q.not_.like("signal", "PLODINY ·%")
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
                    "📥 Stáhnout historii (CSV)",
                    hist.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_historie.csv", "text/csv"
                )

# ============================================================
# TAB 3 – UČÍCÍ SE PŘEHLED
# ============================================================
with tab_learning:
    st.subheader("Vyhodnocení historických signálů")
    st.caption(f"Pouze uložené nákupní signály 4.8 pro aktuální část a horizont {horizon} dní. "
               "Simulace neobnovuje historické počasí, opční data ani sezónní vzorky. "
               "Výstupní pravidla pro držené pozice se zde netestují. Model: vstup při otevření příští seance v nákupní zóně, "
               f"kontrola R:R, výstup do {horizon} obchodních dní. "
               "Jde o simulaci bez nákladů, nikoli ověření ziskovosti. Denní OHLC data nemusí určit "
               "pořadí zásahu stopu a cíle v rámci stejného dne.")
    if mode == "crops":
        st.caption("Stávající databáze ukládá jeden záznam na ticker a signální den. Nový sken přepíše předchozí "
                   "záznam tohoto dne, i při změně horizontu. Podrobnosti počasí uchovejte pomocí JSON a CSV.")

    if sb is None:
        st.info("Vyhodnocení signálů vyžaduje připojení k databázi Supabase.")
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
                c3.metric("Nejasné pořadí výstupů", amb)
                c4.metric(f"Výstup {horizon}. den", opn)
                c5.metric("Vstup nebyl uskutečněn", not_filled)
                if pend:
                    st.caption(f"Počet signálů čekajících na dokončení {horizon} obchodních dní: {pend}.")

                resolved = tp1 + sl
                if resolved > 0:
                    win_rate = tp1 / resolved * 100
                    st.metric("TP1 / (TP1 + SL); bez časových a nejasných výstupů",
                              f"{win_rate:.1f} %",
                              help=f"Počet vyhodnocených obchodů: {resolved}")

                # Průměrný forward return podle outcome
                valid = learning.dropna(subset=[return_column])
                if not valid.empty:
                    st.markdown(f"#### Průměrný výnos (výstup nejpozději {horizon}. den) podle výsledku")
                    agg = valid.groupby("outcome")[return_column].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg, use_container_width=True)

                # Průměrný return podle signálu
                if not valid.empty:
                    st.markdown(f"#### Průměrný výnos (výstup nejpozději {horizon}. den) podle typu signálu")
                    agg2 = valid.groupby("signal")[return_column].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg2, use_container_width=True)

                st.markdown("#### Detailní data")
                st.dataframe(learning, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Stáhnout vyhodnocení (CSV)",
                    learning.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_learning.csv", "text/csv"
                )
