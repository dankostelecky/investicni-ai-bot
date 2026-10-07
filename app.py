# ============================================================
# KLONDIKE SPOT SCANNER 4.7 â€” sjednocenĂˇ a revidovanĂˇ verze
# Spot swing decision-support scanner s feedback loopem
# 4.7: sezĂłnnost kaĹľdĂ©ho tickeru podle data a swingovĂ©ho horizontu.
# ZachovĂˇno: opÄŤnĂ­ rizikovĂ˝ filtr; dividendy/vĂ˝sledky; sprĂˇva long pozice.
# Instalace: pip install streamlit yfinance numpy pandas plotly supabase tzdata
# SpuĹˇtÄ›nĂ­: streamlit run app.py
# Supabase: pĹŻvodnĂ­ schĂ©ma scanner_signals se nemÄ›nĂ­. Detail doplĹkĹŻ exportujte do CSV.
# Ĺ˝ĂˇdnĂˇ data nepĹ™edstĂ­rĂˇme: pĹ™Ă­ĹˇtĂ­ ÄŤĂˇstka dividendy nenĂ­ odvozena z dividendRate.
# OI neodhaluje identitu velkĂ˝ch hrĂˇÄŤĹŻ ani jejich nĂˇkupy/prodeje.
# ============================================================

import re
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

# ---------------------- LOGOVĂNĂŤ ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("klondike")

# ---------------------- KONFIGURACE ---------------------------
st.set_page_config(
    page_title="Spot Scanner 4.7",
    page_icon="đź“",
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

# ---------------------- POMOCNĂ‰ FUNKCE ------------------------

def safe_float(x, default=np.nan) -> float:
    """VracĂ­ default i pro NaN/inf, ne jen pĹ™i vĂ˝jimce."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def is_valid_ticker(t: str) -> bool:
    return bool(TICKER_RE.match(t or ""))


# ---------------------- DATOVĂ‰ FUNKCE -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """StĂˇhne historii z Yahoo Finance s retry a oĹˇetĹ™enĂ­m MultiIndex sloupcĹŻ."""
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(
                period=period, interval=interval, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(df.columns):
                log.warning("download_history(%s): chybĂ­ sloupce %s", ticker,
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


# ---------------------- INDIKĂTORY ----------------------------
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
    """UrÄŤĂ­ prĹŻraz, propad, Bollinger squeeze a neobvykle vysokĂ˝ objem."""
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
    """Klasifikuje swingovou strukturu a poslednĂ­ potvrzenĂ˝ prĹŻraz."""
    empty = {
        "structure_trend": "NEDOSTATEK DAT",
        "structure_event": "Bez potvrzenĂ©ho prĹŻrazu",
        "swing_high": np.nan,
        "swing_low": np.nan,
    }
    if len(d) < max(20, wing * 2 + 5):
        return empty

    highs = d["High"].to_numpy(dtype=float)
    lows = d["Low"].to_numpy(dtype=float)
    high_points, low_points = [], []
    # Swing se potvrdĂ­ aĹľ po wing nĂˇsledujĂ­cĂ­ch svĂ­ÄŤkĂˇch.
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
        trend = "BĂťÄŚĂŤ (HH + HL)"
        trend_code = "BULLISH"
    elif lower_high and lower_low:
        trend = "MEDVÄšDĂŤ (LH + LL)"
        trend_code = "BEARISH"
    else:
        trend = "SMĂŤĹ ENĂ / BOÄŚNĂŤ"
        trend_code = "MIXED"

    close = d["Close"]
    last_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    swing_high = float(last_high)
    swing_low = float(last_low)
    broke_up = last_close > swing_high and previous_close <= swing_high
    broke_down = last_close < swing_low and previous_close >= swing_low

    event = "Bez potvrzenĂ©ho prĹŻrazu"
    if broke_up:
        event = "CHoCH nahoru" if trend_code == "BEARISH" else "BOS nahoru"
    elif broke_down:
        event = "CHoCH dolĹŻ" if trend_code == "BULLISH" else "BOS dolĹŻ"

    return {
        "structure_trend": trend,
        "structure_event": event,
        "swing_high": swing_high,
        "swing_low": swing_low,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_news_context(ticker: str) -> dict:
    """NaÄŤte nÄ›kolik titulkĹŻ a spoÄŤĂ­tĂˇ pouze orientaÄŤnĂ­ slovnĂ­kovĂ˝ sentiment."""
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
            sentiment = "PĹ™evĂˇĹľnÄ› pozitivnĂ­ titulky"
        elif total_score < 0:
            sentiment = "PĹ™evĂˇĹľnÄ› negativnĂ­ titulky"
        else:
            sentiment = "SmĂ­ĹˇenĂ© nebo neutrĂˇlnĂ­ titulky"
        return {"articles": articles, "sentiment": sentiment, "score": total_score}
    except Exception as e:
        log.warning("NaÄŤtenĂ­ zprĂˇv pro %s selhalo: %s", ticker, e)
        return {"articles": [], "sentiment": "ZprĂˇvy nejsou dostupnĂ©", "score": 0}


# ---------------------- SUPPORT / RESISTANCE ------------------
# NastavenĂ­ strategie 4.7: vĂ˝chozĂ­ hypotĂ©zy, nikoli optimalizovanĂ© parametry.
BUY_SIGNAL = "NĂKUPNĂŤ ZĂ“NA Â· 4.7"
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
    """KonzervativnÄ› vynechĂˇ celĂ˝ aktuĂˇlnĂ­ den v timezone burzovnĂ­ho indexu.

    DneĹˇnĂ­ svĂ­ÄŤka se pouĹľije aĹľ dalĹˇĂ­ kalendĂˇĹ™nĂ­ den, i po konci seance.
    U indexu bez timezone vynechĂˇ dneĹˇek podle UTC.
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
    """Pivot musĂ­ bĂ˝t potvrzen uĹľ PĹED poslednĂ­ (signĂˇlnĂ­) svĂ­ÄŤkou."""
    values = d[column].to_numpy(dtype=float)
    points = []
    for i in range(wing, len(values) - wing - 1):
        window = values[i-wing:i+wing+1]
        extremum = np.min(window) if column == "Low" else np.max(window)
        if values[i] == extremum and np.count_nonzero(window == extremum) == 1:
            points.append((i, float(values[i])))
    return points


def structural_levels(d):
    """OpakovanÄ› testovanĂ˝ support z cenovĂ˝ch minim; bez EMA a nĂˇhradnĂ­ch ĂşrovnĂ­."""
    recent = d.tail(STRATEGY["lookback"]).reset_index(drop=True)
    atr = safe_float(d["ATR14"].iloc[-1])
    price = safe_float(d["Close"].iloc[-1])
    empty = {"support": np.nan, "support_floor": np.nan,
             "support_touches": 0, "support_age": np.nan, "resistances": []}
    if not np.isfinite(atr) or atr <= 0 or len(recent) < 15:
        return empty
    lows = confirmed_pivots(recent, "Low", STRATEGY["wing"])
    candidates = []
    # Shluk mĂˇ omezenou CELKOVOU ĹˇĂ­Ĺ™ku, nikoli Ĺ™etÄ›zenĂ­ vzdĂˇlenĂ˝ch minim.
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
        # ProraĹľenĂ˝ support nenĂ­ automaticky znovu platnĂ˝ po nĂˇvratu ceny.
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
        # Rezistence pĹ™ekonanĂˇ uzavĂ­racĂ­ cenou jiĹľ nenĂ­ pĹ™ekĂˇĹľkou nad supportem.
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
    """PovolenĂ˝ interval pĹ™i NEMÄšNNĂ‰M stopu a cĂ­li, bez poplatkĹŻ/slippage."""
    if not all(np.isfinite(v) for v in (support, zone_high, stop, target, atr)) or atr <= 0:
        return np.nan, np.nan
    lower = max(support, stop + STRATEGY["min_stop_atr"] * atr)
    upper = min(zone_high,
                (target + STRATEGY["min_rr"] * stop) / (1 + STRATEGY["min_rr"]),
                stop + STRATEGY["max_stop_atr"] * atr)
    if stop <= 0 or target <= lower or upper < lower - 1e-9:
        return np.nan, np.nan
    return lower, upper


def trader_filters(d, result):
    """Likvidita a vĂ˝jimeÄŤnĂ˝ pohyb jsou blokace; konfluence/objem jen kontext."""
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
    extra = {
        "Likvidita: mediĂˇn obratu 20 dnĹŻ alespoĹ 10 mil.": bool(enough_volume and
            dollar_volume >= STRATEGY["min_dollar_volume"] and safe_float(volume.iloc[-1], 0) > 0),
        "Gap signĂˇlnĂ­ho dne nejvĂ˝Ĺˇe 1 ATR": bool(np.isfinite(gap_atr) and gap_atr <= STRATEGY["max_gap_atr"]),
        "True range signĂˇlnĂ­ho dne nejvĂ˝Ĺˇe 2,5 ATR": bool(np.isfinite(range_atr) and range_atr <= STRATEGY["max_signal_range_atr"]),
    }
    checks = {**r["entry_checks"], **extra}
    eligible = all(checks.values())
    confluence = [col for col in ("EMA20", "EMA50")
                  if np.isfinite(r["support"]) and atr > 0
                  and abs(safe_float(x.get(col)) - r["support"]) <= 0.3 * atr]
    # MenĹˇĂ­ objem poslednĂ­ch 3 dnĹŻ korekce vs pĹ™edchozĂ­ch 17; pouze kontext.
    older = safe_float(valid_volume.iloc[:-3].mean())
    recent = safe_float(valid_volume.iloc[-3:].mean())
    quiet_pullback = bool(older > 0 and recent < older * 0.8
                          and len(d) >= 5 and d["Close"].iloc[-2] < d["Close"].iloc[-5])
    lower, upper = execution_bounds(r["support"], r["zone_high"], r["stop"], r["target1"], atr)
    r.update({
        "entry_checks": checks, "buy_allowed": eligible,
        "entry_score": round(100 * sum(checks.values()) / len(checks), 1),
        "entry_reason": ("VĹˇechny vstupnĂ­ podmĂ­nky splnÄ›ny na poslednĂ­m uzavĹ™enĂ©m dni."
                         if eligible else "NesplnÄ›no: " + "; ".join(k for k, ok in checks.items() if not ok)),
        "signal": BUY_SIGNAL if eligible else ("NEVSTUPOVAT" if r["buy_allowed"] else r["signal"]),
        "median_dollar_volume": dollar_volume, "signal_gap_atr": gap_atr,
        "signal_range_atr": range_atr, "bounce_volume_ratio": volume_ratio,
        "quiet_pullback": quiet_pullback, "confluence": confluence,
        "min_execution_price": lower, "max_execution_price": upper,
        "one_r_price": r["preferred_entry"] + r["risk_per_share"],
        "strategy_version": "4.7",
    })
    return r


def check_execution_price(item, proposed):
    """Pouze ruÄŤnÄ› zadanĂˇ cena vs uloĹľenĂ˝ plĂˇn; nenĂ­ novĂ˝ ĹľivĂ˝ signĂˇl."""
    lower, upper = item["min_execution_price"], item["max_execution_price"]
    risk = proposed - item["stop"]
    rr = (item["target1"] - proposed) / risk if risk > 0 else np.nan
    context = item.get("options_context", {})
    context_fresh = not context.get("usable") or options_are_current(context)
    ok = bool(item["buy_allowed"] and context_fresh and np.isfinite(proposed)
              and lower - 1e-9 <= proposed <= upper + 1e-9
              and np.isfinite(rr) and rr >= STRATEGY["min_rr"] - 1e-9)
    return ok, rr


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
    # Close musĂ­ bĂ˝t nad supportem; prĹŻnik knotem pod nÄ›j mĹŻĹľe bĂ˝t odmĂ­tnutĂ­ ceny.
    span = float(x["High"] - x["Low"])
    bounce = bool(valid and float(x["Low"]) <= zone_high
                  and float(x["High"]) >= zone_low
                  and float(x["Low"]) >= levels["support_floor"] - 0.5 * atr
                  and price > float(x["Open"]) and price >= float(d["Close"].iloc[-2])
                  and span > 0 and (price - float(x["Low"])) / span >= 0.6)
    # Pro aktuĂˇlnĂ­ signĂˇl vĹľdy R:R ze signĂˇlnĂ­ho close; plĂˇn ÄŤekĂˇnĂ­ z hornĂ­ hrany zĂłny.
    entry = price if in_zone else zone_high
    stop = (min(levels["support_floor"] - STRATEGY["stop_buffer_atr"] * atr,
                entry - STRATEGY["min_stop_atr"] * atr) if valid else np.nan)
    risk = entry - stop
    target1 = (min(resistance - 0.1 * atr, entry + STRATEGY["target1_atr"] * atr)
               if valid and np.isfinite(resistance) else np.nan)
    target2 = np.nan
    if valid and len(levels["resistances"]) > 1:
        # DruhĂ˝ cĂ­l je jen scĂ©nĂˇĹ™ po pĹ™ekonĂˇnĂ­ prvnĂ­ rezistence.
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
                    and not structure["structure_trend"].startswith("MEDVÄšDĂŤ"))
    checks = {
        "AktuĂˇlnĂ­ uzavĹ™enĂˇ data": bool(data_ok),
        "PotvrzenĂ˝ opakovanÄ› testovanĂ˝ support": valid,
        "Cena u supportu": in_zone,
        "PotvrzenĂ­ odrazu": bounce,
        "RostoucĂ­ trend": trend_ok,
        "SPY nenĂ­ medvÄ›dĂ­ a je dostupnĂ˝": regime in {"BULLISH", "NEUTRAL"},
        "Quality alespoĹ 65": scores["quality_score"] >= STRATEGY["min_quality"],
        "RSI pod limitem": safe_float(x["RSI14"], 100) < STRATEGY["max_rsi"],
        "StrukturĂˇlnĂ­ cĂ­l a R:R alespoĹ 1,5": bool(np.isfinite(rr1) and rr1 >= STRATEGY["min_rr"]),
        "Stop v povolenĂ©m rozsahu": bool(valid and stop > 0 and
                0 < risk <= STRATEGY["max_stop_atr"] * atr),
    }
    eligible = all(checks.values())
    if eligible:
        signal = BUY_SIGNAL
    elif not valid:
        signal = "BEZ POTVRZENĂ‰HO SUPPORTU"
    elif price > zone_high:
        signal = "ÄŚEKAT NA KOREKCI"
    elif not bounce:
        signal = "ÄŚEKAT NA ODRAZ"
    else:
        signal = "NEVSTUPOVAT"
    reason = ("VĹˇechny vstupnĂ­ podmĂ­nky splnÄ›ny na poslednĂ­m uzavĹ™enĂ©m dni."
              if eligible else "NesplnÄ›no: " + "; ".join(k for k, v in checks.items() if not v))
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
    return trader_filters(d, result)


# ---------------------- SEZĂ“NNOST KAĹ˝DĂ‰HO TICKERU --------------
# Pravidla jsou konzervativnĂ­ vĂ˝chozĂ­ hypotĂ©zy, nikoli validovanĂ˝ model.
# KaĹľdĂ˝ rok pĹ™ispĂ­vĂˇ jednĂ­m vzorkem; aktuĂˇlnĂ­ rok do odhadu nevstupuje.
SEASONALITY_RULES = {
    "lookback_years": 10, "min_years": 5,
    "min_median_pct": 0.25, "positive_hit_rate_pct": 60.0,
    "negative_hit_rate_pct": 40.0, "recent_years": 3,
    "max_anchor_shift_days": 4, "max_gap_days": 7,
    "min_month_bars": 15,
}
MONTH_NAMES = ["Leden", "Ăšnor", "BĹ™ezen", "Duben", "KvÄ›ten", "ÄŚerven",
               "ÄŚervenec", "Srpen", "ZĂˇĹ™Ă­", "ĹĂ­jen", "Listopad", "Prosinec"]


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
    """PodĂ­l kladnĂ˝ch pohybĹŻ nenĂ­ pravdÄ›podobnost ziskovĂ©ho obchodu."""
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
    """StejnĂ© datum v minulĂ˝ch letech, pĹ™Ă­ĹˇtĂ­ open -> close H-tĂ©ho dne.

    VĂ­kend/svĂˇtek: historickĂ˝ referenÄŤnĂ­ den posuneme na prvnĂ­ dostupnou
    seanci v rĂˇmci nejvĂ˝Ĺˇe 4 kalendĂˇĹ™nĂ­ch dnĹŻ. Ăšnor 29 mapujeme na Ăşnor 28.
    MÄ›sĂ­ÄŤnĂ­ statistika je pouze pĹ™ehled; neĹ™Ă­dĂ­ pÄ›tidennĂ­ vstupnĂ­ filtr.
    VĂ˝nosy upravenĂ˝ch cen zahrnujĂ­ vliv dividend; bez SL/TP a nĂˇkladĹŻ.
    """
    reference = pd.Timestamp(as_of_day).date()
    horizon = int(holding_days)
    context = empty_seasonality(reference.isoformat(), horizon,
                                "Historie cen nenĂ­ dostupnĂˇ.")
    if horizon < 1:
        raise ValueError("Horizont sezĂłnnosti musĂ­ bĂ˝t alespoĹ jeden obchodnĂ­ den.")
    if history.empty or not {"Open", "Close"}.issubset(history.columns):
        return context
    h = history[["Open", "Close"]].copy()
    idx = pd.DatetimeIndex(h.index)
    # ZachovĂˇme mĂ­stnĂ­ kalendĂˇĹ™nĂ­ datum burzy, nikoli UTC datum.
    h.index = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    h = h[~h.index.duplicated(keep="last")].sort_index()
    for col in ("Open", "Close"):
        h[col] = pd.to_numeric(h[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    first_year = reference.year - SEASONALITY_RULES["lookback_years"]
    # NeÄŤteme ĹľĂˇdnou cenu z aktuĂˇlnĂ­ho roku ani z budoucnosti.
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
        # SignĂˇlnĂ­ svĂ­ÄŤka konÄŤĂ­; vstup nĂˇsleduje aĹľ na open dalĹˇĂ­ seance.
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

    # VĂ˝nos mÄ›sĂ­ce: poslednĂ­ close mÄ›sĂ­ce / poslednĂ­ close pĹ™edchozĂ­ho mÄ›sĂ­ce.
    # NevytvĂˇĹ™Ă­me vĂ˝nos pĹ™es chybÄ›jĂ­cĂ­ mÄ›sĂ­c ani pĹ™es neĂşplnĂ˝ mÄ›sĂ­c IPO.
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
        context["reason"] = (f"Jen {stats['sample_years']} roÄŤnĂ­ch vzorkĹŻ; minimum je "
                             f"{SEASONALITY_RULES['min_years']}. SezĂłnnĂ­ filtr se nepouĹľil.")
        return context

    values = np.asarray([s["return_pct"] for s in samples])
    # JedinĂ˝ extrĂ©mnĂ­ rok nesmĂ­ urÄŤovat znamĂ©nko mediĂˇnu.
    loo = [float(np.median(np.delete(values, j))) for j in range(len(values))]
    threshold = SEASONALITY_RULES["min_median_pct"]
    positive = (stats["median_pct"] >= threshold and
                stats["positive_pct"] >= SEASONALITY_RULES["positive_hit_rate_pct"] and
                recent_median > 0 and all(v > 0 for v in loo))
    negative = (stats["median_pct"] <= -threshold and
                stats["positive_pct"] <= SEASONALITY_RULES["negative_hit_rate_pct"] and
                recent_median < 0 and all(v < 0 for v in loo))
    context.update({"usable": True, "entry_block": bool(negative),
                    "status": "PĹĂŤZNIVĂ" if positive else ("NEPĹĂŤZNIVĂ" if negative else "SMĂŤĹ ENĂ"),
                    "reason": (f"StejnĂ© obdobĂ­, {horizon} obchodnĂ­ch dnĹŻ: mediĂˇn "
                               f"{stats['median_pct']:+.2f} %, kladnĂ˝ch pohybĹŻ "
                               f"{stats['positive_pct']:.0f} %, vzorek {stats['sample_years']} let. "
                               f"MediĂˇn poslednĂ­ch {len(recent)} vzorkĹŻ {recent_median:+.2f} %. ") +
                              ("OpakovanÄ› nepĹ™Ă­znivĂ© obdobĂ­ blokuje novĂ˝ nĂˇkup." if negative else
                               "Historie podporuje vstup pĹ™i splnÄ›nĂ­ ostatnĂ­ch podmĂ­nek." if positive else
                               "VĂ˝sledky jsou smĂ­ĹˇenĂ©; sezĂłnnost novĂ˝ nĂˇkup neblokuje.")})
    return context


@st.cache_data(ttl=86400, show_spinner=False)
def get_seasonality_context(ticker, as_of_day, holding_days=5):
    """DennĂ­ cache na ticker, signĂˇlnĂ­ datum a obchodnĂ­ horizont."""
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
            log.warning("SezĂłnnĂ­ historie %s, pokus %s: %s", ticker, attempt + 1, e)
            if attempt == 0:
                time.sleep(0.5)
    return empty_seasonality(as_of_day, holding_days,
                             "SezĂłnnĂ­ historii se nepodaĹ™ilo naÄŤĂ­st; filtr se nepouĹľil.")


def apply_seasonality_filter(entry, context):
    """Filtr smĂ­ vstup pouze omezit, nikdy obejĂ­t support nebo opÄŤnĂ­ filtr."""
    r = dict(entry)
    r["pre_seasonality_buy_allowed"] = bool(entry["buy_allowed"])
    r["seasonality_context"] = context
    checks = dict(entry["entry_checks"])
    if context.get("usable"):
        checks["SezĂłnnost: bez opakovanÄ› nepĹ™Ă­znivĂ©ho obdobĂ­"] = not context["entry_block"]
    r["entry_checks"] = checks
    r["buy_allowed"] = bool(entry["buy_allowed"] and all(checks.values()))
    r["entry_score"] = round(100 * sum(checks.values()) / len(checks), 1)
    if entry["buy_allowed"] and not r["buy_allowed"]:
        r["signal"] = "NEVSTUPOVAT Â· SEZĂ“NNOST"
        r["entry_reason"] = "TechnickĂ© a opÄŤnĂ­ podmĂ­nky splnÄ›ny, nĂˇkup blokuje sezĂłnnost."
    else:
        r["entry_reason"] = entry["entry_reason"]
    r["entry_reason"] += " SezĂłnnost: " + context["reason"]
    r["strategy_version"] = "4.7"
    return r


def render_seasonality_panel(ticker, context):
    st.markdown("#### đź“… SezĂłnnost tĂ©to akcie")
    message = f"**{context['status']}** â€” {context['reason']}"
    if context["entry_block"]:
        st.warning(message)
    elif context["status"] == "PĹĂŤZNIVĂ":
        st.success(message)
    else:
        st.info(message)
    st.caption(f"ReferenÄŤnĂ­ datum: {context['as_of_date']} Â· horizont: "
               f"{context['holding_days']} obchodnĂ­ch dnĹŻ Â· poslednĂ­ch nejvĂ˝Ĺˇe "
               f"{SEASONALITY_RULES['lookback_years']} uzavĹ™enĂ˝ch kalendĂˇĹ™nĂ­ch let.")
    if context["samples"]:
        def pct(value):
            return f"{value:+.2f} %" if np.isfinite(value) else "â€”"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MediĂˇn pohybu", pct(context["median_pct"]))
        c2.metric("PrĹŻmÄ›r pohybu", pct(context["mean_pct"]))
        c3.metric("KladnĂ© pohyby", f"{context['positive_pct']:.0f} %")
        c4.metric("RoÄŤnĂ­ vzorky", str(context["sample_years"]))
        samples = pd.DataFrame(context["samples"]).rename(columns={
            "year": "Rok", "reference_date": "ReferenÄŤnĂ­ den",
            "entry_date": "Vstup na open", "exit_date": "VĂ˝stup na close",
            "return_pct": "Pohyb (%)"})
        st.dataframe(samples, hide_index=True, use_container_width=True)
        st.download_button("StĂˇhnout roÄŤnĂ­ sezĂłnnĂ­ vzorky CSV",
                           samples.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{ticker}_sezonnost_{context['as_of_date']}.csv",
                           mime="text/csv", key=f"seasonal_samples_{ticker}")
    if context["monthly"]:
        monthly = pd.DataFrame(context["monthly"])
        st.markdown("**PĹ™ehled kalendĂˇĹ™nĂ­ch mÄ›sĂ­cĹŻ**")
        fig = go.Figure(go.Bar(x=monthly["name"], y=monthly["median_pct"],
                              marker_color=["#16a34a" if v >= 0 else "#dc2626"
                                            for v in monthly["median_pct"]],
                              customdata=monthly[["sample_years", "positive_pct"]].to_numpy(),
                              hovertemplate="%{x}<br>MediĂˇn: %{y:.2f} %<br>Vzorky: %{customdata[0]}"
                                            "<br>KladnĂ© mÄ›sĂ­ce: %{customdata[1]:.0f} %<extra></extra>"))
        fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
        fig.update_layout(height=300, yaxis_title="MediĂˇn mÄ›sĂ­ÄŤnĂ­ho pohybu (%)",
                          margin=dict(l=20, r=20, t=20, b=30))
        st.plotly_chart(fig, use_container_width=True, key=f"seasonality_{ticker}")
        st.dataframe(monthly[["name", "sample_years", "mean_pct", "median_pct", "positive_pct"]]
                     .rename(columns={"name": "MÄ›sĂ­c", "sample_years": "RoÄŤnĂ­ vzorky",
                                      "mean_pct": "PrĹŻmÄ›r (%)", "median_pct": "MediĂˇn (%)",
                                      "positive_pct": "KladnĂ© mÄ›sĂ­ce (%)"}),
                     hide_index=True, use_container_width=True)
    st.caption("Filtr hodnotĂ­ konkrĂ©tnĂ­ datum a swingovĂ˝ horizont; mÄ›sĂ­ÄŤnĂ­ graf slouĹľĂ­ pro kontext. "
               "Vstup na open po historickĂ©m referenÄŤnĂ­m dni, vĂ˝stup na close poslednĂ­ho dne okna. "
               "HistorickĂ˝ referenÄŤnĂ­ den lze posunout nejvĂ˝Ĺˇe o 4 kalendĂˇĹ™nĂ­ dny kvĹŻli svĂˇtku/vĂ­kendu. "
               "UpravenĂ© ceny zahrnujĂ­ vliv dividend a splitĹŻ. Bez stopu/cĂ­le, poplatkĹŻ a skluzu; "
               "podĂ­l kladnĂ˝ch pohybĹŻ nenĂ­ pravdÄ›podobnost zisku strategie. "
               "PĹ™i mĂ©nÄ› neĹľ 5 roÄŤnĂ­ch vzorcĂ­ch se filtr nepouĹľije. Pro pĹ™Ă­znivĂ˝/nepĹ™Ă­znivĂ˝ signĂˇl "
               "je nutnĂ˝ mediĂˇn alespoĹ +0,25/nejvĂ˝Ĺˇe â’0,25 %, kladnĂ© pohyby alespoĹ 60/nejvĂ˝Ĺˇe 40 %, "
               "stejnĂ˝ smÄ›r mediĂˇnu poslednĂ­ch 3 vzorkĹŻ i po vynechĂˇnĂ­ kterĂ©hokoli jednoho roku. "
               "Prahy jsou vĂ˝chozĂ­ hypotĂ©zy; ĂşÄŤinnost nenĂ­ ovÄ›Ĺ™ena backtestem.")


# ---------------------- OPCE A FIREMNĂŤ UDĂLOSTI -----------------
# Pravidla jsou explicitnĂ­ hypotĂ©zy; nejsou optimalizovanĂˇ ani backtestovanĂˇ.
# OI neidentifikuje instituce, smÄ›r obchodĹŻ, zmÄ›nu OI ani dealer gamma exposure.
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
    """Epoch kalendĂˇĹ™e pĹ™edstavuje UTC den; earnings index zachovĂˇ lokĂˇlnĂ­ datum."""
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
    """NezamÄ›Ĺuje roÄŤnĂ­ dividendRate s ÄŤĂˇstkou jednĂ© dividendy.

    Yahoo neposkytuje spolehlivou ÄŤĂˇstku budoucĂ­ jednorĂˇzovĂ© vĂ˝platy.
    lastDividendValue je pouze historickĂˇ reference, nikoli schvĂˇlenĂˇ pĹ™Ă­ĹˇtĂ­ ÄŤĂˇstka.
    DvÄ› data Earnings Date zachovĂˇme jako rozpÄ›tĂ­, ne jako dvÄ› oddÄ›lenĂ© udĂˇlosti.
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
    # ZachovĂˇme i poÄŤĂˇtek rozpÄ›tĂ­ v minulosti, pokud jeho konec jeĹˇtÄ› nenastal.
    earnings_kind = "kalendĂˇĹ™ Yahoo; datum ovÄ›Ĺ™te u emitenta"
    if not earnings or earnings[-1] < today:
        earnings = sorted({d for v in earnings_index
                           if (d := event_date(v)) is not None and d >= today})[:1]
        earnings_kind = "earnings_dates Yahoo; datum ovÄ›Ĺ™te u emitenta"
    last_date = event_date(info.get("lastDividendDate"), epoch=True)
    last_amount = safe_float(info.get("lastDividendValue"), None)
    if last_amount is not None and (last_amount <= 0 or last_date is None or last_date > today):
        last_amount = None
    return {
        "currency": info.get("currency") or "mÄ›na neuvedena",
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
        log.warning("Info udĂˇlostĂ­ %s: %s", ticker, e)
        errors.append("ZĂˇkladnĂ­ Ăşdaje se nepodaĹ™ilo naÄŤĂ­st.")
    try:
        calendar = calendar_dict(obj.calendar)
    except Exception as e:
        log.warning("KalendĂˇĹ™ %s: %s", ticker, e)
        errors.append("FiremnĂ­ kalendĂˇĹ™ se nepodaĹ™ilo naÄŤĂ­st.")
    dates = [event_date(v, epoch=True) for v in date_values(calendar.get("Earnings Date"))]
    if not any(d and d >= local_today() for d in dates):
        try:
            frame = obj.get_earnings_dates(limit=12)
            if isinstance(frame, pd.DataFrame):
                earnings_index = frame.index
        except Exception as e:
            log.warning("VĂ˝sledkovĂˇ data %s: %s", ticker, e)
            errors.append("NĂˇhradnĂ­ vĂ˝sledkovĂ˝ kalendĂˇĹ™ nenĂ­ dostupnĂ˝.")
    result = normalize_events(info, calendar, earnings_index)
    result.update(errors=errors, fetched_at=datetime.now(timezone.utc).isoformat())
    return result


def clean_option_side(frame, side, expiry, price):
    """NeznĂˇmĂ© OI ponechĂˇ NaN; chybÄ›jĂ­cĂ­ data nejsou nula."""
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
    """AnalĂ˝za nejvĂ˝Ĺˇe 3 expiracĂ­; koncentrace se mÄ›Ĺ™Ă­ po jednotlivĂ˝ch expiracĂ­ch."""
    today = today or datetime.now(ZoneInfo("America/New_York")).date()
    empty = {
        "usable": False, "coverage_complete": False, "call_oi": None, "put_oi": None,
        "call_volume": None, "put_volume": None, "put_call_oi": None,
        "put_call_volume": None, "entry_block": False, "put_attention": False,
        "expiry_risk": False, "concentration": [], "top_strikes": [],
        "expiries": list(expected_expiries), "status": "OpÄŤnĂ­ data nejsou dostupnĂˇ.",
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
    # VĂ˝raznĂˇ koncentrace OI v nejbliĹľĹˇĂ­ch 7 dnech tÄ›snÄ› nad cenou mĹŻĹľe omezit vstup.
    # Nejde o prokĂˇzanou rezistenci ani pĹ™edpovÄ›ÄŹ ceny ÄŤi smÄ›ru hedgingu.
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
    status = ("DostateÄŤnĂ˝ vzorek OI pro doplĹkovĂ˝ filtr." if usable else
              "NeĂşplnĂ© Ăşdaje OI nebo pĹ™Ă­liĹˇ malĂ˝ vzorek; filtr nenĂ­ pouĹľit.")
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
                errors.append(f"Expirace {expiry} se nepodaĹ™ila naÄŤĂ­st.")
    except Exception as e:
        log.warning("OpÄŤnĂ­ expirace %s: %s", ticker, e)
        errors.append("Seznam opÄŤnĂ­ch expiracĂ­ se nepodaĹ™ilo naÄŤĂ­st.")
    frames = [f for f in frames if not f.empty]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = summarize_options(rows, price, atr, selected, today)
    if not selected and not errors:
        result["status"] = "Yahoo neposkytuje expirace v rozsahu 0â€“60 dnĹŻ; filtr nenĂ­ pouĹľit."
    if errors:
        result.update(usable=False, entry_block=False, put_attention=False,
                      status="ÄŚĂˇst opÄŤnĂ­ch dat nenĂ­ dostupnĂˇ; filtr nenĂ­ pouĹľit.")
    result.update(errors=errors, fetched_at=datetime.now(timezone.utc).isoformat())
    return result


def options_are_current(context, now=None):
    """Kontroluje stĂˇĹ™Ă­ naÄŤtenĂ­, nikoli neznĂˇmĂ© datum OI od poskytovatele."""
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
    """Opce mohou vstup omezit; nikdy nepovolĂ­ vstup, kterĂ˝ odmĂ­tla technickĂˇ analĂ˝za."""
    r = dict(entry)
    r["technical_buy_allowed"] = r["buy_allowed"]
    r["options_context"] = context
    checks = dict(r["entry_checks"])
    if context.get("usable") and options_are_current(context):
        checks["Bez blĂ­zkĂ© koncentrace CALL OI pĹ™ed expiracĂ­"] = not context["entry_block"]
    r["entry_checks"] = checks
    r["buy_allowed"] = all(checks.values())
    r["entry_score"] = round(100 * sum(checks.values()) / len(checks), 1)
    if not r["buy_allowed"]:
        if r["technical_buy_allowed"]:
            r["signal"] = "NEVSTUPOVAT Â· OPÄŚNĂŤ KONCENTRACE"
        r["entry_reason"] = "NesplnÄ›no: " + "; ".join(k for k, ok in checks.items() if not ok)
    if not context.get("usable"):
        r["entry_reason"] += " OpÄŤnĂ­ Ăşdaje chybĂ­ nebo nejsou ĂşplnĂ©; opÄŤnĂ­ filtr se nepouĹľil."
    r["strategy_version"] = "4.7"
    return r


def assess_trade_action(item: dict, held=False, position_stop=None, position_target=None) -> dict:
    """OddÄ›lenÄ› novĂ˝ vstup a sprĂˇva existujĂ­cĂ­ long pozice, vĹľdy z uzavĹ™enĂ©ho dne.

    VlastnĂ­ stop/cĂ­l mĂˇ pĹ™ednost. NezamÄ›Ĺujeme novĂ˝ modelovĂ˝ stop s plĂˇnem drĹľitele.
    SamotnĂ© OI nikdy nevyvolĂˇ prodej; vyĹľadujeme souÄŤasnÄ› cenovĂ© oslabenĂ­.
    """
    result = {"trend_label": "BĂťÄŚĂŤ" if item["trend_ok"] else "NEPOTVRZENĂť / SLABĂť"}
    if not held:
        opts = item.get("options_context", {})
        if opts.get("usable") and not options_are_current(opts):
            return {**result, "action_label": "ÄŚEKAT / NEVSTUPOVAT",
                    "action_reason": "OpÄŤnĂ­ snapshot je starĹˇĂ­ neĹľ 24 hodin. SpusĹĄte novĂ˝ sken."}
        return {**result, "action_label": "NAKUPOVAT" if item["buy_allowed"] else "ÄŚEKAT / NEVSTUPOVAT",
                "action_reason": item["entry_reason"]}
    price = item["price"]
    stop, target = safe_float(position_stop, 0), safe_float(position_target, 0)
    if stop > 0 and target > 0 and target <= stop:
        return {**result, "action_label": "OVÄšĹIT PLĂN", "action_reason": "U long pozice musĂ­ bĂ˝t cĂ­l nad stopem."}
    if stop > 0 and price <= stop:
        label, reason = "PRODAT", "UzavĹ™enĂˇ cena je na vaĹˇem stopu nebo pod nĂ­m."
    elif target > 0 and price >= target:
        label, reason = "PRODAT", "UzavĹ™enĂˇ cena dosĂˇhla vaĹˇeho cĂ­le nebo jej pĹ™ekonala."
    elif str(item.get("structure_event", "")).endswith("dolĹŻ"):
        label, reason = "PRODAT / ZVĂĹ˝IT REDUKCI", "PotvrzenĂ˝ prĹŻlom swingovĂ© struktury dolĹŻ."
    else:
        opts = item.get("options_context", {})
        weak = price < item["ema50"] and item["macd_hist"] < 0 and not item["trend_ok"]
        option_risk = opts.get("usable") and options_are_current(opts) and (
            opts.get("entry_block") or opts.get("put_attention"))
        if weak and option_risk:
            label = "PRODAT / ZVĂĹ˝IT REDUKCI"
            reason = ("Cena pod EMA50 a zĂˇpornĂ˝ MACD potvrzujĂ­ oslabenĂ­; souÄŤasnÄ› je pĹ™Ă­tomna "
                      "opÄŤnĂ­ koncentrace nebo pĹ™evaha PUT OI. Jde o konzervativnĂ­ heuristiku, "
                      "nikoli dĹŻkaz prodejĹŻ institucĂ­.")
        else:
            label = "DRĹ˝ET / SLEDOVAT"
            reason = "NenĂ­ splnÄ›no vĂ˝stupnĂ­ pravidlo podle uzavĹ™enĂ©ho dne."
            if weak:
                reason += " Technika slĂˇbne; zkontrolujte vlastnĂ­ stop."
            if option_risk:
                reason += " OpÄŤnĂ­ data zvyĹˇujĂ­ pozornost, samotnĂˇ vĹˇak prodej nevyvolajĂ­."
    if not (stop > 0 or target > 0):
        reason += " VlastnĂ­ stop ani cĂ­l nebyl zadĂˇn."
    return {**result, "action_label": label, "action_reason": reason}


def render_events_panel(ticker, events):
    st.markdown("#### đź—“ď¸Ź Dividendy a firemnĂ­ vĂ˝sledky")
    def countdown(day):
        n = days_until(day)
        return "NedostupnĂ© / neoznĂˇmeno" if n is None else ("Dnes" if n == 0 else f"Za {n} dnĹŻ")
    c1, c2, c3 = st.columns(3)
    c1.metric("Do vĂ˝platy dividendy", countdown(events.get("payment_date")))
    c1.caption(events.get("payment_date") or "BudoucĂ­ datum vĂ˝platy nenĂ­ dostupnĂ©.")
    c2.metric("Do ex-dividend dne", countdown(events.get("ex_dividend_date")))
    c2.caption(events.get("ex_dividend_date") or "BudoucĂ­ ex-dividend den nenĂ­ dostupnĂ˝.")
    first, last = events.get("earnings_start"), events.get("earnings_end")
    if first and last and first != last:
        n1, n2 = days_until(first), days_until(last)
        value = f"Za {n1}â€“{n2} dnĹŻ" if n1 is not None else ("V aktuĂˇlnĂ­m rozpÄ›tĂ­" if n2 is not None else "NedostupnĂ© / neoznĂˇmeno")
        c3.metric("Do vĂ˝sledkĹŻ (rozpÄ›tĂ­)", value)
        c3.caption(f"{first} aĹľ {last}")
    else:
        c3.metric("Do vyhlĂˇĹˇenĂ­ vĂ˝sledkĹŻ", countdown(first))
        c3.caption(first or "BudoucĂ­ datum vĂ˝sledkĹŻ nenĂ­ dostupnĂ©.")
    st.caption("PoÄŤty jsou kalendĂˇĹ™nĂ­ dny podle Europe/Prague. Datum vĂ˝sledkĹŻ mĹŻĹľe bĂ˝t odhadem Yahoo; "
               "ÄŤas zveĹ™ejnÄ›nĂ­ zde nenĂ­ potvrzen. VĂ˝plata a ex-dividend den jsou rĹŻznĂ© udĂˇlosti.")
    c4, c5 = st.columns(2)
    currency = events.get("currency", "mÄ›na neuvedena")
    amount = events.get("last_dividend_amount")
    c4.metric("PoslednĂ­ znĂˇmĂˇ dividenda / akcii", f"{amount:.4f} {currency}" if amount is not None else "NedostupnĂ©")
    c4.caption(f"HistorickĂ˝ ex-dividend den: {events.get('last_dividend_ex_date') or 'neuveden'}. "
               "ÄŚĂˇstka nenĂ­ potvrzenĂ­m pĹ™Ă­ĹˇtĂ­ vĂ˝platy ani roÄŤnĂ­ dividendou.")
    confirmed = st.checkbox("ZnĂˇm potvrzenou ÄŤĂˇstku pĹ™Ă­ĹˇtĂ­ vĂ˝platy z oznĂˇmenĂ­ emitenta", key=f"div_confirmed_{ticker}")
    if confirmed:
        manual = st.number_input("PotvrzenĂˇ dividenda za jednu vĂ˝platu / akcii", min_value=0.0,
                                 value=0.0, step=0.01, format="%.4f", key=f"div_amount_{ticker}")
        c5.metric("PĹ™Ă­ĹˇtĂ­ dividenda / akcii â€” ruÄŤnÄ›", f"{manual:.4f} {currency}")
    else:
        c5.metric("PĹ™Ă­ĹˇtĂ­ dividenda / akcii", "ÄŚĂˇstka nepotvrzena")
    n = days_until(first)
    if (n is not None and n <= 7) or (first and last and event_date(first) <= local_today() <= event_date(last)):
        st.warning("VĂ˝sledky jsou blĂ­zko nebo v aktuĂˇlnĂ­m rozpÄ›tĂ­. HrozĂ­ cenovĂ˝ gap; jde o upozornÄ›nĂ­, ne automatickou blokaci vstupu.")
    for error in events.get("errors", []):
        st.caption(error)
    st.caption(f"KalendĂˇĹ™ naÄŤten: {events.get('fetched_at', 'neuvedeno')} (UTC). Pro aktuĂˇlnĂ­ data spusĹĄte sken.")


def render_options_panel(context):
    st.markdown("#### đź‹ OpÄŤnĂ­ zĂˇjem â€” open interest a koncentrace")
    st.write(context.get("status", "OpÄŤnĂ­ data nejsou dostupnĂˇ."))
    if not context.get("usable"):
        st.warning("OpÄŤnĂ­ filtr nebyl pouĹľit. PĹ™Ă­padnĂ˝ nĂˇkupnĂ­ signĂˇl vychĂˇzĂ­ jen z technickĂ˝ch podmĂ­nek.")
    def number(key):
        v = context.get(key)
        return f"{v:,.0f}" if v is not None else "NedostupnĂ©"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CALL open interest", number("call_oi"))
    c2.metric("PUT open interest", number("put_oi"))
    pcr, vcr = context.get("put_call_oi"), context.get("put_call_volume")
    c3.metric("PUT / CALL â€” OI", f"{pcr:.2f}" if pcr is not None else "NedostupnĂ©")
    c4.metric("PUT / CALL â€” objem", f"{vcr:.2f}" if vcr is not None else "NedostupnĂ©")
    st.caption(f"CALL objem: {number('call_volume')} Â· PUT objem: {number('put_volume')}. "
               "OI = poÄŤet otevĹ™enĂ˝ch kontraktĹŻ, objem = zobchodovanĂ© kontrakty; objem/OI neprokazuje novĂ© pozice.")
    st.caption("Vzorek: nejvĂ˝Ĺˇe 3 nejbliĹľĹˇĂ­ expirace do 60 dnĹŻ, standardnĂ­ kontrakty, "
               "strike Â±20 % od signĂˇlnĂ­ ceny. Nejde o celĂ˝ opÄŤnĂ­ trh. "
               "PUT/CALL neodhaluje nĂˇkup/prodej, ĂşÄŤastnĂ­ka ani jeho zĂˇmÄ›r; zahrnuje i zajiĹˇtÄ›nĂ­ a spready.")
    if context.get("concentration"):
        st.write("**BlĂ­zkĂ© koncentrace CALL OI pĹ™ed expiracĂ­:**")
        for level in context["concentration"]:
            st.write(f"Strike {level['strike']:.2f} Â· expirace {level['expiry']} Â· "
                     f"OI {level['oi']:,} Â· {level['share']:.0%} CALL OI tĂ©to expirace ve vzorku.")
    if context.get("top_strikes"):
        frame = pd.DataFrame(context["top_strikes"]).rename(columns={
            "expiry": "Expirace", "side": "Typ", "strike": "Strike", "openInterest": "OI", "volume": "Objem"})
        st.dataframe(frame, hide_index=True, use_container_width=True)
    if context.get("entry_block"):
        st.warning("OpÄŤnĂ­ filtr omezuje novĂ˝ vstup kvĹŻli blĂ­zkĂ© koncentraci pĹ™ed expiracĂ­. "
                   "Koncentrace nenĂ­ prokĂˇzanĂˇ cenovĂˇ rezistence.")
    if context.get("put_attention"):
        st.info("PUT/CALL OI â‰Ą1,5: upozornÄ›nĂ­ na sloĹľenĂ­ pozic; samo o sobÄ› nenĂ­ medvÄ›dĂ­ signĂˇl.")
    for error in context.get("errors", []):
        st.caption(error)
    st.caption(f"NaÄŤteno: {context.get('fetched_at', 'neuvedeno')} (UTC). "
               "ÄŚas naÄŤtenĂ­ nenĂ­ datem samotnĂ©ho OI; jeho pĹ™esnĂ© stĂˇĹ™Ă­ Yahoo nezaruÄŤuje. "
               "OpÄŤnĂ­ snapshot a dennĂ­ signĂˇl nemusejĂ­ pochĂˇzet ze stejnĂ©ho okamĹľiku.")
    if not options_are_current(context):
        st.warning("NaÄŤtenĂ˝ opÄŤnĂ­ kontext je starĹˇĂ­ neĹľ 24 hodin nebo nemĂˇ platnĂ˝ ÄŤas. SpusĹĄte novĂ˝ sken.")


# ---------------------- TRĹ˝NĂŤ REĹ˝IM ---------------------------
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


# ---------------------- SKĂ“ROVĂNĂŤ -----------------------------
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
        log.warning("SUPABASE_URL/KEY chybĂ­ v secrets.toml â€“ historie vypnuta.")
        return None
    except Exception as e:
        log.warning(f"Supabase nenĂ­ pĹ™ipojeno: {e}")
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
        # NovĂ© sloupce do existujĂ­cĂ­ Supabase tabulky nepĹ™idĂˇvĂˇme bez migrace.
        # Detail opcĂ­, sezĂłnnosti, kalendĂˇĹ™ a rozhodnutĂ­ drĹľitele jsou v UI/CSV, ne v DB.
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
        log.info(f"UloĹľeno {len(payloads)} signĂˇlĹŻ.")
        return len(payloads)
    except Exception as e:
        log.error(f"save_signals_bulk selhalo: {e}", exc_info=True)
        return 0


def load_learning_stats(sb, days: int = 90):
    """OrientaÄŤnĂ­ simulace pouze signĂˇlĹŻ 4.7, vstup na pĹ™Ă­ĹˇtĂ­m open, max. 5 dnĹŻ.

    NenĂ­ to walk-forward backtest ani uÄŤenĂ­ parametrĹŻ. Gap mimo zĂłnu = bez vstupu.
    DennĂ­ OHLC neznĂˇ poĹ™adĂ­ SL/TP; takovĂ˝ obchod se nezaĹ™adĂ­ mezi vĂ˝hry/prohry.
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
            # auto_adjust mĹŻĹľe po dividendÄ›/splitu pĹ™epoÄŤĂ­tat historii.
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
    entry = apply_options_filter(entry, get_options_context(ticker, float(x["Close"]), float(x["ATR14"])))
    entry = apply_seasonality_filter(entry, get_seasonality_context(
        ticker, last_day.isoformat(), STRATEGY["holding_days"]))
    events = get_corporate_events(ticker)
    if events.get("last_dividend_amount") is None and "Dividends" in d:
        dividends = d.loc[d["Dividends"] > 0, "Dividends"]
        if not dividends.empty:
            events = {**events, "last_dividend_amount": float(dividends.iloc[-1]),
                      "last_dividend_ex_date": pd.Timestamp(dividends.index[-1]).date().isoformat()}
    patterns = detect_patterns(d)
    structure = analyze_market_structure(d)

    return {
        "ticker":       ticker,
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


# ---------------------- PARALELNĂŤ SCAN ------------------------
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
    <h3>đź“ Spot Scanner 4.7</h3>
    <p>TechnickĂˇ analĂ˝za â€˘ SezĂłnnost kaĹľdĂ© akcie â€˘ OpÄŤnĂ­ open interest â€˘ Dividendy â€˘ VĂ˝sledky â€˘ Risk management</p>
</div>
""", unsafe_allow_html=True)

sb = get_supabase()

# ---------------------- SIDEBAR -------------------------------
with st.sidebar:
    st.header("âš™ď¸Ź NastavenĂ­")

    custom = st.text_input("PĹ™idat ticker", "").upper().strip()
    tickers = DEFAULT_TICKERS.copy()
    if custom:
        if is_valid_ticker(custom):
            if custom not in tickers:
                tickers.insert(0, custom)
        else:
            st.warning("NeplatnĂ˝ ticker (Aâ€“Z, 0â€“9, teÄŤka, pomlÄŤka; max 10 znakĹŻ).")

    st.markdown("---")
    st.subheader("đź’° Risk management")
    capital          = st.number_input("KapitĂˇl ($)", min_value=100.0, value=5000.0, step=500.0)
    risk_pct         = st.slider("Riziko na obchod (%)", 0.25, 3.0, 1.0, 0.25)
    max_position_pct = st.slider("Max. velikost pozice (%)", 5, 100, 25, 5)

    st.markdown("---")
    st.subheader("đź”Ž Filtry")
    min_quality        = st.slider("Min. Quality Score", 0, 100, 60)
    only_buy_zone      = st.checkbox("Pouze NĂKUPNĂŤ ZĂ“NA", False)
    only_positive_rs   = st.checkbox("Pouze RS > S&P 500", False)
    exclude_overbought = st.checkbox("VylouÄŤit RSI > 75", True)
    only_breakout = st.checkbox("Pouze cenovĂ© prĹŻrazy", False)
    only_squeeze = st.checkbox("Pouze BB squeeze", False)
    only_volume_spike = st.checkbox("Pouze zvĂ˝ĹˇenĂ˝ objem", False)
    max_workers        = st.slider("ParalelnĂ­ vlĂˇkna", 2, 12, 4)

    st.markdown("---")
    st.caption("Zdroj dat: Yahoo Finance; pouze pĹ™edchozĂ­ dokonÄŤenĂ© dennĂ­ svĂ­ÄŤky.")
    st.caption("Support: â‰Ą2 oddÄ›lenĂ© testy. Vstup nejvĂ˝Ĺˇe 0,5 ATR a 1,5 % nad nĂ­m. "
               "R:R â‰Ą1,5; horizont 5 obchodnĂ­ch dnĹŻ. Parametry v STRATEGY.")
    st.caption("OpÄŤnĂ­ filtr: dostateÄŤnĂ˝ vzorek â‰Ą5 000 OI; CALL strike do 0,5 ATR nad cenou, "
               "â‰Ą1 000 OI a â‰Ą20 % CALL OI danĂ© expirace, expirace do 7 kalendĂˇĹ™nĂ­ch dnĹŻ. "
               "Jde o nevalidovanou rizikovou heuristiku. ChybÄ›jĂ­cĂ­ opce se neberou jako potvrzenĂ­.")
    st.caption("SezĂłnnost: stejnĂ© datum v poslednĂ­ch 10 uzavĹ™enĂ˝ch letech, horizont podle STRATEGY "
               "(vĂ˝chozĂ­ 5 obchodnĂ­ch dnĹŻ), nejmĂ©nÄ› 5 roÄŤnĂ­ch vzorkĹŻ. OpakovanÄ› nepĹ™Ă­znivĂ© obdobĂ­ "
               "blokuje novĂ˝ nĂˇkup. PĹ™Ă­znivĂ© obdobĂ­ podporuje technicky a opÄŤnÄ› povolenĂ˝ vstup. "
               "ChybÄ›jĂ­cĂ­ historie se oznaÄŤĂ­; sezĂłnnĂ­ filtr se tehdy nepouĹľije.")
    st.caption("Historie sezĂłnnosti je uloĹľena do dennĂ­ cache; prvnĂ­ sken mĹŻĹľe trvat dĂ©le.")
    st.caption("Dividendy a vĂ˝sledky se naÄŤĂ­tajĂ­ pĹ™i skenu. VĂ­ce poĹľadavkĹŻ mĹŻĹľe sken zpomalit.")
    st.caption("AnalytickĂˇ pomĹŻcka â€“ ne automatickĂ˝ obchodnĂ­ systĂ©m.")

# ---------------------- TABS --------------------------------
tab_scan, tab_history, tab_learning = st.tabs([
    "đź”Ž Scanner", "đź—„ď¸Ź Historie signĂˇlĹŻ", "đź§  UÄŤĂ­cĂ­ se pĹ™ehled"
])

if "results" not in st.session_state:
    st.session_state.results = []
elif st.session_state.results and st.session_state.results[0].get("strategy_version") != "4.7":
    st.session_state.results = []
if "regime" not in st.session_state:
    st.session_state.regime = None
if "regime_score" not in st.session_state:
    st.session_state.regime_score = 0

# ============================================================
# TAB 1 â€“ SCANNER
# ============================================================
with tab_scan:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.subheader("TrĹľnĂ­ sken")
    with col_b:
        run = st.button("đźš€ SPUSTIT SKEN PĹEDNASTAVENĂťCH", type="primary", use_container_width=True)

    if run:
        progress_bar = st.progress(0.0, text="Analyzuji trhâ€¦")

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
            st.toast(f"UloĹľeno {saved} signĂˇlĹŻ do Supabase.", icon="đź’ľ")

    # Zobraz reĹľim trhu (pĹ™etrvĂˇvĂˇ mezi reruny)
    if st.session_state.regime:
        color = {"BULLISH": "đźź˘", "BEARISH": "đź”´",
                 "NEUTRAL": "đźźˇ", "UNKNOWN": "âšŞ"}.get(st.session_state.regime, "âšŞ")
        st.info(f"{color} **TrĹľnĂ­ reĹľim (SPY):** {st.session_state.regime} â€” "
                f"skĂłre {st.session_state.regime_score}/5")

    results = st.session_state.results

    if results:
        buy_count   = sum(r["signal"] == BUY_SIGNAL for r in results)
        avg_quality = np.mean([r["quality_score"] for r in results])
        avg_entry   = np.mean([r["entry_score"]   for r in results])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AnalyzovanĂ˝ch",   len(results))
        c2.metric("NĂKUPNĂŤ ZĂ“NA",    buy_count)
        c3.metric("PrĹŻmÄ›r Quality",  f"{avg_quality:.1f}")
        c4.metric("PrĹŻmÄ›r Entry",    f"{avg_entry:.1f}")

        rows = []
        for r in results:
            assessment = assess_trade_action(r, held=st.session_state.get(f"held_{r['ticker']}", False),
                                            position_stop=st.session_state.get(f"held_stop_{r['ticker']}"),
                                            position_target=st.session_state.get(f"held_target_{r['ticker']}"))
            rows.append({
                "Ticker":   r["ticker"],
                "Cena":     round(r["price"], 2),
                "SignĂˇl":   r["signal"],
                "Quality":  r["quality_score"],
                "Entry":    r["entry_score"],
                "ZĂ“NA OD":  round(r["zone_low"], 2),
                "ZĂ“NA DO":  round(r["zone_high"], 2),
                "Vstup":    round(r["preferred_entry"], 2),
                "Stop":     round(r["stop"], 2),
                "TP1":      round(r["target1"], 2),
                "R:R":      round(r["rr1"], 2),
                "Support": r["support"],
                "Od supportu (ATR)": round(r["support_distance_atr"], 2),
                "Testy supportu": r["support_touches"],
                "DĹŻvod": r["entry_reason"],
                "Max. vstup (plĂˇn)": r["max_execution_price"],
                "Obrat 20d (mediĂˇn)": r["median_dollar_volume"],
                "Gap / ATR": r["signal_gap_atr"],
                "Konfluence": ", ".join(r["confluence"]),
                "Datum dat": str(r["data"].index[-1].date()),
                "RS 30d":   r["rs_30d"],
                "RSI":      round(r["rsi"], 1),
                "Objem":    round(r["volume_ratio"], 2),
                "PrĹŻraz":   r["breakout"],
                "Squeeze":  r["squeeze"],
                "ObjemovĂ˝ spike": r["volume_spike"],
                "Struktura trhu": r["structure_trend"],
                "PrĹŻlom struktury": r["structure_event"],
                "Trend": assessment["trend_label"],
                "OrientaÄŤnĂ­ akce": assessment["action_label"],
                "DĹŻvod akce": assessment["action_reason"],
                "TechnickĂ˝ vstup pĹ™ed opÄŤnĂ­m filtrem": r["technical_buy_allowed"],
                "OpÄŤnĂ­ filtr blokuje": r["options_context"]["entry_block"],
                "PUT/CALL OI": r["options_context"]["put_call_oi"],
                "OpÄŤnĂ­ data naÄŤtena": r["options_context"]["fetched_at"],
                "Vstup pĹ™ed sezĂłnnĂ­m filtrem": r["pre_seasonality_buy_allowed"],
                "SezĂłnnost": r["seasonality_context"]["status"],
                "SezĂłnnĂ­ filtr pouĹľit": r["seasonality_context"]["usable"],
                "SezĂłnnĂ­ filtr blokuje": r["seasonality_context"]["entry_block"],
                "SezĂłnnĂ­ horizont (obchodnĂ­ dny)": r["seasonality_context"]["holding_days"],
                "SezĂłnnĂ­ roÄŤnĂ­ vzorky": r["seasonality_context"]["sample_years"],
                "SezĂłnnĂ­ mediĂˇn (%)": r["seasonality_context"]["median_pct"],
                "SezĂłnnĂ­ prĹŻmÄ›r (%)": r["seasonality_context"]["mean_pct"],
                "SezĂłnnĂ­ kladnĂ© pohyby (%)": r["seasonality_context"]["positive_pct"],
                "SezĂłnnĂ­ mediĂˇn poslednĂ­ch vzorkĹŻ (%)": r["seasonality_context"]["recent_median_pct"],
                "SezĂłnnĂ­ referenÄŤnĂ­ datum": r["seasonality_context"]["as_of_date"],
                "SezĂłnnĂ­ dĹŻvod": r["seasonality_context"]["reason"],
                "VĂ˝plata dividendy": r["corporate_events"].get("payment_date"),
                "Dny do dividendy": days_until(r["corporate_events"].get("payment_date")),
                "PoslednĂ­ dividenda/akcii (historickĂˇ)": r["corporate_events"].get("last_dividend_amount"),
                "PĹ™Ă­ĹˇtĂ­ dividenda/akcii (potvrzenĂˇ ruÄŤnÄ›)": (
                    st.session_state.get(f"div_amount_{r['ticker']}")
                    if st.session_state.get(f"div_confirmed_{r['ticker']}", False) else None),
                "MÄ›na dividendy": r["corporate_events"].get("currency"),
                "VĂ˝sledky od": r["corporate_events"].get("earnings_start"),
                "VĂ˝sledky do": r["corporate_events"].get("earnings_end"),
                "Dny do zaÄŤĂˇtku vĂ˝sledkĹŻ": days_until(r["corporate_events"].get("earnings_start")),
            })
        df_show = pd.DataFrame(rows)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        st.download_button(
            "đź“Ą StĂˇhnout CSV",
            df_show.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"klondike_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        st.markdown("### đź“Š Podrobnosti jednotlivĂ˝ch instrumentĹŻ")
        st.caption("Rozbalte instrument. KaĹľdĂ˝ panel obsahuje jen jeho vlastnĂ­ metriky, signĂˇl a grafy.")

        for item in results:
            ticker = item["ticker"]
            assessment = assess_trade_action(item, held=st.session_state.get(f"held_{ticker}", False),
                                             position_stop=st.session_state.get(f"held_stop_{ticker}"),
                                             position_target=st.session_state.get(f"held_target_{ticker}"))
            with st.expander(
                f"{ticker} Â· {assessment['action_label']} Â· trend {assessment['trend_label']} Â· "
                f"{item['signal']}",
                expanded=False,
            ):
                st.subheader(f"{ticker} Â· {item['signal']}")
                held = st.checkbox("Tuto akcii uĹľ drĹľĂ­m â€” hodnotit existujĂ­cĂ­ long pozici", key=f"held_{ticker}")
                own_stop, own_target = None, None
                if held:
                    ps, pt = st.columns(2)
                    own_stop = ps.number_input("VlastnĂ­ stop pozice (0 = nezadĂˇn)", min_value=0.0,
                                               value=0.0, step=0.01, key=f"held_stop_{ticker}")
                    own_target = pt.number_input("VlastnĂ­ cĂ­l pozice (0 = nezadĂˇn)", min_value=0.0,
                                                 value=0.0, step=0.01, key=f"held_target_{ticker}")
                    st.caption("Stop/cĂ­l zadejte ve stejnĂ© mÄ›nÄ› a cenovĂ©m zĂˇkladu jako zobrazenĂˇ cena. "
                               "HodnocenĂ­ porovnĂˇvĂˇ uzavĹ™enou cenu, nikoli intradennĂ­ zĂˇsah pokynu. "
                               "Volba pozice a ruÄŤnĂ­ Ăşdaje platĂ­ v tĂ©to relaci aplikace.")
                assessment = assess_trade_action(item, held, own_stop, own_target)
                action_message = (
                    f"**OrientaÄŤnĂ­ akce: {assessment['action_label']}** â€” "
                    f"{assessment['action_reason']}"
                )
                if assessment["action_label"] in {"NAKUPOVAT", "POMALU DOKUPOVAT"}:
                    st.success(action_message)
                elif assessment["action_label"] in {"ÄŚEKAT / NEVSTUPOVAT", "DRĹ˝ET / SLEDOVAT"}:
                    st.info(action_message)
                elif assessment["action_label"] in {"PRODAT / ZVĂĹ˝IT REDUKCI", "OVÄšĹIT PLĂN"}:
                    st.warning(action_message)
                else:
                    st.error(action_message)
                st.info(f"**Trend: {assessment['trend_label']}** Â· {item['structure_event']}")
                st.caption("AutomatickĂ© technickĂ© vyhodnocenĂ­; zohlednÄ›te vlastnĂ­ strategii a riziko.")
                st.caption(
                    f"Cena ${item['price']:.2f} Â· VstupnĂ­ zĂłna ${item['zone_low']:.2f}â€“"
                    f"${item['zone_high']:.2f} Â· PreferovanĂ˝ vstup ${item['preferred_entry']:.2f}"
                )

                st.caption(f"SignĂˇlnĂ­ den: {item['data'].index[-1].date()}. "
                           "DneĹˇnĂ­ dennĂ­ svĂ­ÄŤka je vynechĂˇna. Cena nenĂ­ ĹľivĂˇ kotace.")
                st.write(f"Support: {item['support_touches']} potvrzenĂ˝ch testĹŻ; "
                         f"vzdĂˇlenost {item['support_distance_atr']:.2f} ATR "
                         f"({item['support_distance_pct']:.2f} %).")
                st.dataframe(pd.DataFrame([
                    {"PodmĂ­nka": label, "SplnÄ›no": ok}
                    for label, ok in item["entry_checks"].items()
                ]), hide_index=True, use_container_width=True)
                render_events_panel(ticker, item["corporate_events"])
                render_options_panel(item["options_context"])
                render_seasonality_panel(ticker, item["seasonality_context"])
                st.markdown("#### đź§­ Struktura trhu")
                st.write(f"**{item['structure_trend']}** Â· {item['structure_event']}")
                st.caption(
                    f"PoslednĂ­ potvrzenĂ© swingovĂ© ĂşrovnÄ›: maximum ${item['swing_high']:.2f} Â· "
                    f"minimum ${item['swing_low']:.2f}. Swingy se potvrzujĂ­ aĹľ po dalĹˇĂ­ch svĂ­ÄŤkĂˇch."
                )

                # SĂ­ĹĄovĂ© doplĹky se naÄŤĂ­tajĂ­ jen po kliknutĂ­ v konkrĂ©tnĂ­ ticker zĂˇloĹľce.
                if st.button("NaÄŤĂ­st nĂˇzev, pre-market a zprĂˇvy", key=f"load_context_{ticker}"):
                    st.session_state[f"context_loaded_{ticker}"] = True
                if st.session_state.get(f"context_loaded_{ticker}", False):
                    name, sector = basic_info(ticker)
                    pre_price, pre_change = premarket(ticker)
                    st.markdown(f"**{name}** Â· Sektor: {sector}")
                    if pre_price is not None:
                        st.caption(f"Pre-market: ${pre_price:.2f} ({pre_change:+.2f} %)")
                    news = get_news_context(ticker)
                    st.markdown("#### đź“° Kontext zprĂˇv")
                    st.caption(
                        f"{news['sentiment']} (orientaÄŤnĂ­ skĂłre titulkĹŻ: {news['score']:+d}). "
                        "Jde o jednoduchĂ© klĂ­ÄŤovĂ© frĂˇze, ne o porozumÄ›nĂ­ vĂ˝znamu ÄŤlĂˇnku."
                    )
                    if news["articles"]:
                        for article_index, article in enumerate(news["articles"]):
                            st.write(f"**{article['title']}**")
                            meta = " Â· ".join(
                                part for part in [article["source"], article["published"]] if part
                            )
                            if meta:
                                st.caption(meta)
                            if article["summary"]:
                                st.write(article["summary"])
                            if article["url"].startswith(("https://", "http://")):
                                st.link_button(
                                    "OtevĹ™Ă­t ÄŤlĂˇnek", article["url"],
                                    key=f"news_{ticker}_{article_index}"
                                )
                    else:
                        st.info("Pro tento ticker nejsou dostupnĂ© zprĂˇvy.")

                d = item["data"].tail(180)
                st.markdown("#### đź“Š KlĂ­ÄŤovĂ© metriky")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Quality", f"{item['quality_score']:.1f}/100")
                k2.metric("Entry", f"{item['entry_score']:.1f}/100")
                k3.metric("Trend", f"{item['trend_score']:.1f}/100")
                k4.metric("Momentum", f"{item['momentum_score']:.1f}/100")
                k5, k6, k7, k8 = st.columns(4)
                k5.metric("RS vs SPY (30d)", f"{item['rs_30d']:+.2f} %")
                k6.metric("RSI (14)", f"{item['rsi']:.1f}")
                k7.metric("ATR", f"${item['atr']:.2f} ({item['atr_pct']:.2f} %)")
                k8.metric("Objem vs prĹŻmÄ›r", f"{item['volume_ratio']:.2f}Ă—")

                st.markdown("#### đź§° TraderskĂ˝ plĂˇn")
                t1, t2, t3 = st.columns(3)
                t1.metric("Max. vstup podle plĂˇnu", f"${item['max_execution_price']:.4f}")
                t2.metric("Obrat 20d â€“ mediĂˇn", f"{item['median_dollar_volume']/1e6:.1f} mil.")
                t3.metric("Objem odrazu / pĹ™edchozĂ­ch 20 dnĹŻ", f"{item['bounce_volume_ratio']:.2f}Ă—")
                st.caption("Obrat = upravenĂˇ cena Ă— objem, orientaÄŤnÄ› v mÄ›nÄ› kotace (u US tickerĹŻ USD). "
                           "NenĂ­ to mÄ›Ĺ™enĂ­ spreadu ani hloubky trhu. Maximum vstupu nenĂ­ pokyn k nĂˇkupu.")
                st.write("**SoubÄ›h se supportem:** " + (", ".join(item["confluence"]) or "bez blĂ­zkĂ© EMA20/EMA50"))
                st.write("**Korekce na klesajĂ­cĂ­m objemu:** " + ("ano" if item["quiet_pullback"] else "nepotvrzena"))
                st.caption("Konfluence a objem jsou doplĹkovĂ© informace; nenahrazujĂ­ povinnĂ© podmĂ­nky.")
                st.write(f"**PlĂˇn od signĂˇlnĂ­ho vstupu:** stop ${item['stop']:.2f}; "
                         f"TP1 ${item['target1']:.2f}; ĂşroveĹ +1R ${item['one_r_price']:.2f}; "
                         f"ÄŤasovĂ˝ vĂ˝stup nejpozdÄ›ji na konci {STRATEGY['holding_days']}. obchodnĂ­ho dne vÄŤetnÄ› vstupnĂ­ho.")
                st.caption("+1R je orientaÄŤnĂ­ milnĂ­k, ne automatickĂ˝ pĹ™esun stopu. "
                           "PĹ™ed objednĂˇvkou ovÄ›Ĺ™te aktuĂˇlnost kalendĂˇĹ™e a opÄŤnĂ­ch dat, zprĂˇvy, spread a graf. "
                           "KalendĂˇĹ™ se naÄŤĂ­tĂˇ automaticky; aplikace pokyny neprovĂˇdĂ­.")
                proposed = st.number_input("ZkuĹˇebnĂ­ vstupnĂ­ cena (ruÄŤnÄ›, nenĂ­ ĹľivĂˇ kotace)",
                                           min_value=0.01, value=max(0.01, float(item["price"])),
                                           step=0.01, format="%.4f", key=f"proposed_{ticker}")
                price_ok, proposed_rr = check_execution_price(item, proposed)
                st.write(f"R:R pĹ™i tĂ©to cenÄ›, se stejnĂ˝m stopem a TP1: {proposed_rr:.2f}")
                if price_ok:
                    st.success("ZadanĂˇ cena vyhovuje uloĹľenĂ©mu plĂˇnu. AktuĂˇlnĂ­ trĹľnĂ­ situace nebyla znovu ovÄ›Ĺ™ena.")
                else:
                    st.warning("ZadanĂˇ cena nebo pĹŻvodnĂ­ signĂˇl nesplĹuje plĂˇn: nevstupovat podle tohoto vĂ˝poÄŤtu.")
                st.caption("Pro novĂ˝ signĂˇlnĂ­ den spusĹĄte novĂ˝ sken. Simulace ceny nĂ­Ĺľe nemÄ›nĂ­ uloĹľenĂ˝ signĂˇl.")

                shares, value, _, per_share = position_size(
                    capital, risk_pct, proposed, item["stop"], max_position_pct
                )
                if not price_ok:
                    shares, value, per_share = 0, 0.0, 0.0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("PoÄŤet akciĂ­ â€“ zkuĹˇebnĂ­ cena", shares)
                p2.metric("Hodnota pozice", f"${value:,.2f}")
                p3.metric("Riziko na akcii", f"${per_share:,.2f}")
                p4.metric("Riziko pozice", f"${shares * per_share:,.2f}")
                st.caption(
                    "VĂ˝poÄŤty nezahrnujĂ­ poplatky, skluz, mÄ›novĂ© riziko ani cenovĂ© gapy. "
                    "Nejde o investiÄŤnĂ­ doporuÄŤenĂ­."
                )

                st.markdown("#### đź“ Cena, klouzavĂ© prĹŻmÄ›ry a obchodnĂ­ ĂşrovnÄ›")
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

                st.markdown("#### đź“‰ RSI a MACD")
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
                    x=d.index, y=d["MACD_SIGNAL"], name="SignĂˇl",
                    line={"color": "#f59e0b"}
                ), row=2, col=1)
                fig2.update_layout(
                    height=360, margin={"l": 10, "r": 10, "t": 20, "b": 10},
                    showlegend=True, template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True, key=f"indicators_chart_{ticker}")



# ============================================================
# TAB 2 â€“ HISTORIE SIGNĂLĹ®
# ============================================================
with tab_history:
    st.subheader("UloĹľenĂ© signĂˇly")
    if sb is None:
        st.info("Historie nenĂ­ dostupnĂˇ: nenĂ­ nakonfigurovĂˇno pĹ™ipojenĂ­ Supabase.")
    else:
        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            limit = st.number_input("Max. Ĺ™ĂˇdkĹŻ", 50, 2000, 500, 50)
        with col_f2:
            only_today = st.checkbox("Pouze dneĹˇnĂ­", False)

        if st.button("đź”„ NaÄŤĂ­st historii", key="load_history"):
            try:
                q = sb.table("scanner_signals").select("*").order("signal_date", desc=True)
                if only_today:
                    today = datetime.now(timezone.utc).date().isoformat()
                    q = q.eq("signal_date", today)
                resp = q.limit(int(limit)).execute()
                hist = pd.DataFrame(resp.data or [])
                st.session_state.history_df = hist
            except Exception as e:
                log.exception("NaÄŤtenĂ­ historie selhalo")
                st.error(f"Historii se nepodaĹ™ilo naÄŤĂ­st: {e}")

        hist = st.session_state.get("history_df")
        if hist is not None:
            if hist.empty:
                st.info("V databĂˇzi zatĂ­m nejsou uloĹľenĂ© signĂˇly.")
            else:
                st.dataframe(hist, use_container_width=True, hide_index=True)
                st.download_button(
                    "đź“Ą StĂˇhnout historii CSV",
                    hist.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_historie.csv", "text/csv"
                )

# ============================================================
# TAB 3 â€“ UÄŚĂŤCĂŤ SE PĹEHLED
# ============================================================
with tab_learning:
    st.subheader("VyhodnocenĂ­ historickĂ˝ch signĂˇlĹŻ")
    st.caption("Pouze uloĹľenĂ© nĂˇkupnĂ­ signĂˇly 4.7 po sezĂłnnĂ­m filtru; simulace neobnovuje historickĂ© opÄŤnĂ­ snapshoty "
               "ani sezĂłnnĂ­ vzorky. VĂ˝stupnĂ­ pravidla drĹľitele se zde netestujĂ­. Model: pĹ™Ă­ĹˇtĂ­ open v zĂłnÄ›, "
               "kontrola R:R, vĂ˝stup do 5 dnĹŻ. "
               "Jde o simulaci bez nĂˇkladĹŻ, nikoli validaci ziskovosti. DennĂ­ OHLC data nemusĂ­ urÄŤit "
               "poĹ™adĂ­ zĂˇsahu stopu a cĂ­le v rĂˇmci stejnĂ©ho dne.")

    if sb is None:
        st.info("UÄŤĂ­cĂ­ pĹ™ehled vyĹľaduje pĹ™ipojenĂ­ Supabase.")
    else:
        days = st.slider("Historie (dny)", 30, 365, 90, key="learning_days")

        if st.button("đź§  Vyhodnotit signĂˇly", key="run_learning"):
            with st.spinner("Vyhodnocuji historickĂ© signĂˇlyâ€¦"):
                learning = load_learning_stats(sb, days=days)
            st.session_state.learning_df = learning

        learning = st.session_state.get("learning_df")
        if learning is not None:
            if learning.empty:
                st.info("Nejsou dostupnĂˇ data k vyhodnocenĂ­.")
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
                c4.metric("VĂ˝stup 5. den", opn)
                c5.metric("Vstup nevyplnÄ›n", not_filled)
                if pend:
                    st.caption(f"ÄŚekĂˇ na dokonÄŤenĂ­ 5 obchodnĂ­ch dnĹŻ: {pend} signĂˇlĹŻ.")

                resolved = tp1 + sl
                if resolved > 0:
                    win_rate = tp1 / resolved * 100
                    st.metric("TP1 / (TP1 + SL); bez ÄŤasovĂ˝ch a nejasnĂ˝ch vĂ˝stupĹŻ",
                              f"{win_rate:.1f} %",
                              help=f"Vyhodnoceno: {resolved} obchodĹŻ")

                # PrĹŻmÄ›rnĂ˝ forward return podle outcome
                valid = learning.dropna(subset=["forward_return_5d"])
                if not valid.empty:
                    st.markdown("#### PrĹŻmÄ›rnĂ˝ vĂ˝nos (vĂ˝stup nejpozdÄ›ji 5. den) podle vĂ˝sledku")
                    agg = valid.groupby("outcome")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg, use_container_width=True)

                # PrĹŻmÄ›rnĂ˝ return podle signĂˇlu
                if not valid.empty:
                    st.markdown("#### PrĹŻmÄ›rnĂ˝ vĂ˝nos (vĂ˝stup nejpozdÄ›ji 5. den) podle typu signĂˇlu")
                    agg2 = valid.groupby("signal")["forward_return_5d"].agg(
                        ["count", "mean"]).round(2)
                    st.dataframe(agg2, use_container_width=True)

                st.markdown("#### DetailnĂ­ data")
                st.dataframe(learning, use_container_width=True, hide_index=True)

                st.download_button(
                    "đź“Ą StĂˇhnout vyhodnocenĂ­ CSV",
                    learning.to_csv(index=False).encode("utf-8-sig"),
                    "klondike_learning.csv", "text/csv"
                )
