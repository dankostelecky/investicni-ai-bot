# VERZE: KLONDIKE R5.1 | 2026-10-24 | 20s/30d HIGH + PRE/POST
"""Klondike Spot Scanner — přehledná edice pro akcie a zemědělské komodity.

Spuštění: streamlit run app.py
Analytické jádro 4.8 a Supabase scanner_signals zůstávají kompatibilní.
Osobní rozhodovací filtr, bezpečné snímky signálů a adaptivní učení.
"""

import json
import urllib.request
import urllib.parse
from calendar import monthrange
import logging
import re
import time
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
from adaptive_core import (LEARNING_MODEL_VERSION, TRAINING_COST_PCT,
                           outcome_from_bars, learned_adjustment, bucket_features)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("klondike")

# USA-listed USD stocks. Mid-cap membership and liquidity vary over time:
# hard liquidity/price checks are performed on every scan, never assumed.
STOCK_GROUPS = {
    "Základní výběr": ["VMI", "CCJ", "AMBA", "CRSP", "NVDA", "AAPL", "GOOGL", "MSFT",  "MO",
                       "AMZN", "META", "TSLA", "BRK-B", "WMT", "MU", "JPM", "ORCL",
                       "XOM", "AMD", "JNJ", "COST", "HD", "UNH", "BAC", "IBM", "INTC", "BABA"],
    "Banky": ["WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "FITB", "KEY"],
    "Pojišťovny": ["PGR", "CB", "TRV", "ALL", "AIG", "HIG", "MET", "PRU"],
    "Těžaři a suroviny": ["NEM", "AEM", "KGC", "WPM", "FCX", "SCCO", "TECK", "RIO", "VALE"],
    "Distributoři": ["GWW", "WCC", "FAST", "POOL", "SYY", "RS"],
    "Likvidní střední společnosti – kandidáti": ["CROX", "FIVE", "DOCU", "GTLS", "ON", "FLEX", "CELH", "ELF"],
}
DEFAULT_TICKERS = list(dict.fromkeys(STOCK_GROUPS["Základní výběr"] +
    ["PGR", "CB", "NEM", "FCX", "GWW", "WCC", "CROX", "FLEX"]))
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
@st.cache_data(ttl=150, show_spinner=False)
def extended_market_quote(ticker: str, latest_daily_date=None):
    """Recent regular/pre/post 5m Yahoo bars. A stale/unknown quote is NEVER live.

    The intraday auto-adjust convention matches adjusted daily OHLC used by the
    strategy; Yahoo's intraday and daily corrections may still differ.
    """
    try:
        bars = yf.Ticker(ticker).history(period="5d", interval="5m", prepost=True,
                                        auto_adjust=True)
        if bars is None or bars.empty or "Close" not in bars:
            return None
        bars = bars.dropna(subset=["Close"])
        if "Volume" in bars:
            bars = bars.loc[pd.to_numeric(bars["Volume"], errors="coerce").fillna(0) > 0]
        if bars.empty:
            return None
        last = bars.iloc[-1]
        stamp = pd.Timestamp(bars.index[-1])
        if stamp.tzinfo is None:
            return None  # exchange timezone missing; refuse ambiguous quote
        eastern = stamp.tz_convert("America/New_York")
        minute = eastern.hour * 60 + eastern.minute
        if 4*60 <= minute < 9*60+30:
            kind = "PRE"
        elif 9*60+30 <= minute < 16*60:
            kind = "REG"
        elif 16*60 <= minute < 20*60:
            kind = "POST"
        else:
            return None
        now = pd.Timestamp.now(tz="UTC")
        age_hours = (now - stamp.tz_convert("UTC")).total_seconds() / 3600
        # At most 3 days for weekends/US market holidays, otherwise <= 20h.
        weekend = now.weekday() in (5, 6) or eastern.weekday() in (5, 6)
        allowed_age = 72 if weekend else 20
        price = safe_float(last["Close"], None)
        if price is None or price <= 0 or not (0 <= age_hours <= allowed_age):
            return None
        daily_date = pd.Timestamp(latest_daily_date).date() if latest_daily_date is not None else None
        if daily_date is not None and eastern.date() < daily_date:
            return None
        # For a completed daily bar of the same calendar date, only POST adds
        # new information. Otherwise the daily bar is the more reliable source.
        if daily_date == eastern.date() and kind != "POST":
            return None
        return {"price": price, "kind": kind, "asof": stamp.isoformat(),
                "age_hours": age_hours}
    except Exception as exc:
        log.info("Mimoburzovní data %s nedostupná: %s", ticker, exc)
        return None


@st.cache_data(ttl=900, show_spinner=False)
def extended_30day_high(ticker: str, window_start: str, window_end: str):
    """Maximum of real intraday High in the 30-calendar-day window, PRE/REG/POST.

    Yahoo 60-minute bars are used only as a historical high supplement to daily
    OHLC.  Bars with unknown timezone or missing/zero reported volume are not
    treated as executable observations. Missing data must not imply full
    pre/post coverage.
    """
    try:
        bars = yf.Ticker(ticker).history(period="45d", interval="60m",
                                        prepost=True, auto_adjust=True)
        if bars is None or bars.empty or "High" not in bars:
            return None
        idx = pd.DatetimeIndex(bars.index)
        if idx.tz is None:
            return None
        idx = idx.tz_convert("America/New_York")
        first = pd.Timestamp(window_start).date()
        last = pd.Timestamp(window_end).date()
        mask = (idx.date >= first) & (idx.date <= last)
        minute = idx.hour * 60 + idx.minute
        mask &= (minute >= 4 * 60) & (minute < 20 * 60)
        if "Volume" in bars:
            mask &= (pd.to_numeric(bars["Volume"], errors="coerce").fillna(0).to_numpy() > 0)
        highs = pd.to_numeric(bars.loc[mask, "High"], errors="coerce")
        highs = highs[(highs > 0) & np.isfinite(highs)]
        return float(highs.max()) if not highs.empty else None
    except Exception as exc:
        log.info("Historická PRE/POST maxima %s nejsou dostupná: %s", ticker, exc)
        return None


def calculate_peak_potentials(d, effective_price, ticker, mode, quote=None):
    """20s: daily High over 20 sessions; 30d: daily + intraday High.

    The 30-day interval is anchored to the latest available quote's US trading
    date, or to the last completed daily session when no quote is available.
    This avoids misrepresenting stale observations as current.
    """
    if not np.isfinite(effective_price) or effective_price <= 0:
        return np.nan, np.nan, "nedostupná cena"
    high20 = float(pd.to_numeric(d["High"].tail(20), errors="coerce").max())
    last_day = pd.Timestamp(d.index[-1]).date()
    anchor = last_day
    if quote and quote.get("asof"):
        try:
            anchor = max(last_day, pd.Timestamp(quote["asof"]).tz_convert("America/New_York").date())
        except (ValueError, TypeError):
            pass
    start = anchor - timedelta(days=29)
    dates = pd.to_datetime(d.index).date
    recent = d.loc[(dates >= start) & (dates <= anchor)]
    daily_peak = safe_float(pd.to_numeric(recent["High"], errors="coerce").max(), np.nan)
    peak = daily_peak
    source = "D1 High (PRE/POST historie nedostupná)" if mode == "stocks" else "D1 High"
    if mode == "stocks":
        extra = extended_30day_high(ticker, start.isoformat(), anchor.isoformat())
        if extra is not None and np.isfinite(extra):
            peak = max(daily_peak, extra) if np.isfinite(daily_peak) else extra
            source = "D1 + dostupná PRE/REG/POST High"
    return (max(0.0, high20 / effective_price - 1),
            max(0.0, peak / effective_price - 1) if np.isfinite(peak) else np.nan,
            source)


def personal_style_score(item):
    """Transparent preference match. NOT a probability of profit or a buy signal."""
    d = item.get("data")
    score = 0
    distance = safe_float(item.get("support_distance_atr"))
    if np.isfinite(distance):
        score += 30 if 0 <= distance <= .5 else (17 if 0 <= distance <= 1. else 0)
    if int(item.get("support_touches") or 0) >= 2:
        score += 20
    if isinstance(d, pd.DataFrame) and len(d) >= 12:
        prior = safe_float(d["Close"].iloc[-12])
        now = safe_float(d["Close"].iloc[-1])
        if prior > 0 and now / prior - 1 <= -.03:
            score += 15  # pullback favored, but not sufficient for buying
        c = d.iloc[-1]
        span = safe_float(c["High"]) - safe_float(c["Low"])
        if span > 0 and (safe_float(c["Close"]) - safe_float(c["Low"])) / span >= .6:
            score += 15
    if safe_float(item.get("volume_ratio"), 0) >= 1.2:
        score += 10
    if item.get("quiet_pullback"):
        score += 10
    return min(score, 100)


def risk_quote_context(item):
    """Independent current-price sanity check; never replaces completed-bar setup."""
    value = safe_float(item.get("effective_price"), item["price"])
    lower = safe_float(item.get("min_execution_price"))
    upper = safe_float(item.get("max_execution_price"))
    stop = safe_float(item.get("stop"))
    rr = (safe_float(item.get("target1")) - value) / (value - stop) if value > stop else np.nan
    if value <= stop:
        return "VAROVÁNÍ", "Aktuální referenční cena je na invalidaci nebo pod ní.", rr
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= value <= upper):
        return "MIMO VSTUP", "Aktuální referenční cena leží mimo povolenou nákupní zónu.", rr
    if not np.isfinite(rr) or rr < STRATEGY["min_rr"]:
        return "MIMO VSTUP", "Aktuální poměr potenciálního zisku k riziku nestačí.", rr
    return "V ZÓNĚ", "Referenční cena vyhovuje; intradenní potvrzení nebylo automaticky ověřeno.", rr


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
    context_fresh = (weather_is_current(item.get("weather_context", {}))
                     if item.get("mode") == "crops" else True)
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
    """Sezónní filtr může vstup omezit, nikdy obejít základní supportní filtr."""
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
        r["entry_reason"] = "Technické podmínky jsou splněny, ale nákup blokuje sezónnost."
    else:
        r["entry_reason"] = entry["entry_reason"]
    r["entry_reason"] += " Sezónnost: " + context["reason"]
    r["strategy_version"] = "4.8"
    return r



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
        if "Earnings Date" in raw.index:
            return raw.iloc[:, 0].to_dict()
        return raw.iloc[0].to_dict()
    return {}


def normalize_events(info, calendar, earnings_index=(), today=None):
    """Preserve an estimated earnings range, including an interval already in progress."""
    today = today or local_today()
    calendar = calendar_dict(calendar)
    earnings = sorted({d for v in date_values(calendar.get("Earnings Date"))
                       if (d := event_date(v, epoch=True)) is not None})
    kind = "Kalendář Yahoo Finance; datum může být odhadem."
    if not earnings or earnings[-1] < today:
        earnings = sorted({d for v in earnings_index
                           if (d := event_date(v)) is not None and d >= today})[:1]
        kind = "Výsledkový kalendář Yahoo Finance; datum ověřte u emitenta."
    return {
        "earnings_start": earnings[0].isoformat() if earnings else None,
        "earnings_end": earnings[-1].isoformat() if earnings else None,
        "earnings_kind": kind,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_corporate_events(ticker):
    obj = yf.Ticker(ticker)
    errors, info, calendar, earnings_index = [], {}, {}, []
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


# Opční řetězce byly odebrány z uživatelského rozhraní i rozhodovací logiky.


def assess_trade_action(item: dict, held=False, position_stop=None, position_target=None) -> dict:
    """Odděleně nový vstup a správa existující long pozice, vždy z uzavřeného dne.

    Vlastní stop/cíl má přednost. Nezaměňujeme nový modelový stop s plánem držitele.
    Volitelné mimoburzovní kotace slouží jen ke kontrole aktuálního rizika.
    """
    result = {"trend_label": "BÝČÍ" if item["trend_ok"] else "NEPOTVRZENÝ / SLABÝ"}
    if not held:
        if item.get("mode") == "crops" and not weather_is_current(item.get("weather_context", {})):
            return {**result, "action_label": "ČEKAT / NEVSTUPOVAT",
                    "action_reason": "Počasí chybí, není úplné nebo je snímek starší než 6 hodin. Spusťte nový sken."}
        context, reason, _ = risk_quote_context(item)
        if item.get("mode") == "stocks" and context != "V ZÓNĚ":
            return {**result, "action_label": "ČEKAT / NEVSTUPOVAT",
                    "action_reason": reason + " (Zohledněna poslední dostupná cena včetně mimořádných hodin.)"}
        return {**result, "action_label": "NAKUPOVAT" if item["buy_allowed"] else "ČEKAT / NEVSTUPOVAT",
                "action_reason": item["entry_reason"] + " " + reason}
    price = item.get("effective_price", item["price"])
    stop, target = safe_float(position_stop, 0), safe_float(position_target, 0)
    if stop > 0 and target > 0 and target <= stop:
        return {**result, "action_label": "OVĚŘIT PLÁN", "action_reason": "U nákupní pozice musí být cílová cena nad úrovní stopu."}
    extended_illiquid = (item.get("extended_quote") or {}).get("kind") in {"PRE", "POST"}
    if stop > 0 and price <= stop:
        label = "OVĚŘIT STOP" if extended_illiquid else "PRODAT"
        reason = ("PRE/POST cena je pod vaším stopem; mimo řádnou seanci nemusí být stop aktivní. "
                  "Ověřte kotaci a podmínky brokera." if extended_illiquid else
                  "Poslední referenční cena je na úrovni vašeho stopu nebo pod ní.")
    elif target > 0 and price >= target:
        label = "OVĚŘIT CÍL" if extended_illiquid else "PRODAT"
        reason = ("PRE/POST cena dosáhla cíle, ale může mít nízkou likviditu. Ověřte obchodovatelnost." if extended_illiquid else
                  "Poslední referenční cena dosáhla vašeho cíle nebo jej překonala.")
    elif str(item.get("structure_event", "")).endswith("dolů"):
        label, reason = "PRODAT / ZVÁŽIT REDUKCI", "Potvrzený průlom swingové struktury dolů."
    else:
        weak = price < item["ema50"] and item["macd_hist"] < 0 and not item["trend_ok"]
        label = "DRŽET / SLEDOVAT"
        reason = "Podle poslední dokončené denní svíčky nebyla potvrzena podmínka pro výstup."
        if weak:
            reason += " Technické ukazatele slábnou; ověřte aktuální stop."
    if not (stop > 0 or target > 0):
        reason += " Nebyla zadána vlastní úroveň stopu ani cílová cena."
    return {**result, "action_label": label, "action_reason": reason}

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
        # Detail opcí, sezónnosti, kalendář a rozhodnutí držitele nejsou ve starém DB schématu.
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


def load_learning_stats(sb, days: int = 90, mode="stocks", holding_days=5):
    """Simulace nákupů 4.8 aktuální části a horizontu; vstup na příštím open.

    Není to walk-forward backtest ani učení parametrů. Gap mimo zónu = bez vstupu.
    Denní OHLC nezná pořadí SL/TP; takový obchod se nezařadí mezi výhry/prohry.
    """
    if sb is None:
        return pd.DataFrame()
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        response = sb.table("scanner_signals").select("*").gte(
            "signal_date", cutoff).eq("signal", buy_signal(mode, holding_days)).execute()
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
            window = hist.iloc[i+1:i+1+holding_days]
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
                if len(window) < holding_days:
                    outcome = "pending"
                else:
                    outcome = "TIME_EXIT"
                    result = (float(window["Close"].iloc[-1]) / entry - 1) * 100
            returns.append(result); outcomes.append(outcome)
        df[f"forward_return_{holding_days}d"] = returns
        df["outcome"] = outcomes
        return df
    except Exception as e:
        log.error("load_learning_stats selhalo: %s", e, exc_info=True)
        return pd.DataFrame()

# ---------------------- ADAPTIVNÍ UČENÍ -----------------------
@st.cache_resource(show_spinner=False)
def get_adaptive_supabase():
    """Only server-side service key can write private RLS-protected snapshots."""
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        return create_client(url, key) if url and key else None
    except Exception as exc:
        log.warning("Adaptivní Supabase není dostupná: %s", exc)
        return None


def snapshot_signal(item):
    """Frozen features, not modified retroactively by learning."""
    row = {
        "ticker": item["ticker"], "signal_date": pd.Timestamp(item["data"].index[-1]).date().isoformat(),
        "mode": item.get("mode", "stocks"), "horizon": int(item.get("holding_days", 5)),
        "model_version": LEARNING_MODEL_VERSION,
        "strategy_version": str(item.get("strategy_version", "4.8")),
        "signal": str(item.get("signal", "")), "buy_allowed": bool(item.get("buy_allowed", False)),
        "snapshot_close": safe_float(item.get("price")),
        "entry_price": safe_float(item.get("preferred_entry")),
        "stop_price": safe_float(item.get("stop")),
        "target_price": safe_float(item.get("target1")),
        "min_execution_price": safe_float(item.get("min_execution_price")),
        "max_execution_price": safe_float(item.get("max_execution_price")),
        "support_distance_atr": safe_float(item.get("support_distance_atr")),
        "volume_ratio": safe_float(item.get("volume_ratio")),
        "trend_score": safe_float(item.get("trend_score")),
        "quality_score": safe_float(item.get("quality_score")),
        "entry_score": safe_float(item.get("entry_score")),
        "style_score": safe_float(item.get("style_score")),
        "rr1": safe_float(item.get("rr1")),
        "price_type": (item.get("extended_quote") or {}).get("kind", "DAILY"),
        "price_time": (item.get("extended_quote") or {}).get("asof"),
        "features": bucket_features(item),
    }
    for k, v in row.items():
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            row[k] = None
    row["status"] = "PENDING" if row["buy_allowed"] else "INELIGIBLE"
    row["outcome"] = None if row["buy_allowed"] else "INELIGIBLE"
    return row


def save_adaptive_snapshots(sb, rows):
    if sb is None or not rows:
        return False
    try:
        snapshots = [snapshot_signal(item) for item in rows]
        # Ignore duplicate same-day scans. Never rewrite a historical snapshot.
        sb.table("scanner_adaptive_signals").upsert(
            snapshots, on_conflict="ticker,signal_date,mode,horizon,model_version",
            ignore_duplicates=True).execute()
        return True
    except Exception as exc:
        log.warning("Adaptivní snímky nelze uložit; ověřte migraci a RLS: %s", exc)
        return False


def update_adaptive_outcomes(sb, max_rows=160):
    """Matured-only outcome updates, with zero use of the current partial daily bar."""
    if sb is None:
        return 0
    updated = 0
    try:
        pending = (sb.table("scanner_adaptive_signals").select("*")
                   .eq("status", "PENDING").order("signal_date")
                   .limit(max_rows).execute().data or [])
        now = datetime.now(timezone.utc).date()
        for row in pending:
            if pd.Timestamp(row["signal_date"]).date() >= now:
                continue
            age = (now - pd.Timestamp(row["signal_date"]).date()).days
            period = "5y" if age > 650 else ("2y" if age > 300 else "1y")
            hist = download_history(row["ticker"], period, "1d")
            result = outcome_from_bars(row, hist)
            if result["status"] == "PENDING":
                continue
            (sb.table("scanner_adaptive_signals").update(result)
             .eq("id", row["id"]).eq("status", "PENDING").execute())
            updated += 1
    except Exception as exc:
        log.warning("Vyhodnocení adaptivních výsledků selhalo: %s", exc)
    return updated


def load_adaptive_resolved(sb, mode, horizon, limit=1000):
    if sb is None:
        return []
    try:
        # Filter same market and evaluation horizon to avoid mixing strategies.
        return (sb.table("scanner_adaptive_signals").select("*")
                .eq("status", "RESOLVED").eq("mode", mode).eq("horizon", horizon)
                .eq("model_version", LEARNING_MODEL_VERSION)
                .order("resolved_at", desc=True).limit(limit).execute().data or [])
    except Exception as exc:
        log.warning("Nelze načíst modelovou historii: %s", exc)
        return []


def apply_adaptive_ranking(results, training):
    """Update rank only. Never bypass technical invalidation or risk limits."""
    ranked = []
    for original in results:
        item = dict(original)
        learned = learned_adjustment(item, training)
        item["style_score"] = personal_style_score(item)
        item["adaptive_learning"] = learned
        item["adaptive_score"] = round(float(np.clip(float(item["entry_score"]) + learned["adjustment"], 0, 100)), 1)
        ranked.append(item)
    return sorted(ranked, key=lambda r: (r["buy_allowed"], r["adaptive_score"],
                                          r["quality_score"]), reverse=True)


def render_adaptive_learning(sb, mode, horizon):
    st.subheader("Adaptivní učení z dokončených signálů")
    st.caption("Model průběžně přepočítává konzervativní korekci pořadí podle reálných výsledků "
               "historických simulovaných vstupů. Vstup na open následující seance, "
               "pouze dokončené denní svíčky; nejasné a nevyplněné vstupy se neučí. "
               f"Simulované náklady {TRAINING_COST_PCT:.2f} % za celý obchod. "
               "Jde o učení z výsledků signálů, nikoli automatický trading ani pravděpodobnost úspěchu.")
    if sb is None:
        st.warning("Adaptivní databáze zatím není připojena. Spusťte SQL migraci a nastavte "
                   "SUPABASE_SERVICE_ROLE_KEY ve Streamlit Secrets. Bez databáze se skóre neučí.")
        return
    if st.button("Aktualizovat dokončené výsledky", key=f"refresh_adapt_{mode}_{horizon}"):
        with st.spinner("Kontroluji historické signály…"):
            updated = update_adaptive_outcomes(sb)
        st.info(f"Nově vyhodnocených záznamů: {updated}")
    resolved = load_adaptive_resolved(sb, mode, horizon)
    st.metric("Jednoznačně vyhodnocené vstupy", len(resolved))
    if not resolved:
        st.info("První učící data vzniknou až po uložených signálech a uplynutí jejich horizontu.")
        return
    df = pd.DataFrame(resolved)
    cols = st.columns(3)
    cols[0].metric("Průměrné hrubé R", display_number(pd.to_numeric(df["realized_r"],errors="coerce").mean(),3))
    cols[1].metric("Zasažené cíle", str(int((df["outcome"]=="TP1").sum())))
    cols[2].metric("Zasažené stopy", str(int((df["outcome"]=="SL").sum())))
    st.dataframe(df[[c for c in ["ticker","signal_date","outcome","realized_return_pct",
                                 "realized_r","resolved_at"] if c in df]],
                 use_container_width=True, hide_index=True)


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
    st.caption("Cena jedné jednotky ETC obchodovaného v Londýně v USD. Výnos ovlivňují futures, rolování a náklady produktu.")
    st.link_button("Informace emitenta", product["source"])
    context = item["weather_context"]
    st.markdown("#### 🌦️ Počasí v pěstitelských oblastech")
    st.write(context["reason"])
    st.caption(f"Načteno: {context['fetched_at'] or 'nedostupné'} · "
               f"pokrytí modelových vah {display_number(context['coverage'] * 100, 0)} % · soubor předpovědí GFS na 14 kalendářních dnů.")
    st.caption("Předpověď nepokrývá vždy celou zvolenou dobu držení. Počasí i cenový plán je potřeba průběžně kontrolovat.")
    c1, c2 = st.columns(2)
    c1.metric("Modelový index stresu plodin", display_number(context["stress"]) if context["stress"] is not None else "—")
    c2.metric("Změna na shodných dnech", display_number(context["revision"], sign=True) if context["revision"] is not None else "—")
    with st.expander("Podrobnosti počasí podle oblastí", expanded=False):
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
def buy_signal(mode="stocks", holding_days=5):
    """Signál s identifikátorem původní strategie; horizont plodin je součástí signálu."""
    return f"PLODINY · NÁKUPNÍ ZÓNA · 4.8 · {holding_days}D" if mode == "crops" else BUY_SIGNAL


class ScanDataError(ValueError):
    """Předvídatelný problém s daty, jehož důvod se zobrazí u instrumentu."""


def scan_ticker(ticker: str, spy: pd.DataFrame, mode="stocks", weather_context=None, holding_days=5):
    data = download_history(ticker)
    if data.empty:
        raise ScanDataError("Yahoo Finance neposkytlo použitelnou cenovou historii. Může jít o dočasnou nedostupnost nebo omezení počtu požadavků. Zkuste sken později zopakovat.")
    if len(data) < 60:
        raise ScanDataError(f"Pro výpočet je potřeba alespoň 60 dokončených seancí. Dostupných seancí: {len(data)}.")

    d = add_indicators(data)
    x = d.iloc[-1]

    # Neznámý objem nezahazuje cenový graf. Podmínky likvidity níže nákup blokují.
    required = ["EMA20", "EMA50", "EMA200", "RSI14", "ATR14"]
    missing = [key for key in required if not np.isfinite(safe_float(x[key]))]
    if missing:
        raise ScanDataError("Nelze vypočítat cenové ukazatele: " + ", ".join(missing) + ".")

    scores = calculate_scores(d, spy)
    regime, _ = market_regime(spy)
    last_day = pd.Timestamp(d.index[-1]).date()
    today = datetime.now(timezone.utc).date()
    data_ok = 0 <= (today - last_day).days <= 4
    if spy.empty or pd.Timestamp(spy.index[-1]).date() != last_day:
        data_ok = False
    entry = calculate_entry_engine(d, scores, regime, data_ok, mode)
    # Opční výpočty byly z rozhodování záměrně odstraněny.
    entry["technical_buy_allowed"] = entry["buy_allowed"]
    entry = apply_seasonality_filter(entry, get_seasonality_context(
        ticker, last_day.isoformat(), holding_days))
    if mode == "crops":
        entry = apply_weather_filter(entry, weather_context or {})
        entry["signal"] = buy_signal(mode, holding_days) if entry["buy_allowed"] else f"PLODINY · {holding_days}D · " + entry["signal"]
    events = get_corporate_events(ticker) if mode == "stocks" else {}
    quote = extended_market_quote(ticker, last_day) if mode == "stocks" else None
    effective_price = quote["price"] if quote is not None else float(x["Close"])
    gain_per_dollar, gain_30c_per_dollar, gain_30_source = calculate_peak_potentials(
        d, effective_price, ticker, mode, quote)
    patterns = detect_patterns(d)
    structure = analyze_market_structure(d)

    return {
        "ticker":       ticker,
        "mode": mode, "holding_days": holding_days,
        "corporate_events": events,
        "gain_per_dollar": gain_per_dollar,
        "gain_30c_per_dollar": gain_30c_per_dollar,
        "gain_30_source": gain_30_source,
        "extended_quote": quote,
        "effective_price": effective_price,
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
                  mode="stocks", weather_contexts=None, holding_days=5, diagnostics=None):
    results = []
    diagnostics = diagnostics if diagnostics is not None else {}
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(scan_ticker, t, spy, mode,
                   (weather_contexts or {}).get(CROP_PRODUCTS[t]["crop"]) if mode == "crops" else None,
                   holding_days): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
                else:
                    diagnostics[t] = "Analýza neposkytla výsledek. Zopakujte sken."
            except ScanDataError as exc:
                diagnostics[t] = str(exc)
                log.warning("Data %s: %s", t, exc)
            except Exception:
                diagnostics[t] = "Analýzu se nepodařilo dokončit. Cenová data nebo některý výpočet nejsou dostupné. Zopakujte sken."
                log.exception("Sken %s selhal", t)
            done += 1
            if progress_cb:
                progress_cb(done, total)
    return results


# ============================================================

# ---------------------- PŘEHLEDNÉ ZOBRAZENÍ --------------------
def display_number(value, precision=2, currency=False, sign=False, grouped=True):
    number = safe_float(value, None)
    if number is None:
        return "—"
    spec = ("+" if sign else "") + ("," if grouped else "") + f".{precision}f"
    formatted = format(number, spec).replace(",", "\u00a0").replace(".", ",")
    return formatted + (" USD" if currency else "")


def display_date(value):
    day = event_date(value)
    return day.strftime("%d. %m. %Y") if day else "Datum není známé"


def display_days(count):
    if count is None:
        return "Datum není známé"
    if count == 0:
        return "Dnes"
    unit = "den" if count == 1 else ("dny" if 2 <= count <= 4 else "dnů")
    return f"{count} {unit}"


OUTCOME_LABELS = {
    "TP1": "Dosažen první cíl", "SL": "Výstup na stopu",
    "AMBIGUOUS": "Nejasné pořadí výstupů", "TIME_EXIT": "Časový výstup",
    "NOT_FILLED": "Obchod nebyl otevřen", "pending": "Čeká na dokončení",
    "DATA_CHANGED": "Změněná historická data", "unknown": "Nelze vyhodnotit", "OPEN": "Otevřený obchod",
}
COLUMN_LABELS = {
    "id": "ID", "ticker": "Ticker", "signal_date": "Datum signálu", "price": "Cena",
    "quality_score": "Skóre kvality", "entry_score": "Vstupní skóre", "trend_score": "Skóre trendu",
    "momentum_score": "Síla pohybu", "rs_score": "Skóre relativní síly", "risk_score": "Skóre rizika",
    "preferred_entry": "Preferovaný vstup", "zone_low": "Dolní hranice vstupní zóny",
    "zone_high": "Horní hranice vstupní zóny", "stop_price": "Úroveň stopu",
    "target1": "První cíl (TP1)", "target2": "Druhý cíl (TP2)", "rr1": "Poměr výnosu k riziku pro TP1",
    "rr2": "Poměr výnosu k riziku pro TP2", "support": "Podpora (support)", "resistance": "Rezistence",
    "rsi": "RSI", "atr": "ATR", "atr_pct": "ATR (%)", "volume_ratio": "Objem vůči průměru",
    "rs_30d": "Relativní síla za 30 seancí (%)", "signal": "Signál", "outcome": "Výsledek",
    "count": "Počet obchodů", "mean": "Průměrný výnos (%)", "created_at": "Čas vytvoření",
    "updated_at": "Čas aktualizace",
}


def localize_table(frame):
    """Překládá pouze zobrazenou kopii; databáze a datové exporty se nemění."""
    shown = frame.copy()
    if "outcome" in shown:
        shown["outcome"] = shown["outcome"].map(lambda value: OUTCOME_LABELS.get(value, value))
    if shown.index.name == "outcome":
        shown.index = shown.index.map(lambda value: OUTCOME_LABELS.get(value, value))
        shown.index.name = "Výsledek"
    elif shown.index.name == "signal":
        shown.index.name = "Signál"
    names = dict(COLUMN_LABELS)
    for column in shown:
        if isinstance(column, str) and column.startswith("forward_return_"):
            names[column] = "Výnos obchodu (%)"
    shown = shown.rename(columns=names)
    return shown



def earnings_in_window(events, window=7, today=None):
    """True for an earnings date/range intersecting today through day +window."""
    today = today or local_today()
    first = event_date(events.get("earnings_start"))
    last = event_date(events.get("earnings_end")) or first
    return bool(first and last and first <= today + timedelta(days=window) and last >= today)


def filter_results(results, high_gain=False, outperforming=False, safe_earnings=True, high_gain30=True):
    """20 session close high (6%) and 30-calendar-day close high (15%) are independent filters."""
    return [item for item in results
            if (not high_gain or safe_float(item.get("gain_per_dollar"), -1) >= 0.06 - 1e-12)
            and (not high_gain30 or safe_float(item.get("gain_30c_per_dollar"), -1) >= 0.15 - 1e-12)
            and (not outperforming or safe_float(item.get("rs_30d"), 0) > 0)
            and (not safe_earnings or not earnings_in_window(item.get("corporate_events", {})))]


def render_events_panel(ticker, events):
    st.markdown("#### 🗓️ Zveřejnění hospodářských výsledků")
    first, last = events.get("earnings_start"), events.get("earnings_end")
    n1, n2 = days_until(first), days_until(last)
    if first and last and first != last:
        value = f"{n1}–{display_days(n2)}" if n1 is not None else ("V odhadovaném období" if n2 is not None else "Datum není známé")
        subtitle = f"{display_date(first)} až {display_date(last)}"
    else:
        value = display_days(n1)
        subtitle = display_date(first) if first else "Datum příštího zveřejnění není dostupné."
    st.metric("Do zveřejnění výsledků", value)
    st.caption(f"{subtitle} · {events.get('earnings_kind', 'Yahoo Finance; datum může být odhadem.')}")
    if earnings_in_window(events):
        st.warning("Výsledky budou zveřejněny během příštích 7 dnů nebo již začalo odhadované období jejich zveřejnění.")


def render_seasonality_panel(ticker, context):
    """Only the original monthly seasonality chart; no tables or stat cards."""
    monthly = pd.DataFrame(context.get("monthly", []))
    if monthly.empty:
        st.info("Pro graf sezónnosti není dostatek historických dat.")
        return
    fig = go.Figure(go.Bar(
        x=monthly["name"], y=monthly["median_pct"],
        marker_color=["#16a085" if v >= 0 else "#e76f73" for v in monthly["median_pct"]],
        customdata=monthly[["sample_years", "positive_pct"]].to_numpy(),
        hovertemplate="%{x}<br>Medián: %{y:.2f} %<br>Počet let: %{customdata[0]}"
                      "<br>Podíl růstových měsíců: %{customdata[1]:.0f} %<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        title="Sezónnost · medián měsíčního pohybu", height=310,
        yaxis_title="Medián (%)", template="plotly_white", separators=", ",
        margin=dict(l=15, r=15, t=55, b=20),
        font=dict(family="Arial, sans-serif", color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"seasonality_{ticker}")


def apply_design():
    st.markdown("""
    <style>
    .stApp { background: #f4f7fb; color: #17283b; }
    header[data-testid="stHeader"] { background: rgba(244,247,251,.95); }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e4eaf2; }
    h1,h2,h3,h4 { color: #17283b; letter-spacing: -.035em; }
    .hero { padding: 1.5rem 1.8rem; border: 1px solid #dbe7ef; border-radius: 20px;
            background: linear-gradient(115deg,#ffffff,#eaf6f5); margin-bottom: 1rem;
            box-shadow: 0 8px 28px rgba(33,62,83,.04); }
    .hero .eyebrow { color: #158879; font-size: .74rem; letter-spacing: .16em; font-weight: 700; }
    .hero h1 { font-size: 2rem; padding: .3rem 0; margin: 0; }
    .hero p { color: #607086; margin: .2rem 0 0; font-size: 1rem; }
    [data-testid="stExpander"] { background: #fff; border: 1px solid #dfe7ef; border-radius: 14px; }
    [data-testid="stExpander"] summary { padding-top: .9rem; padding-bottom: .9rem; font-weight: 600; }
    [data-testid="stMetric"] { background: #fff; border: 1px solid #e2eaf2;
                              padding: .85rem 1rem; border-radius: 12px; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; color: #17283b; }
    [data-testid="stMetricLabel"] { color: #63748a; }
    .stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
    .stButton > button[kind="primary"] { background: #138777; border-color: #138777; color: white; }
    .stButton > button[kind="primary"]:hover { background: #0d7164; border-color: #0d7164; color: white; }
    [data-baseweb="tab-list"] { gap: 1rem; }
    [data-baseweb="tab"] { padding: .7rem .25rem; }
    @media(max-width: 700px) { .hero { padding: 1.2rem; } .hero h1 { font-size: 1.5rem; } }
    </style>
    """, unsafe_allow_html=True)


def render_personal_decision(item, capital, risk_pct, max_position_pct):
    mode, ticker = item.get("mode", "stocks"), item["ticker"]
    st.markdown("#### 🧠 Osobní filtr: invalidace → vstup → riziko")
    action = assess_trade_action(item)
    state, message, rr_live = risk_quote_context(item)
    if not item.get("buy_allowed"):
        st.warning(f"**Nový nákup podle technických pravidel blokován:** {item['entry_reason']}")
    elif state != "V ZÓNĚ":
        st.warning(f"**Nový nákup mimo vstupní oblast:** {message}")
    else:
        st.info(f"**Předběžný kandidát:** {message}")
    if item.get("extended_quote"):
        st.caption("Upozornění: PRE/POST cena kontroluje aktuální vzdálenost ke stopu a vstupní zóně, "
                   "ale nepotvrzuje denní ani 30minutový obrat. Cena mimo řádnou seanci může mít široký spread.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Support", display_number(item.get("support"),currency=True))
    c2.metric("Invalidace / stop", display_number(item.get("stop"),currency=True))
    c3.metric("Maximální vstup", display_number(item.get("max_execution_price"),currency=True))
    c4.metric("R:R při referenční ceně", display_number(rr_live,2))
    learned = item.get("adaptive_learning", {})
    st.caption(f"Shoda se stylem {item.get('style_score', 0):.0f}/100 (ne pravděpodobnost zisku) · "
               f"Nezávislá kvalita {item['quality_score']:.1f}/100 · "
               f"Původní vstupní skóre {item['entry_score']:.1f}/100 · "
               f"Adaptivní priorita {item.get('adaptive_score', item['entry_score']):.1f}/100 "
               f"(korekce {learned.get('adjustment', 0):+.2f}; ověřených výsledků {learned.get('samples', 0)}). "
               "Adaptivní korekce ovlivňuje pořadí, nikdy neobchází invalidaci.")
    with st.expander("Kontrola podmínek před nákupem", expanded=False):
        for label, ok in item.get("entry_checks", {}).items():
            st.write(("✅ " if ok else "⬜ ") + label)
        st.caption(learned.get('explanation', 'Učení zatím nemá vyhodnocené signály.'))
    st.write("**Výpočet pozice z pevného rizika**")
    proposed = st.number_input("Zamýšlená cena nákupu (USD)", min_value=0.01,
         value=max(.01, safe_float(item.get("effective_price"), item["price"])),
         step=.01, format="%.4f", key=f"proposed_{mode}_{ticker}")
    price_ok, rr = check_execution_price(item, proposed)
    shares, full_value, _, per_share = position_size(capital, risk_pct, proposed,
                                                       item["stop"], max_position_pct)
    if not price_ok or (mode == "stocks" and state == "VAROVÁNÍ"):
        shares, full_value = 0, 0.
        st.warning("Tato cena / původní signál nesplňuje povinné podmínky; výpočet nákupu blokován.")
    else:
        st.caption("Pozor: povolená ruční cena není potvrzení aktuálního intradenního signálu.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Maximálně jednotek",shares)
    c2.metric("Celá pozice",display_number(full_value,currency=True))
    c3.metric("První tranše 20 %",display_number(full_value*.2,currency=True))
    c4.metric("Plánované riziko celku",display_number(shares*per_share,currency=True))
    st.caption("Plánované maximum dle zadaného kapitálu a stopu; neobsahuje poplatky, "
               "skluz ani skoky přes stop. Rozdělení 20/30/50 je pracovní pravidlo pro celý rizikový rozpočet.")
    held = st.checkbox("Již držím: zobrazit kontrolu stávající pozice",key=f"held_{mode}_{ticker}")
    if held:
        x1,x2=st.columns(2)
        own_stop=x1.number_input("Můj ochranný stop (0 = neznámý)",min_value=0.,value=0.,
                                  key=f"held_stop_{mode}_{ticker}")
        own_target=x2.number_input("Můj cíl (0 = neurčen)",min_value=0.,value=0.,
                                  key=f"held_target_{mode}_{ticker}")
        assessment=assess_trade_action(item, True,own_stop,own_target)
        st.info(f"**{assessment['action_label']}** — {assessment['action_reason']}")
    st.caption("Návrh je analytická pomůcka, nikoli příkaz u brokera.")


def render_instrument(item, capital, risk_pct, max_position_pct):
    ticker, mode = item["ticker"], item.get("mode", "stocks")
    st.subheader(f"{ticker} · {item['signal']}")
    st.caption(
        f"Cena {display_number(item['price'], 2, currency=True)} · Vstupní zóna {display_number(item['zone_low'], currency=True)}–"
        f"{display_number(item['zone_high'], currency=True)} · Preferovaný vstup {display_number(item['preferred_entry'], currency=True)}"
    )

    quote = item.get("extended_quote")
    if quote:
        st.caption(f"Signální den: {display_date(item['data'].index[-1])} (dokončený). "
                   f"Referenční {quote['kind']} cena: {display_number(quote['price'], currency=True)} "
                   f"· čas {quote['asof']}. Nejde o živou garantovanou kotaci.")
    else:
        st.caption(f"Signální den: {display_date(item['data'].index[-1])}. "
                   "Intraday/PRE/POST nejsou k dispozici; denní data jsou dokončená.")
    st.write(f"Počet potvrzených testů podpory: {item['support_touches']}; "
             f"vzdálenost {display_number(item['support_distance_atr'])} ATR "
             f"({display_number(item['support_distance_pct'])} %).")
    if mode == "crops":
        render_crop_panel(item)
    else:
        render_events_panel(ticker, item["corporate_events"])
    render_seasonality_panel(ticker, item["seasonality_context"])
    # Síťové doplňky se načítají jen po kliknutí v konkrétní ticker záložce.
    if mode == "stocks" and st.button("Načíst název, cenu před otevřením trhu a zprávy", key=f"load_context_{ticker}"):
        st.session_state[f"context_loaded_{ticker}"] = True
    if mode == "stocks" and st.session_state.get(f"context_loaded_{ticker}", False):
        name, sector = basic_info(ticker)
        quote = item.get("extended_quote")
        st.markdown(f"**{name}** · Sektor: {sector}")
        if quote:
            st.metric(f"Referenční cena {quote['kind']} (PRE/REG/POST)",
                      display_number(quote['price'], 2, currency=True),
                      f"{display_number((quote['price']/item['price']-1)*100, 2, sign=True)} % vůči poslednímu dokončenému dni")
            st.caption(f"Čas 5min baru: {quote['asof']}. Zpožděná kotace Yahoo; nejedná se o živý závazný kurz.")
        else:
            st.caption("Aktuální mimoburzovní cena není dostupná; výpočty vycházejí z dokončených denních dat.")
        news = get_news_context(ticker)
        st.markdown("#### 📰 Zprávy a jejich vyznění")
        st.caption(
            f"{news['sentiment']} (orientační hodnocení zpráv: {news['score']:+d}). "
            "Hodnocení vychází z klíčových slov a frází; neposuzuje význam celého článku."
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
    k1.metric("Kvalita", f"{display_number(item['quality_score'], 1)}/100")
    k2.metric("Vstupní skóre", f"{display_number(item['entry_score'], 1)}/100")
    k3.metric("Trend", f"{display_number(item['trend_score'], 1)}/100")
    k4.metric("Síla pohybu", f"{display_number(item['momentum_score'], 1)}/100")
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Relativní síla vůči koši ETC (30 seancí)" if mode == "crops" else "Relativní síla vůči SPY (30 seancí)", f"{display_number(item['rs_30d'], 2, sign=True)} %")
    k6.metric("RSI (14)", f"{display_number(item['rsi'], 1)}")
    k7.metric("ATR", f"{display_number(item['atr'], 2, currency=True)} ({display_number(item['atr_pct'], 2)} %)")
    k8.metric("Objem vůči průměru", f"{display_number(item['volume_ratio'], 2)}×" if np.isfinite(item["volume_ratio"]) else "Nedostupné")
    if not np.isfinite(item["volume_ratio"]):
        st.warning("Objemová data nejsou dostatečná pro výpočet poměru objemů. Podmínky likvidity nadále blokují nový nákup.")

    g1, g2 = st.columns(2)
    g1.metric("Potenciál k D1 High za 20 seancí / 1 USD", f"{display_number(item['gain_per_dollar'], 3, currency=True)}")
    g2.metric("Potenciál k High za 30 kal. dnů / 1 USD", f"{display_number(item['gain_30c_per_dollar'], 3, currency=True)}")
    st.caption("20s = maximum denní High za 20 seancí. 30d = maximum High za 30 kalendářních "
               "dnů, včetně dostupných PRE/POST hodinových svíček. "
               f"Zdroj 30d: {item.get('gain_30_source', 'nezjištěn')}. "
               "Oba ukazatele jsou historické vzdálenosti, nikoli předpověď zisku; "
               "mohou být shodné, pokud mají stejné maximum.")
    render_personal_decision(item, capital, risk_pct, max_position_pct)

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
    for col, label in [("BB_UPPER", "Horní Bollingerovo pásmo"), ("BB_LOWER", "Dolní Bollingerovo pásmo")]:
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
                      annotation_text="Oblast podpory")
    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.05},
        template="plotly_white", separators=", ",
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
        x=d.index, y=d["MACD_HIST"], name="Histogram MACD",
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
        showlegend=True, template="plotly_white", separators=", "
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"indicators_chart_{ticker}")

def render_history(sb, mode="stocks", holding_days=5):
    st.subheader("Uložené signály")
    if sb is None:
        st.info("Historie vyžaduje připojení k Supabase.")
        return
    left, right = st.columns(2)
    limit = left.number_input("Maximální počet řádků", 50, 2000, 500, 50)
    only_today = right.checkbox("Pouze signály s dnešním datem", False)
    if st.button("Načíst historii", key="load_history"):
        try:
            query = sb.table("scanner_signals").select("*").order("signal_date", desc=True)
            if mode == "crops":
                query = query.like("signal", "PLODINY ·%").like("signal", f"%{holding_days}D%")
            else:
                query = query.not_.like("signal", "PLODINY ·%")
            if only_today:
                query = query.eq("signal_date", local_today().isoformat())
            st.session_state.history_df = pd.DataFrame(query.limit(int(limit)).execute().data or [])
        except Exception:
            log.exception("Načtení historie selhalo")
            st.error("Historii se nepodařilo načíst. Ověřte připojení a oprávnění databáze.")
    hist = st.session_state.get("history_df")
    if hist is not None:
        if hist.empty:
            st.info("Nejsou dostupné uložené signály pro zvolený filtr.")
        else:
            st.dataframe(localize_table(hist), use_container_width=True, hide_index=True)
            st.download_button("Stáhnout historii (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                               "klondike_historie.csv", "text/csv")


def render_learning(sb, mode="stocks", holding_days=5):
    horizon = holding_days
    column = f"forward_return_{horizon}d"
    st.subheader("Vyhodnocení historických signálů")
    st.caption(f"Simulace nákupních signálů jádra 4.8: vstup při otevření další seance v nákupní zóně, "
               f"kontrola R:R a výstup nejpozději {horizon}. obchodní den. Bez nákladů. "
               "Nejde o učení parametrů ani ověření osobní strategie. Historická opční data, počasí ani sezónnost se neobnovují.")
    if sb is None:
        st.info("Vyhodnocení signálů vyžaduje připojení k Supabase.")
        return
    days = st.slider("Délka historie (dny)", 30, 365, 90, key="learning_days")
    if st.button("Vyhodnotit signály", key="run_learning"):
        with st.spinner("Vyhodnocuji historické signály…"):
            st.session_state.learning_df = load_learning_stats(sb, days=days, mode=mode, holding_days=holding_days)
    learning = st.session_state.get("learning_df")
    if learning is None:
        return
    if learning.empty:
        st.info("Nejsou k dispozici žádná data k vyhodnocení.")
        return
    counts = learning["outcome"].value_counts()
    outcomes = [("TP1", "TP1"), ("Stop", "SL"), ("Nejasné pořadí výstupů", "AMBIGUOUS"),
                (f"Výstup {horizon}. den", "TIME_EXIT"), ("Bez vstupu", "NOT_FILLED")]
    for col, (label, key) in zip(st.columns(5), outcomes):
        col.metric(label, int(counts.get(key, 0)))
    pending = int(counts.get("pending", 0))
    if pending:
        st.caption(f"Signály čekající na dokončení: {pending}.")
    resolved = int(counts.get("TP1", 0)) + int(counts.get("SL", 0))
    if resolved:
        st.metric("TP1 / (TP1 + SL)", f"{display_number(counts.get('TP1', 0) / resolved * 100, 1)} %",
                  help="Nezahrnuje časové výstupy, výstupy s nejasným pořadím ani neuskutečněné obchody.")
    valid = learning.dropna(subset=[column])
    if not valid.empty:
        st.markdown("#### Průměrný výnos podle výsledku")
        st.dataframe(localize_table(valid.groupby("outcome")[column].agg(["count", "mean"]).round(2)), use_container_width=True)
        st.markdown("#### Průměrný výnos podle signálu")
        st.dataframe(localize_table(valid.groupby("signal")[column].agg(["count", "mean"]).round(2)), use_container_width=True)
    st.markdown("#### Detailní data")
    st.dataframe(localize_table(learning), use_container_width=True, hide_index=True)
    st.download_button("Stáhnout vyhodnocení (CSV)", learning.to_csv(index=False).encode("utf-8-sig"),
                       "klondike_learning.csv", "text/csv")


# ---------------------- STAV OBOU ČÁSTÍ APLIKACE ----------------
PROFILE_DEFAULTS = {
    "results": [], "regime": None, "regime_score": 0, "scan_has_run": False,
    "scan_total": 0, "scan_finished": "—", "save_status": "",
    "history_df": None, "learning_df": None, "scan_diagnostics": {}, "scanned_tickers": [],
    "adaptive_save_status": "",
}


def activate_profile(mode, holding_days):
    """Oddělené výsledky a historie pro akcie a každý horizont plodin."""
    profile_key = f"{mode}_{holding_days}"
    profiles = st.session_state.setdefault("scan_profiles", {})
    previous = st.session_state.get("active_profile")
    if previous == profile_key:
        return
    if previous:
        profiles[previous] = {key: st.session_state.get(key, default)
                             for key, default in PROFILE_DEFAULTS.items()}
    restored = profiles.get(profile_key, {})
    for key, default in PROFILE_DEFAULTS.items():
        st.session_state[key] = restored.get(key, default)
    st.session_state.active_profile = profile_key


def scan_weather_contexts(tickers, progress=None):
    """Obnoví původní porovnání předpovědí pouze na shodných budoucích dnech."""
    records = st.session_state.setdefault("weather_records", {})
    contexts = {}
    for crop in sorted({CROP_PRODUCTS[t]["crop"] for t in tickers}):
        if progress is not None:
            progress.progress(0.0, text=f"Načítám počasí: {CROP_NAMES[crop]}…")
        saved = records.get(crop, {})
        latest = saved.get("latest")
        context = prepare_crop_weather(crop, latest)
        if latest and context["fetched_at"] == latest.get("fetched_at"):
            context = calculate_crop_weather(crop, context["raw_regions"], saved.get("reference"))
        elif context["usable"]:
            records[crop] = {"latest": context["snapshot"], "reference": latest}
        contexts[crop] = context
    return contexts


def filter_exclusion_reasons(item, high_gain, outperforming, safe_earnings, high_gain30=True):
    reasons = []
    if high_gain and safe_float(item.get("gain_per_dollar"), -1) < .06:
        reasons.append("Potenciál k D1 High za 20 seancí nedosahuje 0,06 USD / 1 USD.")
    if high_gain30 and safe_float(item.get("gain_30c_per_dollar"), -1) < .15:
        reasons.append("Potenciál k High za 30 kalendářních dnů nedosahuje 0,15 USD / 1 USD.")
    if outperforming and safe_float(item.get("rs_30d"), 0) <= 0:
        reasons.append("Instrument nepřekonává srovnávací koš za 30 seancí.")
    if safe_earnings and earnings_in_window(item.get("corporate_events", {})):
        reasons.append("Termín výsledků spadá do příštích 7 dnů.")
    return reasons


def ticker_summary(item):
    quote = item.get("extended_quote")
    price_tag = quote["kind"] if quote else "D1"
    price = item.get("effective_price",item["price"])
    return (f"{item['ticker']} · {item['signal']} · {price_tag} {display_number(price,currency=True)}"
            f" · 20s High {display_number(item.get('gain_per_dollar',0),3)} USD/USD"
            f" · 30d High {display_number(item.get('gain_30c_per_dollar',0),3)} USD/USD"
            f" · R:R {display_number(item.get('rr1'),2)}"
            f" · styl {item.get('style_score', 0):.0f}"
            f" · priorita {item.get('adaptive_score',item['entry_score']):.1f}")


def render_scan_panels(raw, results, tickers, mode, capital, risk_pct, max_position_pct,
                       high_gain=False, outperforming=False, safe_earnings=False, high_gain30=True):
    """Vybrané plodiny mají panel i při chybě dat nebo nesplnění filtru."""
    known = {item["ticker"]: item for item in raw}
    visible = {item["ticker"]: item for item in results}
    panel_tickers = tickers if mode == "crops" else list(visible)
    for ticker in panel_tickers:
        label = f"{CROP_PRODUCTS[ticker]['name']} · {ticker}" if mode == "crops" else ticker
        item = visible.get(ticker)
        if item is None:
            filtered = known.get(ticker)
            state = "Nesplňuje filtry" if filtered is not None else "Analýza není dostupná"
            with st.expander(f"{label} · {state}", expanded=False):
                if filtered is not None:
                    for reason in filter_exclusion_reasons(filtered, high_gain, outperforming, safe_earnings, high_gain30):
                        st.info(reason)
                    st.caption("Pro zobrazení úplné analýzy vypněte příslušný filtr. Filtry nemění technická pravidla nákupu.")
                else:
                    scanned = st.session_state.get("scanned_tickers", [])
                    message = (st.session_state.get("scan_diagnostics", {}).get(ticker)
                               if ticker in scanned else "Tato plodina nebyla zahrnuta do posledního skenu. Spusťte nový sken.")
                    st.warning(message or "Pro tento instrument nejsou dostupná použitelná data. Zopakujte sken.")
                    st.caption("Bez úplné analýzy se nevytváří nákupní signál a instrument se neukládá do databáze.")
            continue
        assessment = assess_trade_action(item, held=st.session_state.get(f"held_{mode}_{ticker}", False),
            position_stop=st.session_state.get(f"held_stop_{mode}_{ticker}"),
            position_target=st.session_state.get(f"held_target_{mode}_{ticker}"))
        with st.expander(ticker_summary(item), expanded=False):
            render_instrument(item, capital, risk_pct, max_position_pct)


def main():
    st.set_page_config(page_title="Klondike · Akcie a zemědělské komodity", page_icon="📈", layout="wide",
                       initial_sidebar_state="expanded")
    apply_design()
    st.caption("Verze 2026-10-24-R5 · Osobní filtr, adaptivní učení, PRE/POST")
    mode_label = st.radio("Část aplikace", ["Akcie", "Zemědělské komodity"], horizontal=True, key="asset_mode")
    mode = "crops" if mode_label == "Zemědělské komodity" else "stocks"
    horizon = st.selectbox("Maximální doba držení (obchodní dny)", [5, 10, 20], index=1,
                           key="crop_horizon") if mode == "crops" else 5
    activate_profile(mode, horizon)
   


  
    st.caption(f"Technická hypotéza z dokončených denních svíček · horizont {horizon} seancí · "
               "ceny PRE/POST jsou oddělené a slouží ke kontrole aktuálního vstupu. "
               "W1/H4 a intradenní potvrzení aplikace automaticky nevyhodnocuje.")
    sb = get_supabase()
    adaptive_sb = get_adaptive_supabase()
    with st.sidebar:
        if mode == "stocks":
            st.header("🔎 Výběr akcií")
            custom = st.text_input("Přidat burzovní symbol (ticker)", "", placeholder="Např. NFLX nebo AVGO",
                                   key="custom_ticker").upper().strip()
            groups = st.multiselect("Sledované skupiny", list(STOCK_GROUPS),
                                    default=list(STOCK_GROUPS), key="stock_groups",
                                    help="Seznam rozšířen o banky, pojišťovny, těžaře, distributory "
                                         "a kandidáty střední kapitalizace. Likvidita se ověřuje při skenu.")
            suggested = list(dict.fromkeys(t for group in groups for t in STOCK_GROUPS[group]))
            tickers = st.multiselect("Tickery pro tento sken (omezíte tím počet požadavků)",
                                    suggested, default=[t for t in DEFAULT_TICKERS if t in suggested],
                                    key="stock_selected_tickers")
            if custom:
                if is_valid_ticker(custom):
                    if custom not in tickers:
                        tickers.insert(0, custom)
                else:
                    st.warning("Ticker: nejvýše 10 znaků A–Z, 0–9, tečka nebo pomlčka.")
        else:
            st.header("🌾 Výběr plodin")
            tickers = st.multiselect("Plodiny – ETC obchodovaná v Londýně v USD", list(CROP_PRODUCTS),
                                    default=list(CROP_PRODUCTS), key="crop_tickers",
                                    format_func=lambda t: f"{CROP_PRODUCTS[t]['name']} · {t}")
            st.caption("Ceny patří burzovním produktům ETC navázaným na futures, nikoli fyzickým plodinám.")
            st.caption("Panel každé vybrané plodiny zůstává viditelný i při chybě dat nebo nesplnění filtru a vysvětlí důvod.")
        st.divider()
        st.subheader("Filtry zobrazení")
        high_gain = st.toggle("20 seancí · zisk / 1 USD ≥ 0,06 USD", value=False,
                              key=f"filter_high_gain_{mode}",
                              help="Prostor k nejvyšší dokončené závěrečné ceně za 20 seancí, nikoli predikce.")
        high_gain30 = st.toggle("30 kal. dnů · zisk / 1 USD ≥ 0,15 USD", value=True,
                              key=f"filter_high_gain30_{mode}",
                              help="Prostor k nejvyšší dokončené závěrečné ceně za posledních 30 kalendářních dnů "
                                   "proti poslední referenční ceně. Oba filtry lze kombinovat.")
        outperforming = st.toggle("Pouze instrumenty překonávající koš ETC" if mode == "crops" else "Pouze akcie překonávající S&P 500",
                                  value=False, key=f"filter_outperforming_{mode}",
                                  help="Kladná relativní síla za 30 obchodních seancí vůči koši ETC, nebo vůči ETF SPY u akcií.")
        safe_earnings = False
        if mode == "stocks":
            safe_earnings = st.toggle("Skrýt akcie s výsledky do 7 dnů", value=True, key="filter_safe_earnings",
                                     help="Včetně dneška a sedmého dne. Odhadované rozmezí se posuzuje celé. Akcie s neznámým termínem zůstávají v přehledu.")
            st.caption("Akcie s neznámým termínem výsledků zůstávají v přehledu. U těchto akcií nelze ověřit sedmidenní odstup.")
        else:
            st.caption("Srovnávací koš tvoří všechna tři ETC se stejnou váhou. Sledovaný instrument je součástí koše.")
            st.caption("Počasí: Open-Meteo, soubor předpovědí GFS. Chybějící, neúplná nebo starší data blokují nový nákup.")
        st.divider()
        st.caption("Zdroj cen: Yahoo Finance, včetně dostupných PRE/POST 5minutových hodnot pro akcie. "
                   "Referenční kotace může být zpožděná. Změna filtrů upraví již načtený přehled.")
        st.caption("⚪ Supabase historie není připojena." if sb is None else "🟢 Supabase historie je připojena.")
        st.caption("⚪ Adaptivní učení nemá zabezpečené DB připojení." if adaptive_sb is None
                   else "🟢 Adaptivní databáze připojena (server-side).")
    tab_scan, tab_history, tab_learning = st.tabs([
        "🌾 Přehled plodin" if mode == "crops" else "🔎 Přehled akcií", "🗄️ Historie signálů", "🧠 Vyhodnocení signálů"])
    with tab_scan:
        with st.expander("💰 Nastavení velikosti pozice", expanded=False):
            c1, c2, c3 = st.columns(3)
            capital = c1.number_input("Kapitál (USD)", min_value=100.0, value=24000.0, step=500.0, key="capital")
            risk_pct = c2.slider("Riziko na obchod (%)", 0.25, 3.0, 2.0, 0.25, key="risk_pct")
            max_position_pct = c3.slider("Maximální velikost pozice (%)", 5, 100, 25, 5, key="max_position_pct")
            st.caption("2 % kapitálu je pracovní rizikový limit pro CELOU pozici; nikoli pro každou tranši zvlášť.")
        left, right = st.columns([3, 1])
        left.subheader("Tržní sken")
        run = right.button("Spustit sken plodin" if mode == "crops" else "Spustit sken akcií",
                           type="primary", use_container_width=True, key="run_scan", disabled=not tickers)
        if run:
            progress = st.progress(0.0, text="Načítám ceny a počasí…" if mode == "crops" else "Načítám srovnávací index a analyzuji akcie…")
            def update_progress(done, total):
                progress.progress(done / max(total, 1), text=f"Analyzuji {done}/{total}")
            try:
                spy = get_crop_benchmark() if mode == "crops" else get_spy()
                weather_contexts = scan_weather_contexts(tickers, progress) if mode == "crops" else {}
                regime, score = market_regime(spy)
                diagnostics = {}
                raw = parallel_scan(tickers, spy, max_workers=4, progress_cb=update_progress,
                                    mode=mode, weather_contexts=weather_contexts, holding_days=horizon,
                                    diagnostics=diagnostics)
                update_adaptive_outcomes(adaptive_sb)
                learned_history = load_adaptive_resolved(adaptive_sb, mode, horizon)
                raw = apply_adaptive_ranking(raw, learned_history)
                snapshots_ok = save_adaptive_snapshots(adaptive_sb, raw)
                st.session_state.adaptive_save_status = ("Adaptivní snímky uloženy." if snapshots_ok
                    else "Adaptivní snímky nejsou ukládány (chybí služba, migrace nebo oprávnění).")
                st.session_state.update(results=raw, regime=regime, regime_score=score, scan_has_run=True,
                                        scan_total=len(tickers), scan_diagnostics=diagnostics, scanned_tickers=list(tickers), scan_finished=datetime.now(ZoneInfo("Europe/Prague")).strftime("%d. %m. %Y %H:%M"))
                visible = filter_results(raw, high_gain, outperforming, safe_earnings, high_gain30)
                # Jeden zápis na explicitní sken; změny widgetů zápis neopakují.
                saved = save_signals_bulk(sb, visible)
                if sb is None:
                    st.session_state.save_status = "Sken dokončen bez zápisu: Supabase není připojena."
                elif not visible:
                    st.session_state.save_status = "Aktivním filtrům nevyhověl žádný signál; databáze nebyla změněna."
                elif saved:
                    st.session_state.save_status = f"Počet signálů uložených do Supabase: {saved}."
                    st.toast(st.session_state.save_status, icon="💾")
                else:
                    st.session_state.save_status = "Zápis do Supabase selhal. Ověřte připojení, tabulku a oprávnění."
                    st.error(st.session_state.save_status)
                if spy.empty:
                    st.warning("Srovnávací cenová data nejsou dostupná; nový vstup je blokován.")
            except Exception:
                log.exception("Sken selhal")
                st.error("Sken se nepodařilo dokončit. Zkuste jej spustit znovu.")
            finally:
                progress.empty()
        if st.session_state.regime:
            icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(st.session_state.regime, "⚪")
            regime_label = {"BULLISH": "Rostoucí", "BEARISH": "Klesající", "NEUTRAL": "Neutrální", "UNKNOWN": "Nelze určit"}.get(st.session_state.regime, "Nelze určit")
            benchmark = "koš zemědělských ETC" if mode == "crops" else "SPY"
            st.info(f"{icon} **Tržní režim ({benchmark}):** {regime_label} · skóre {st.session_state.regime_score}/5")
        raw = st.session_state.results
        results = filter_results(raw, high_gain, outperforming, safe_earnings, high_gain30)
        if st.session_state.scan_has_run:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzované plodiny" if mode == "crops" else "Analyzované akcie", len(raw))
            c2.metric("Instrumenty splňující filtry" if mode == "crops" else "Akcie splňující filtry", len(results))
            c3.metric("Nákupní zóna", sum(item["signal"] == buy_signal(mode, horizon) for item in results))
            c4.metric("Průměrná kvalita", display_number(np.mean([i["quality_score"] for i in results]), 1) if results else "—")
            st.caption(f"Poslední sken: {st.session_state.get('scan_finished', '—')} · "
                       f"{st.session_state.get('save_status','')} · "
                       f"{st.session_state.get('adaptive_save_status','')}")
            skipped = st.session_state.get("scan_total", len(raw)) - len(raw)
            if skipped:
                st.caption(f"Počet instrumentů bez dostatečných cenových dat nebo s chybou: {skipped}.")
            if not results:
                if raw:
                    st.info("Aktivním filtrům nevyhovuje žádný instrument. Pro širší přehled vypněte některý filtr.")
                else:
                    st.warning("Pro vybrané instrumenty se nepodařilo dokončit analýzu. Podrobnosti najdete v jejich panelech.")
            if results or mode == "crops":
                st.markdown("### Podrobnosti jednotlivých plodin" if mode == "crops" else "### Podrobnosti jednotlivých akcií")
                st.caption("Rozbalením panelu zobrazíte analýzu nebo důvod její nedostupnosti.")
                render_scan_panels(raw, results, tickers, mode, capital, risk_pct, max_position_pct,
                                   high_gain, outperforming, safe_earnings, high_gain30)
            if mode == "stocks" and st.session_state.get("scan_diagnostics"):
                with st.expander("Instrumenty s neúplnými daty", expanded=False):
                    for ticker, reason in st.session_state.scan_diagnostics.items():
                        st.write(f"**{ticker}**: {reason}")
        else:
            st.info("Spusťte sken. Výsledky se zobrazí v rozbalovacích panelech jednotlivých instrumentů.")
    with tab_history:
        render_history(sb, mode, horizon)
    with tab_learning:
        render_adaptive_learning(adaptive_sb, mode, horizon)
        with st.expander("Původní kontrolní simulace historických signálů", expanded=False):
            render_learning(sb, mode, horizon)


if __name__ == "__main__":
    main()
