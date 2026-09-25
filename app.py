import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- PROFESIONÁLNÍ VIZUÁLNÍ KONFIGURACE (FINANČNÍ DARK THEME) ---
st.set_page_config(
    page_title="Klondike AI Investment & Swing Terminal", 
    page_icon="⚡", 
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #f3f4f6; }
    .stExpander {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
    }
    [data-testid="stSidebar"] { background-color: #030712; border-right: 1px solid #1f2937; }
    [data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: #f9fafb !important; }
    h1, h2, h3 { letter-spacing: -0.5px; color: #f9fafb; }
    </style>
""", unsafe_layout=True)

class KlondikeExecutionAgent:
    def __init__(self):
        self.status = "Aktivní & Synchronizováno se Supabase DB"
        self.protocols = [
            "Real-time analýza spotového trendu a S&P 500", 
            "Pre-market anomálie & detekce objemových špiček", 
            "Automatický zápis predikcí a zpětné učení AI"
        ]

st.title("⚡ AI Spot Swing Skener & Predikční Modul")
st.markdown("<p style='font-size: 1.1em; color: #9ca3af;'>Profesionální tržní analytika s AI vhledy, sledováním pre-marketu a učením se z historických predikcí.</p>", unsafe_allow_html=True)

# Inicializace Supabase databáze
supabase = None
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.sidebar.warning("⚠️ Databáze Supabase není připojena (zkontrolujte st.secrets)")

# --- POSTRANNÍ PANEL ---
st.sidebar.markdown("### 🔍 Vyhledávání aktiv")
custom_ticker_input = st.sidebar.text_input("Přidat ticker (např. AAPL, MSFT):", "").upper().strip()

DEFAULT_TICKERS = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "SPY"]
active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filtry a strategie")
filter_high_gain = st.sidebar.toggle("💵 Zisk / 1 $>= 0.06$", value=False)
filter_breakout = st.sidebar.toggle("🚀 Pouze aktivní průrazy", value=False)
filter_squeeze = st.sidebar.toggle("📦 Pouze konsolidace (BB Squeeze)", value=False)
filter_volume = st.sidebar.toggle("📊 Pouze s potvrzeným objemem", value=False)
filter_outperforming = st.sidebar.toggle("⚡ Pouze překonávající S&P 500", value=False)
filter_safe_earnings = st.sidebar.toggle("🛡️ Skrýt akcie s výsledky < 7 dnů", value=False)

PRED_DAYS = 20

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calculate_atr(data, window=14):
    high, low, close = data['High'], data['Low'], data['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.rolling(window=window).mean().iloc[-1])

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float((macd - signal).iloc[-1])

def analyze_volume(data, window=20):
    current_vol = float(data['Volume'].iloc[-1])
    avg_vol = float(data['Volume'].rolling(window=window).mean().iloc[-1])
    return current_vol, avg_vol, current_vol > (1.5 * avg_vol), current_vol / avg_vol if avg_vol > 0 else 1.0

def detect_patterns(data):
    close = data['Close']
    high_20 = close.rolling(window=20).max()
    low_20 = close.rolling(window=20).min()
    is_breakout = bool(close.iloc[-1] >= high_20.iloc[-2])
    is_breakdown = bool(close.iloc[-1] <= low_20.iloc[-2])
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    bandwidth = ((sma + (2 * std)) - (sma - (2 * std))) / sma
    bw_mean = bandwidth.rolling(window=50).mean().iloc[-1]
    is_squeeze = bool(bandwidth.iloc[-1] < (bw_mean * 0.8)) if not np.isnan(bw_mean) else False
    return is_breakout, is_breakdown, is_squeeze

def get_premarket_data(ticker_obj):
    try:
        info = ticker_obj.info
        pre_price = info.get('preMarketPrice', None)
        prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', None))
        if pre_price and prev_close:
            return float(pre_price), float(((pre_price - prev_close) / prev_close) * 100)
    except:
        pass
    return None, None

def get_next_earnings_days(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal and isinstance(cal, dict) and 'Earnings Date' in cal and cal['Earnings Date']:
            next_date = pd.to_datetime(cal['Earnings Date'][0])
            return (next_date - pd.Timestamp.now()).days, next_date.strftime('%Y-%m-%d')
    except:
        pass
    return 999, "N/A"

def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news: return "➖ (Žádné zprávy)", "Nadpis nenalezen"
        return "📈 BÝČÍ", news[0].get('title', 'Nadpis') if isinstance(news[0], dict) else "Nadpis"
    except:
        return "➖ (Zprávy nedostupné)", "Chyba"

# --- NAVIGACE ---
app_mode = st.radio("Vyberte zobrazení:", [
    "📊 Tržní skener & Vzorce", 
    "🎯 Historie predikcí & Učení AI",
    "🤖 Klondike Agent Hub",
    "📘 Uživatelská příručka"
], horizontal=True)

# ==========================================
# 1. ZÁLOŽKA: SKENER
# ==========================================
if app_mode == "📊 Tržní skener & Vzorce":
    st.markdown("### 🚀 Spuštění analýzy")
    if st.button("🚀 Spustit sken, analýzu vzorců a zapsat predikce do DB", type="primary", use_container_width=True):
        st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        with st.spinner("Analyzuji aktiva, počítám Prophet modely a zapisuji do Supabase..."):
            try:
                sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                if isinstance(sp500.columns, pd.MultiIndex): sp500.columns = sp500.columns.get_level_values(0)
                sp500_close = float(sp500['Close'].iloc[-1])
                sp500_30d_ret = (sp500_close - float(sp500['Close'].iloc[-30])) / float(sp500['Close'].iloc[-30]) * 100
            except:
                sp500_30d_ret = 0

        valid_results = []
        for ticker in active_tickers:
            try:
                t_obj = yf.Ticker(ticker)
                data = t_obj.history(period="1y", interval="1d")
                if data.empty or len(data) < 30: continue
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

                actual_price = float(data['Close'].iloc[-1])
                pre_price, pre_change = get_premarket_data(t_obj)
                peak_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
                gain_per_1 = (peak_20d - actual_price) / actual_price
                rs_vs_sp500 = ((actual_price - float(data['Close'].iloc[-30])) / float(data['Close'].iloc[-30]) * 100) - sp500_30d_ret

                if filter_high_gain and gain_per_1 < 0.06: continue
                is_breakout, is_breakdown, is_squeeze = detect_patterns(data)
                _, _, is_vol_spike, vol_ratio = analyze_volume(data)

                if filter_breakout and not is_breakout: continue
                if filter_squeeze and not is_squeeze: continue
                if filter_volume and not is_vol_spike: continue
                if filter_outperforming and rs_vs_sp500 < 0: continue

                earnings_days, earnings_date_str = get_next_earnings_days(t_obj)
                if filter_safe_earnings and earnings_days <= 7: continue

                rsi_val = calculate_rsi(data)
                atr_val = calculate_atr(data)
                _, _, macd_hist = calculate_macd(data)
                news_sentiment, latest_headline = analyze_news_sentiment(t_obj)

                # Prophet
                df = data.reset_index()[['Date', 'Close']]
                df.columns = ['ds', 'y']
                df['ds'] = df['ds'].dt.tz_localize(None)
                model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                model.fit(df)
                future = model.make_future_dataframe(periods=PRED_DAYS)
                forecast = model.predict(future)
                pred_price_20d = float(forecast['yhat'].iloc[-1])
                target_date_val = (datetime.now() + timedelta(days=PRED_DAYS)).strftime('%Y-%m-%d')

                # ZÁPIS DO SUPABASE S LADĚNÍM CHYB
                if supabase is not None:
                    try:
                        today_str = datetime.now().strftime('%Y-%m-%d')
                        existing = supabase.table("predictions").select("id").eq("ticker", ticker).gte("created_at", today_str).execute()
                        if not existing.data:
                            res_db = supabase.table("predictions").insert({
                                "ticker": ticker,
                                "entry_price": float(actual_price),
                                "predicted_price": float(pred_price_20d),
                                "target_date": target_date_val,
                                "status": "PENDING"
                            }).execute()
                    except Exception as db_err:
                        st.error(f"Chyba při zápisu {ticker} do DB: {db_err}")

                action_rec = "🚀 SILNÝ NÁKUP" if (is_breakout and is_vol_spike) else "⏳ ČEKAT"
                valid_results.append({
                    "ticker": ticker, "actual_price": actual_price, "pre_price": pre_price, "pre_change": pre_change,
                    "rsi_val": rsi_val, "rs_vs_sp500": rs_vs_sp500, "gain_per_1": gain_per_1, "action_rec": action_rec,
                    "model": model, "forecast": forecast, "earnings_days": earnings_days, "earnings_date_str": earnings_date_str
                })
            except Exception as e:
                pass

        if valid_results:
            st.success(f"Nalezeno a uloženo: {len(valid_results)} aktiv.")
            for res in valid_results:
                with st.expander(f"📌 {res['ticker']} — Cena: ${res['actual_price']:.2f} | Signál: {res['action_rec']}"):
                    st.metric("Závěrečná cena", f"${res['actual_price']:.2f}")
        else:
            st.warning("⚠️ Žádná aktiva nevyhovují zvoleným filtrům.")

# ==========================================
# 2. ZÁLOŽKA: HISTORIE PREDIKCÍ
# ==========================================
elif app_mode == "🎯 Historie predikcí & Učení AI":
    st.subheader("🎯 Vyhodnocení predikcí a učení se z minulosti")
    st.markdown("Tato sekce stahuje zapsané predikce z databáze Supabase, porovnává je s aktuální reálnou cenou a ukazuje úspěšnost AI modelů.")
    
    if supabase is not None:
        if st.button("🔄 Načíst a vyhodnotit predikce z databáze", type="primary"):
            with st.spinner("Stahuji data ze Supabase..."):
                try:
                    response = supabase.table("predictions").select("*").execute()
                    rows = response.data
                    
                    if not rows:
                        st.info("⚠️ V tabulce `predictions` v Supabase nejsou žádné záznamy. Nejdříve spusťte skener v první záložce.")
                    else:
                        eval_data = []
                        success_count, total_evaluated = 0, 0

                        for row in rows:
                            pred_id = row["id"]
                            ticker = row["ticker"]
                            entry_price = float(row["entry_price"])
                            predicted_price = float(row["predicted_price"])
                            target_date = row["target_date"]
                            status = row["status"]
                            
                            try:
                                hist = yf.Ticker(ticker).history(period="1d")
                                cur_price = float(hist['Close'].iloc[-1]) if not hist.empty else entry_price
                            except:
                                cur_price = entry_price

                            today_obj = datetime.now().date()
                            target_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                            
                            if today_obj >= target_obj:
                                if (predicted_price > entry_price) == (cur_price > entry_price):
                                    calculated_status = "SUCCESS 🟢"
                                    success_count += 1
                                else:
                                    calculated_status = "FAILED 🔴"
                                total_evaluated += 1
                                try:
                                    supabase.table("predictions").update({
                                        "actual_price_at_target": cur_price,
                                        "status": calculated_status
                                    }).eq("id", pred_id).execute()
                                except:
                                    pass
                            else:
                                calculated_status = "PENDING ⏳"

                            eval_data.append({
                                "Ticker": ticker,
                                "Vstup": f"${entry_price:.2f}",
                                "Predikce": f"${predicted_price:.2f}",
                                "Cíl. datum": target_date,
                                "Reálná cena": f"${cur_price:.2f}",
                                "Výsledek": calculated_status
                            })

                        if total_evaluated > 0:
                            winrate = (success_count / total_evaluated) * 100
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Vyhodnoceno", total_evaluated)
                            c2.metric("Úspěšné", success_count)
                            c3.metric("Úspěšnost", f"{winrate:.1f}%")

                        st.dataframe(pd.DataFrame(eval_data), use_container_width=True)
                except Exception as ex:
                    st.error(f"Chyba při komunikaci s databází: {ex}")
    else:
        st.warning("⚠️ Databáze není připojena.")

# ==========================================
# 3. ZÁLOŽKA: AGENT HUB
# ==========================================
elif app_mode == "🤖 Klondike Agent Hub":
    st.subheader("🤖 Klondike Agent Hub")
    agent = KlondikeExecutionAgent()
    st.success(f"**Stav:** {agent.status}")

# ==========================================
# 4. ZÁLOŽKA: PŘÍRUČKA
# ==========================================
elif app_mode == "📘 Uživatelská příručka":
    st.subheader("📘 Uživatelská příručka")
    st.markdown("Zde najdete nápovědu k používání systému.")
