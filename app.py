import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Investment Scanner", page_icon="🤖", layout="wide")

# --- NAČTENÍ PŘEKLADŮ ZE SOUBORU ---
@st.cache_data
def load_translations():
    if os.path.exists("translations.json"):
        with open("translations.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Záložní fallback, kdyby soubor chyběl
        return {
            "Čeština": {
                "title": "🤖 Klondike AI Investment Scanner",
                "desc": "Chybí soubor translations.json!",
                "btn": "Spustit",
                "spinner": "Načítám...",
                "macro_warn": "Varování",
                "macro_ok": "OK",
                "macro_err": "Chyba",
                "menu_label": "Navigace",
                "menu_dashboard": "Dashboard",
                "menu_manual": "Manuál",
                "menu_support": "Podpora"
            }
        }

translations = load_translations()
languages = list(translations.keys())

# --- ČTENÍ JAZYKA Z URL (Klíčové pro spolehlivost uvnitř <iframe>) ---
query_params = st.query_params
url_lang = query_params.get("lang")

if url_lang in languages:
    st.session_state.lang = url_lang
elif 'lang' not in st.session_state:
    st.session_state.lang = "Čeština" if "Čeština" in languages else languages[0]

# --- SIDEBAR: VÝBĚR JAZYKA ---
st.sidebar.markdown("### 🌍 Language / Jazyk")
selected_lang = st.sidebar.selectbox(
    "Choose language / Zvolte jazyk:", 
    languages, 
    index=languages.index(st.session_state.lang) if st.session_state.lang in languages else 0,
    label_visibility="collapsed",
    key="lang_selector"
)

# Pokud se jazyk změnil, uložíme ho do URL a vynutíme okamžitý refresh
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    st.query_params["lang"] = selected_lang
    st.rerun()

t = translations[st.session_state.lang]

st.sidebar.markdown("---")

# --- SIDEBAR: PŘEPÍNAČ SEKCÍ (RADIO TLAČÍTKA) ---
st.sidebar.markdown(f"### {t['menu_label']}")
menu_options_dict = {
    t['menu_dashboard']: "dashboard",
    t['menu_manual']: "manual",
    t['menu_support']: "support"
}

selected_menu_label = st.sidebar.radio(
    t['menu_label'],
    list(menu_options_dict.keys()),
    label_visibility="collapsed"
)
current_page = menu_options_dict[selected_menu_label]

# --- HLAVNÍ NADPISY ---
st.title(t["title"])
st.write(t["desc"])

# --- SUPABASE CONFIGURATION ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

# Configuration
TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "AMD", "GLD", "BTC-USD", "VT", "^GSPC", "ETH-USD", "SOL-USD", "QQQ", "SPY", "XRP-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]
PRED_DAYS = 20

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calculate_atr(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return float(atr.iloc[-1])

def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (No recent news)", "No available headlines"
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain"]
        
        score = 0
        latest_headline = "Unknown headline"
        
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if latest_headline == "Unknown headline" and title:
                latest_headline = title
            title_lower = title.lower()
            for kw in bullish_keywords:
                if kw in title_lower: score += 1
            for kw in bearish_keywords:
                if kw in title_lower: score -= 1
                    
        if score > 0: return "📈 BULLISH", latest_headline
        elif score < 0: return "📉 BEARISH", latest_headline
        else: return "➖ NEUTRAL", latest_headline
    except Exception:
        return "➖ (News unavailable)", "Error loading news"

# --- VYKRESLENÍ OBSAHU PODLE VYBRANÉ SEKCE ---

if current_page == "dashboard":
    if st.button(t["btn"], type="primary"):
        with st.spinner(t["spinner"]):
            
            # Macro status
            try:
                sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                sp500_close = float(sp500['Close'].iloc[-1])
                sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
                if sp500_close < sp500_sma50:
                    st.warning(t["macro_warn"])
                else:
                    st.success(t["macro_ok"])
            except:
                st.info(t["macro_err"])

            for ticker in TICKERS:
                with st.expander(f"Analysis for: {ticker}"):
                    try:
                        t_obj = yf.Ticker(ticker)
                        data = t_obj.history(period="1y", interval="1d")
                        if data.empty or len(data) < 30:
                            st.error(f"Insufficient data for {ticker}")
                            continue

                        news_sentiment, latest_headline = analyze_news_sentiment(t_obj)
                        rsi_val = calculate_rsi(data)
                        atr_val = calculate_atr(data)
                        sma_200 = float(data['Close'].rolling(window=200).mean().iloc[-1]) if len(data) >= 200 else float(data['Close'].mean())
                        
                        skutecna_cena = float(data['Close'].iloc[-1])
                        predchozi_cena = float(data['Close'].iloc[-2])
                        current_volume = float(data['Volume'].iloc[-1])
                        avg_volume_30d = float(data['Volume'].rolling(window=30).mean().iloc[-1])
                        
                        crowd_buying = current_volume > (avg_volume_30d * 2.0) and skutecna_cena > predchozi_cena
                        crowd_panicking = current_volume > (avg_volume_30d * 2.0) and skutecna_cena < predchozi_cena
                        is_bullish_trend = skutecna_cena > sma_200

                        vrchol_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
                        rozdil_usd = vrchol_20d - skutecna_cena
                        potencial_procent = (rozdil_usd / skutecna_cena) * 100
                        zisk_na_1_usd = rozdil_usd / skutecna_cena if skutecna_cena > 0 else 0

                        if rsi_val < 35:
                            verdict = "🟢 Verdict: OVERSOLD (ENTRY)"
                        elif rsi_val > 65:
                            verdict = "🔴 Verdict: OVERBOUGHT (CAUTION)"
                        else:
                            verdict = "🟡 Verdict: NEUTRAL (WAIT)"

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"{skutecna_cena:.2f} USD")
                        col2.metric("RSI (14)", f"{rsi_val:.1f}")
                        col3.metric("Profit / $1 Invested", f"+{zisk_na_1_usd:.2f} USD")

                        st.markdown(f"### {verdict}")
                        st.write(f"**News Sentiment:** {news_sentiment} | *\"{latest_headline}\"*")
                        st.write(f"**Distance to 20d Peak:** +{rozdil_usd:.2f} USD (+{potencial_procent:.2f}%)")
                        st.write(f"**ATR Volatility:** {atr_val:.2f}")
                        
                        if crowd_buying:
                            st.markdown("🔥 **Crowd Alert:** Mass buying detected (High volume + Price up)!")
                        elif crowd_panicking:
                            st.markdown("🚨 **Crowd Alert:** Panic selling detected (High volume + Price down)!")
                        else:
                            st.markdown("👥 **Crowd Behavior:** Calm / Normal volume.")

                        trend_status = "✅ OK (Bullish vs SMA200)" if is_bullish_trend else "❌ Below SMA200 (Caution)"
                        st.write(f"**Long-term Trend:** {trend_status}")

                        df = data.reset_index()[['Date', 'Close']]
                        df.columns = ['ds', 'y']
                        df['ds'] = df['ds'].dt.tz_localize(None)

                        model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                        model.fit(df)
                        future = model.make_future_dataframe(periods=PRED_DAYS)
                        forecast = model.predict(future)

                        predicted_price_20d = float(forecast.iloc[-1]['yhat'])
                        target_date = forecast.iloc[-1]['ds'].strftime('%Y-%m-%d')

                        if supabase:
                            try:
                                supabase.table("predictions").insert({
                                    "ticker": ticker,
                                    "predicted_price": round(predicted_price_20d, 2),
                                    "target_date": target_date,
                                    "actual_price_at_prediction": round(skutecna_cena, 2)
                                }).execute()
                                st.info(f"🧠 AI Learning: Prediction for {ticker} saved to database")
                            except Exception as db_err:
                                st.warning(f"Could not save to DB: {db_err}")

                        fig, ax = plt.subplots(figsize=(10, 4))
                        model.plot(forecast, ax=ax)
                        ax.set_title(f"Prediction for {ticker} (20 days ahead)")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error processing {ticker}: {e}")

elif current_page == "manual":
    html_manual = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: 'Helvetica', sans-serif; line-height: 1.6; color: #222; max-width: 3800px; margin: 40px auto; padding: 20px; }
            h1 { color: #1a3a5f; border-bottom: 3px solid #1a3a5f; padding-bottom: 10px; }
            h2 { color: #2c5e8e; margin-top: 30px; border-bottom: 1px solid #ccc; }
            .metric-box { background: #f8f9fa; border-left: 5px solid #2c5e8e; padding: 15px; margin: 10px 0; }
            pre { background: #eee; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>AI-Driven Quantitative Trading Scanner</h1>
        <p><strong>Technical Documentation & Operations Manual</strong></p>
        <h2>1. Executive Summary</h2>
        <p>This document outlines the operational protocols and analytical methodologies governing the AI-Driven Trading Scanner.</p>
    </body>
    </html>
    """
    st.components.v1.html(html_manual, height=800, scrolling=True)

elif current_page == "support":
    st.subheader("☕ Support the Creator - David_Seda")
    try:
        st.image("qr_solana.png", width=180)
    except Exception:
        st.info("📌 QR code image not found. Please add 'qr_solana.png' to the project folder.")
    st.markdown("<p style='font-size: 1.1em; color: gray;'>If this app entertains you or makes you money, buy me a coffee! ☕</p>", unsafe_allow_html=True)
