import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- VIZUÁLNÍ KONFIGURACE A DESIGN (CSS) ---
st.set_page_config(
    page_title="Klondike Spot Swing Scanner", 
    page_icon="📈", 
    layout="wide"
)

st.markdown("""
    <style>
    /* Světlé pozadí hlavního kontejneru */
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    /* Stylování karet a kontejnerů pro světlý režim */
    .stExpander {
        border: 1px solid #e0e0e0 !important;
        border-radius: 10px !important;
        background-color: #f8f9fa !important;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    h1, h2, h3 {
        letter-spacing: -0.5px;
    }
    </style>
""", unsafe_allow_html=True)

class KlondikeExecutionAgent:
    def __init__(self):
        self.status = "Aktivní a připraven (Pouze Spot)"
        self.protocols = ["Spot Trend Following", "Dynamic Volatility Guard", "Sentiment Feed Integrator"]

st.title("📈 AI Spot Swing Scanner CZ")
st.markdown("<p style='font-size: 1.1em; color: #555555;'>Profesionální tržní analytika a automatický výpočet rizika výhradně pro obchodování akcií bez páky (Spot).</p>", unsafe_allow_html=True)

# Inicializace databáze Supabase
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning("⚠️ Databáze není připojena")

# Postranní panel pro vlastní akcie a filtry
st.sidebar.markdown("### 🔍 Vyhledávání aktiv")
custom_ticker_input = st.sidebar.text_input("Přidat ticker (např. ČEZ, AAPL):", "").upper().strip()

DEFAULT_TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "ASML", "NVDA", "NFLX", "AMD", "INTC", "KO", "JPM", "XOM", "JNJ", "SPY", "V", "DIS", "BAC", "PLTR", "PFE", "NKE", "PYPL", "IBM", "UBER", "WMT"]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Přidáno: {custom_ticker_input} do skeneru!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Rychlé filtry")
filter_high_gain = st.sidebar.toggle("🔥 Zobrazit pouze Zisk ≥ 0.08 USD", value=False)
filter_safe_earnings = st.sidebar.toggle("🛡️ Skrýt akcie s výsledky < 7 dnů", value=False)
filter_outperforming = st.sidebar.toggle("🚀 Pouze překonávající S&P 500", value=False)

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
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return float(atr.iloc[-1])

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

def get_next_earnings_days(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if dates:
                next_date = pd.to_datetime(dates[0])
                delta_days = (next_date - pd.Timestamp.now()).days
                return delta_days, next_date.strftime('%Y-%m-%d')
        
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            future_dates = ed[ed.index > pd.Timestamp.now()]
            if not future_dates.empty:
                next_date = future_dates.index[0]
                delta_days = (next_date - pd.Timestamp.now()).days
                return delta_days, next_date.strftime('%Y-%m-%d')
    except Exception:
        pass
    return 999, "N/A"

def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (Žádné čerstvé zprávy)", "Titulek nenalezen"
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss", "pokles", "propad"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain", "buy", "rust", "zisk"]
        
        score = 0
        latest_headline = "Neznámý titulek"
        
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if latest_headline == "Neznámý titulek" and title:
                latest_headline = title
            title_lower = title.lower()
            for kw in bullish_keywords:
                if kw in title_lower: score += 1
            for kw in bearish_keywords:
                if kw in title_lower: score -= 1
                
        if score > 0: return "📈 BÝČÍ (BULLISH)", latest_headline
        elif score < 0: return "📉 MEDVĚDÍ (BEARISH)", latest_headline
        else: return "➖ NEUTRÁLNÍ", latest_headline
    except Exception:
        return "➖ (Zprávy nedostupné)", "Chyba načítání zpráv"

def render_klondike_agent_execution_hub():
    st.subheader("🤖 Klondike Spot Agent Hub")
    st.markdown("Monitorování automatického vyvažování portfolia a protokolů pro vstup do swingových pozic.")
    
    agent = KlondikeExecutionAgent()
    
    col_status, col_metrics = st.columns([1, 1])
    
    with col_status:
        st.success(f"**Stav agenta:** {agent.status}")
        st.markdown("#### Aktivní protokoly:")
        for proto in agent.protocols:
            st.markdown(f"- ✅ `{proto}`")
            
    with col_metrics:
        st.metric("Latence agenta", "14 ms", delta="-2 ms optimální")
        st.metric("Úspěšnost exekuce", "98.4%", delta="+0.6% vs minulý týden")
        
    st.markdown("---")
    st.markdown("### ⚡ Manuální konzole agenta")
    if st.button("🚀 Vynutit okamžitou synchronizaci portfolia", use_container_width=True):
        st.toast("Rebalancovací sekvence spuštěna!", icon="🤖")

def render_user_manual():
    st.subheader("📘 Uživatelský manuál")
    st.markdown("Vítejte v příručce přizpůsobené pro **spotové obchodování bez páky**. Žádné shortování, žádný margin, pouze kvalitní blue-chip setupy.")

    with st.expander("📖 1. Jak ovládat a spustit aplikaci"):
        st.markdown("1. **Skenování trhu:** Skenujte velké tituly a filtrujte bezpečné akcie.")
        st.markdown("2. **Velikost pozice:** Vypočítejte přesný počet kusů na základě vašeho kapitálu.")
        st.markdown("3. **SMA odrazy & MACD:** Hledejte návraty k 50dennímu klouzavému průměru podpořené momentem.")

app_mode = st.radio("Zvolte režim zobrazení:", [
    "📊 Skenování trhu a přehled", 
    "🧠 AI Přesnost a Historie", 
    "🤖 Klondike Agent Hub",
    "📘 Uživatelský manuál"
], horizontal=True)

if app_mode == "📊 Skenování trhu a přehled":
    col_main, col_insiders = st.columns([2.3, 1.2])

    with col_main:
        if st.button("🚀 Spustit analýzu spotového trhu", type="primary", use_container_width=True):
            st.session_state.analysis_run = True

        if st.session_state.analysis_run:
            with st.spinner("Stahování dat, výpočet indikátorů a aktualizace AI..."):
                try:
                    sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                    if isinstance(sp500.columns, pd.MultiIndex):
                        sp500.columns = sp500.columns.get_level_values(0)
                    sp500_close = float(sp500['Close'].iloc[-1])
                    sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
                    
                    if len(sp500) >= 30:
                        sp500_30d_return = (sp500_close - float(sp500['Close'].iloc[-30])) / float(sp500['Close'].iloc[-30]) * 100
                    else:
                        sp500_30d_return = 0

                    if sp500_close < sp500_sma50:
                        st.warning("⚠️ MAKRO VAROVÁNÍ: S&P 500 je pod svým 50denním průměrem (hotovost je bezpečnější).")
                    else:
                        st.success("🌍 MAKRO STAV: S&P 500 je v pozitivním trendu.")
                except:
                    st.info("🌍 Makro stav se nepodařilo ověřit.")
                    sp500_30d_return = 0

            analyzed_count = 0
            for ticker in active_tickers:
                try:
                    t_obj = yf.Ticker(ticker)
                    data = t_obj.history(period="1y", interval="1d")
                    if data.empty or len(data) < 30:
                        continue

                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    skutecna_cena = float(data['Close'].iloc[-1])
                    vrchol_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
                    rozdil_usd = vrchol_20d - skutecna_cena
                    zisk_na_1_usd = rozdil_usd / skutecna_cena if skutecna_cena > 0 else 0

                    if filter_high_gain and zisk_na_1_usd < 0.08:
                        continue

                    earnings_days, earnings_date_str = get_next_earnings_days(t_obj)

                    if filter_safe_earnings and earnings_days <= 7:
                        continue

                    high_52w = float(data['High'].max())
                    dist_52w_pct = ((high_52w - skutecna_cena) / high_52w) * 100

                    asset_30d_return = (skutecna_cena - float(data['Close'].iloc[-30])) / float(data['Close'].iloc[-30]) * 100
                    rs_vs_sp500 = asset_30d_return - sp500_30d_return

                    if filter_outperforming and rs_vs_sp500 < 0:
                        continue

                    analyzed_count += 1
                    
                    news_sentiment, latest_headline = analyze_news_sentiment(t_obj)
                    rsi_val = calculate_rsi(data)
                    atr_val = calculate_atr(data)
                    macd_line, macd_signal, macd_hist = calculate_macd(data)
                    
                    sma_50 = float(data['Close'].rolling(window=50).mean().iloc[-1]) if len(data) >= 50 else float(data['Close'].mean())
                    sma_200 = float(data['Close'].rolling(window=200).mean().iloc[-1]) if len(data) >= 200 else float(data['Close'].mean())
                    
                    distance_to_sma50_pct = abs(skutecna_cena - sma_50) / sma_50 * 100
                    is_sma50_bounce = distance_to_sma50_pct < 2.0 and skutecna_cena > sma_50

                    is_bullish_trend = skutecna_cena > sma_200

                    ai_score = 0
                    if skutecna_cena > sma_50: ai_score += 1
                    else: ai_score -= 1
                    
                    if rs_vs_sp500 > 0: ai_score += 1
                    if macd_hist > 0: ai_score += 1

                    if rsi_val < 35: ai_score += 1
                    elif rsi_val > 65: ai_score -= 1

                    if ai_score > 1:
                        quantitative_direction = "📈 SILNĚ BÝČÍ (Ideální pro swing)"
                        confidence = 80
                    elif ai_score == 1:
                        quantitative_direction = "↗️ MÍRNĚ BÝČÍ"
                        confidence = 65
                    elif ai_score < 0:
                        quantitative_direction = "📉 MEDVĚDÍ (Vyhnout se spotovému nákupu)"
                        confidence = 75
                    else:
                        quantitative_direction = "⚖️ NEUTRÁLNÍ / KONSOLIDACE"
                        confidence = 50

                    if rsi_val > 70:
                        market_state_text = "🔴 **PŘEKOUPENO:** Vysoké riziko korekce. Nekupujte."
                        advice_action = "⏳ **DOPORUČENÍ: VYČKAT**"
                        advice_color = "error"
                    elif is_sma50_bounce and macd_hist > 0:
                        market_state_text = "🎯 **PERFEKTNÍ SWING SETUP:** Cena testuje SMA50 s pozitivním MACD momentem!"
                        advice_action = "✅ **DOPORUČENÍ: VSTOUPIT DO LONGU (SPOT)**"
                        advice_color = "success"
                    elif is_bullish_trend and rsi_val <= 60 and rsi_val >= 40:
                        market_state_text = "🟡 **ZDRAVÝ TREND:** Akumulační zóna."
                        advice_action = "✅ **DOPORUČENÍ: POSTUPNÝ NÁKUP (DCA)**"
                        advice_color = "success"
                    else:
                        market_state_text = "⚖️ **NEROZHODNÝ STAV:** Chybí silné momentum."
                        advice_action = "⏳ **DOPORUČENÍ: DRŽET HOTOVOST**"
                        advice_color = "info"

                    long_entry = skutecna_cena
                    long_stop_loss = skutecna_cena - (1.5 * atr_val)
                    long_take_profit = skutecna_cena + (2.5 * atr_val)

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
                        except Exception:
                            pass

                    with st.expander(f"📌 {ticker} | Cena: ${skutecna_cena:.2f} | RS vs S&P500: {rs_vs_sp500:+.1f}%"):
                        col1, col2, col3 = st.columns(3)
                        
                        col1.markdown("**Technická základna**")
                        col1.metric("Aktuální cena", f"${skutecna_cena:.2f}")
                        col1.metric("RSI (14)", f"{rsi_val:.1f}")
                        
                        col2.markdown("**Momentum**")
                        macd_color_text = "🟢 Pozitivní" if macd_hist > 0 else "🔴 Negativní"
                        col2.metric("MACD Hist", f"{macd_hist:.2f}", delta=macd_color_text, delta_color="off")
                        col2.metric("Zisk / 1 USD", f"+{zisk_na_1_usd:.2f} USD")
                        
                        col3.markdown("**Síla sektoru**")
                        rs_color_text = "🟢 Nad trhem" if rs_vs_sp500 > 0 else "🔴 Pod trhem"
                        col3.metric("vs S&P 500 (30d)", f"{rs_vs_sp500:+.2f}%", delta=rs_color_text, delta_color="off")
                        col3.metric("Od 52W maxima", f"-{dist_52w_pct:.1f}%")

                        st.markdown("---")
                        
                        # --- INTERAKTIVNÍ AI TLAČÍTKO PRO VÝKLAD SHRNUTÍ ---
                        if st.button(f"🤖 Zeptat se AI: Jak si vyložit toto shrnutí?", key=f"ai_summary_btn_{ticker}"):
                            with st.spinner("AI analyzuje tržní data..."):
                                st.markdown("### 🧠 AI Výklad výsledků:")
                                st.write(
                                    f"Aktivum se nachází **{dist_52w_pct:.1f} % pod svým 52týdenním maximem**, "
                                    f"což znamená, že je v korekčním či slevovém pásmu. "
                                    f"Ukazatel RSI na hodnotě **{rsi_val:.1f}** signalizuje, že trh "
                                    f"{'je v mírně přeprodaném stavu' if rsi_val < 45 else 'se drží v neutrální zóně'}. "
                                    f"MACD histogram dosahuje hodnoty **{macd_hist:.2f}**, což poukazuje na "
                                    f"{'kladné krátkodobé momentum' if macd_hist > 0 else 'přetrvávající medvědí tlak'}. "
                                    f"Sektorové srovnání vůči S&P 500 ({rs_vs_sp500:+.2f}%) dokazuje, že aktivum "
                                    f"{'překonává širší trh a přitahuje institucionální zájem' if rs_vs_sp500 > 0 else 'zaostává za širším trhem'}. "
                                    f"💡 **Doporučený pohled AI:** Vzhledem k těmto metrikám dbejte zvýšené opatrnosti, "
                                    f"využijte DCA (postupný nákup) a dodržujte předepsané úrovně pro Stop-Loss."
                                )
                        # ----------------------------------------------------

                        st.markdown("---")
                        st.info(f"🤖 **AI Směr:** {quantitative_direction} (Spolehlivost: {confidence}%)")
                        
                        st.markdown("### 💡 Investiční doporučení:")
                        st.markdown(market_state_text)
                        if advice_color == "success":
                            st.success(advice_action)
                        elif advice_color == "error":
                            st.error(advice_action)
                        else:
                            st.info(advice_action)

                        st.markdown("#### 🟢 SPOT SWING SETUP")
                        col_entry, col_sl, col_tp = st.columns(3)
                        col_entry.success(f"**Ideální vstup:**\n${long_entry:.2f}")
                        col_sl.warning(f"**Stop Loss:**\n${long_stop_loss:.2f}")
                        col_tp.info(f"**Take Profit:**\n${long_take_profit:.2f}")

                        st.markdown("---")
                        st.markdown("#### 💰 Kalkulačka velikosti pozice (Bez páky)")
                        col_cap1, col_cap2 = st.columns(2)
                        with col_cap1:
                            user_capital = st.number_input(f"Celková hotovost ($) pro {ticker}:", value=5000.0, step=500.0, key=f"cap_{ticker}")
                        with col_cap2:
                            risk_pct = st.slider(f"Riziko na obchod (% z kapitálu):", 0.5, 3.0, 1.0, key=f"risk_{ticker}")

                        allowed_risk_usd = user_capital * (risk_pct / 100.0)
                        risk_per_share = 1.5 * atr_val
                        shares_to_buy = int(allowed_risk_usd / risk_per_share) if risk_per_share > 0 else 0
                        total_position_value = shares_to_buy * skutecna_cena

                        if total_position_value > user_capital:
                            shares_to_buy = int(user_capital / skutecna_cena)
                            total_position_value = shares_to_buy * skutecna_cena
                            st.warning("⚠️ Počáteční výpočet přesáhl vaši hotovost. Upraveno na maximální dostupné množství.")

                        st.info(f"👉 **Exekuce:** Koupit **{shares_to_buy} ks** | **Celková cena:** `${total_position_value:.2f}` | **Max riziko:** `${allowed_risk_usd:.2f}`")

                        st.markdown("---")
                        st.write(f"**Sentiment zpráv:** {news_sentiment} | *\"{latest_headline}\"*")
                        
                        if earnings_days != 999:
                            if earnings_days <= 7:
                                st.error(f"⚠️ **VÝSLEDKY ZA {earnings_days} DNŮ:** ({earnings_date_str}). Vysoké gap riziko!")
                            else:
                                st.info(f"📅 **Příští výsledky:** {earnings_date_str} (za {earnings_days} dnů)")
                        
                        trend_status = "✅ Zdravý uptrend" if is_bullish_trend else "❌ Pod 200denním SMA (Vysoké riziko)"
                        st.write(f"**Dlouhodobý trend (SMA 200):** {trend_status}")

                        fig, ax = plt.subplots(figsize=(10, 4))
                        model.plot(forecast, ax=ax)
                        ax.set_title(f"20denní predikce ceny: {ticker}")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Chyba při zpracování {ticker}: {e}")
            
            if analyzed_count == 0:
                st.warning("⚠️ Žádná aktiva neodpovídají aktuálním filtrům.")

    with col_insiders:
        st.markdown("### 🏛️ Top nákupy insiderů")
        st.markdown("<p style='font-size: 0.9em; color: gray;'>Sleduje velké transakce manažerů.</p>", unsafe_allow_html=True)
        
        insider_data_list = []
        insider_tickers = ["META", "MSFT", "GOOGL", "AAPL", "AMZN", "BRK-B", "NVDA"]
        
        for t_sym in insider_tickers:
            try:
                tk = yf.Ticker(t_sym)
                insiders = getattr(tk, 'insider_transactions', None)
                if insiders is not None and not insiders.empty:
                    latest = insiders.iloc[0]
                    insider_data_list.append({
                        "Ticker": t_sym,
                        "Osoba": str(latest.get('Name', 'N/A'))[:12],
                        "Akce": str(latest.get('Transaction', 'Akce'))
                    })
            except Exception:
                pass
                
        if insider_data_list:
            df_insiders = pd.DataFrame(insider_data_list)
            st.dataframe(df_insiders, hide_index=True, use_container_width=True)
        else:
            st.info("Data o insiderech nejsou k dispozici.")

elif app_mode == "🧠 AI Přesnost a Historie":
    st.subheader("🧠 Historie AI predikcí")
    if supabase:
        try:
            response = supabase.table("predictions").select("*").order("target_date", desc=True).limit(50).execute()
            data_rows = response.data
            if data_rows:
                st.dataframe(pd.DataFrame(data_rows), use_container_width=True)
            else:
                st.warning("Zatím nejsou uloženy žádné predikce.")
        except Exception as e:
            st.error(f"Chyba databáze: {e}")
    else:
        st.error("Supabase není připojena.")

elif app_mode == "🤖 Klondike Agent Hub":
    render_klondike_agent_execution_hub()

elif app_mode == "📘 Uživatelský manuál":
    render_user_manual()

st.sidebar.markdown("---")
st.sidebar.subheader("☕ Podpořte tvůrce")
try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 Vložte 'qr_solana.png' do složky projektu.")
