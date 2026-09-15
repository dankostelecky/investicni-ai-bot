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
    page_title="Klondike Spot Swing Skener", 
    page_icon="📈", 
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        color: #000000;
    }
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
        self.status = "Aktivní a připraveno (pouze Spot)"
        self.protocols = ["Sledování spotového trendu", "Dynamická ochrana volatility", "Potvrzení objemu a vzorců"]

st.title("📈 AI Spot Swing Skener & Analytik Vzorců")
st.markdown("<p style='font-size: 1.1em; color: #555555;'>Profesionální tržní analytika s AI vhledy, detekcí objemových špiček, průrazů a relativní síly vůči S&P 500.</p>", unsafe_allow_html=True)

# Inicializace Supabase databáze
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning("⚠️ Databáze nepřipojena")

# Postranní panel pro vlastní tickery a filtry
st.sidebar.markdown("### 🔍 Vyhledávání aktiv")
custom_ticker_input = st.sidebar.text_input("Přidat ticker (např. AAPL, MSFT):", "").upper().strip()

# Čistý seznam bez duplicit (včetně globálních firem v USD / ADR)
DEFAULT_TICKERS = [
    # --- Americké Mega-Cap a špičky ---
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "WMT", "LLY",
    "MU", "JPM", "ORCL", "XOM", "V", "MA", "AMD", "NFLX", "JNJ", "COST", "HD", "CRM", 
    "UNH", "PG", "ABBV", "BAC", "IBM", "DIS", "INTC", "KO", "PLTR", "UBER", "PYPL", "PFE", "NKE",
    
    # --- Světoví giganti obchodovaní v USD (ADR) ---
    "ASML",  # ASML Holding (Nizozemsko - Polovodiče)
    "TSM",   # Taiwan Semiconductor (Taiwan - Čipy)
    "NVO",   # Novo Nordisk (Dánsko - Farmacie)
    "BABA",  # Alibaba Group (Čína - E-commerce)
    "TM",    # Toyota Motor (Japonsko - Automobily)
    "AZN",   # AstraZeneca (Spojené království - Biopharmaceuticals)
    "SHEL",  # Shell plc (Spojené království / Nizozemsko - Energetika)
    "NSRGY", # Nestlé (Švýcarsko - Potraviny)
    "SAP",   # SAP SE (Německo - Software)
    "TTE",   # TotalEnergies (Francie - Energetika)
    "HSBC",  # HSBC Holdings (Spojené království - Bankovnictví)
    "SONY",  # Sony Group (Japonsko - Technologie a zábava)
    "MELI",  # MercadoLibre (Latinská Amerika - E-commerce / Fintech)
    "RIO",   # Rio Tinto (Spojené království / Austrálie - Těžba)
    "BP",    # BP p.l.c. (Spojené království - Energetika)
    
    # --- Benchmark ---
    "SPY"    # S&P 500 ETF
]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Přidáno: {custom_ticker_input} do skeneru!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filtry a strategie")

filter_high_gain = st.sidebar.toggle("💵 Zisk / 1 $ >= 0.06 $", value=False, help="Zobrazí pouze aktiva s růstovým potenciálem k 20dennímu vrcholu 0.06 $ nebo více na 1 $ investice.")
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

def analyze_volume(data, window=20):
    current_vol = float(data['Volume'].iloc[-1])
    avg_vol = float(data['Volume'].rolling(window=window).mean().iloc[-1])
    is_spike = current_vol > (1.5 * avg_vol)
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    return current_vol, avg_vol, is_spike, vol_ratio

def detect_patterns(data):
    close = data['Close']
    high_20 = close.rolling(window=20).max()
    low_20 = close.rolling(window=20).min()
    
    is_breakout = bool(close.iloc[-1] >= high_20.iloc[-2])
    is_breakdown = bool(close.iloc[-1] <= low_20.iloc[-2])
    
    sma = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    bandwidth = (upper - lower) / sma
    bw_rolling_mean = bandwidth.rolling(window=50).mean().iloc[-1]
    is_squeeze = bool(bandwidth.iloc[-1] < (bw_rolling_mean * 0.8)) if not np.isnan(bw_rolling_mean) else False
    
    return is_breakout, is_breakdown, is_squeeze

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
            return "➖ (Žádné čerstvé zprávy)", "Nadpis nenalezen", []
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss", "miss"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain", "buy"]
        
        score = 0
        latest_headline = "Neznámý nadpis"
        headlines = []
        
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if title:
                headlines.append(title)
                if latest_headline == "Neznámý nadpis":
                    latest_headline = title
                title_lower = title.lower()
                for kw in bullish_keywords:
                    if kw in title_lower: score += 1
                for kw in bearish_keywords:
                    if kw in title_lower: score -= 1
                
        if score > 0: sentiment_label = "📈 BÝČÍ"
        elif score < 0: sentiment_label = "📉 MEDVĚDÍ"
        else: sentiment_label = "➖ NEUTRÁLNÍ"
        
        return sentiment_label, latest_headline, headlines
    except Exception:
        return "➖ (Zprávy nedostupné)", "Chyba při načítání zpráv", []

app_mode = st.radio("Vyberte zobrazení:", [
    "📊 Tržní skener & Vzorce", 
    "🤖 Klondike Agent Hub",
    "📘 Uživatelská příručka"
], horizontal=True)

if app_mode == "📊 Tržní skener & Vzorce":
    if st.button("🚀 Spustit sken a analýzu vzorců", type="primary", use_container_width=True):
        st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        with st.spinner("Zpracovává se benchmark S&P 500, objemové profily a vzorce..."):
            try:
                sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = sp500.columns.get_level_values(0)
                sp500_close = float(sp500['Close'].iloc[-1])
                
                if len(sp500) >= 30:
                    sp500_30d_return = (sp500_close - float(sp500['Close'].iloc[-30])) / float(sp500['Close'].iloc[-30]) * 100
                else:
                    sp500_30d_return = 0
            except:
                sp500_30d_return = 0

        valid_results = []
        analyzed_count = 0
        
        for ticker in active_tickers:
            try:
                t_obj = yf.Ticker(ticker)
                data = t_obj.history(period="1y", interval="1d")
                if data.empty or len(data) < 30:
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                actual_price = float(data['Close'].iloc[-1])
                
                peak_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
                diff_usd = peak_20d - actual_price
                gain_per_1_usd = diff_usd / actual_price if actual_price > 0 else 0

                asset_30d_return = (actual_price - float(data['Close'].iloc[-30])) / float(data['Close'].iloc[-30]) * 100
                rs_vs_sp500 = asset_30d_return - sp500_30d_return

                if filter_high_gain and gain_per_1_usd < 0.06:
                    continue

                is_breakout, is_breakdown, is_squeeze = detect_patterns(data)
                current_vol, avg_vol, is_vol_spike, vol_ratio = analyze_volume(data)
                
                if filter_breakout and not is_breakout:
                    continue
                if filter_squeeze and not is_squeeze:
                    continue
                if filter_volume and not is_vol_spike:
                    continue
                if filter_outperforming and rs_vs_sp500 < 0:
                    continue

                earnings_days, earnings_date_str = get_next_earnings_days(t_obj)
                if filter_safe_earnings and earnings_days <= 7:
                    continue

                analyzed_count += 1
                
                news_sentiment, latest_headline, headlines_list = analyze_news_sentiment(t_obj)
                rsi_val = calculate_rsi(data)
                atr_val = calculate_atr(data)
                macd_line, macd_signal, macd_hist = calculate_macd(data)
                
                pattern_label = "⚖️ Standardní pohyb"
                if is_breakout and is_vol_spike:
                    pattern_label = "🚀 Silný objemový průraz (Silně býčí)"
                elif is_breakout:
                    pattern_label = "↗️ Cenový průraz"
                elif is_squeeze:
                    pattern_label = "📦 Konsolidace (BB Squeeze)"
                elif is_breakdown:
                    pattern_label = "📉 Varování před propadem / Výprodej"

                long_entry = actual_price
                long_stop_loss = actual_price - (1.5 * atr_val)
                long_take_profit = actual_price + (2.5 * atr_val)

                # Výpočet síly a procenta trendu
                rsi_score_component = (rsi_val - 50) / 50 * 30
                rs_score_component = np.clip(rs_vs_sp500, -20, 20) / 20 * 40
                macd_score_component = 30 if macd_hist > 0 else -30
                
                raw_trend_score = rsi_score_component + rs_score_component + macd_score_component
                trend_pct = float(np.clip(raw_trend_score, -100, 100))

                if trend_pct >= 40:
                    trend_text = f"Silný býčí trend (+{trend_pct:.1f}%)"
                elif trend_pct > 0:
                    trend_text = f"Mírný býčí trend (+{trend_pct:.1f}%)"
                elif trend_pct <= -40:
                    trend_text = f"Silný medvědí trend / Výprodej ({trend_pct:.1f}%)"
                else:
                    trend_text = f"Mírný medvědí trend ({trend_pct:.1f}%)"

                # Logika doporučení akce
                if is_breakout and is_vol_spike and rsi_val < 70 and rs_vs_sp500 > 0:
                    action_rec = "🚀 SILNÝ NÁKUP (Okamžité potvrzení průrazu)"
                elif is_squeeze and rs_vs_sp500 >= 0:
                    action_rec = "📦 AKUMULACE / DCA (Konsolidace před expanzí)"
                elif rs_vs_sp500 > 0 and rsi_val <= 65:
                    action_rec = "🛒 NÁKUP NA PROPADECH (Překonává relativní sílu)"
                elif rsi_val > 75 or is_breakdown:
                    action_rec = "⚠️ REDUKOVAT / PRODAT (Překoupeno nebo riziko propadu)"
                else:
                    action_rec = "⏳ ČEKAT / SLEDOVAT (Neutrální nastavení, vyčkejte)"

                score = rs_vs_sp500 + (vol_ratio * 10) if is_vol_spike else rs_vs_sp500

                valid_results.append({
                    "ticker": ticker,
                    "t_obj": t_obj,
                    "data": data,
                    "actual_price": actual_price,
                    "gain_per_1_usd": gain_per_1_usd,
                    "rs_vs_sp500": rs_vs_sp500,
                    "rsi_val": rsi_val,
                    "vol_ratio": vol_ratio,
                    "is_vol_spike": is_vol_spike,
                    "pattern_label": pattern_label,
                    "long_entry": long_entry,
                    "long_stop_loss": long_stop_loss,
                    "long_take_profit": long_take_profit,
                    "atr_val": atr_val,
                    "news_sentiment": news_sentiment,
                    "latest_headline": latest_headline,
                    "headlines_list": headlines_list,
                    "earnings_days": earnings_days,
                    "earnings_date_str": earnings_date_str,
                    "trend_pct": trend_pct,
                    "trend_text": trend_text,
                    "action_rec": action_rec,
                    "score": score
                })

            except Exception as e:
                st.error(f"Chyba při zpracování {ticker}: {e}")
        
        if analyzed_count > 0 and valid_results:
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)

            st.markdown("### 📊 Výsledky filtrovaného skenu")

            for res in valid_results:
                ticker = res["ticker"]
                actual_price = res["actual_price"]
                gain_per_1_usd = res["gain_per_1_usd"]
                rs_vs_sp500 = res["rs_vs_sp500"]
                rsi_val = res["rsi_val"]
                vol_ratio = res["vol_ratio"]
                is_vol_spike = res["is_vol_spike"]
                pattern_label = res["pattern_label"]
                long_entry = res["long_entry"]
                long_stop_loss = res["long_stop_loss"]
                long_take_profit = res["long_take_profit"]
                atr_val = res["atr_val"]
                news_sentiment = res["news_sentiment"]
                latest_headline = res["latest_headline"]
                headlines_list = res["headlines_list"]
                earnings_days = res["earnings_days"]
                earnings_date_str = res["earnings_date_str"]
                trend_pct = res["trend_pct"]
                trend_text = res["trend_text"]
                action_rec = res["action_rec"]
                data = res["data"]

                df = data.reset_index()[['Date', 'Close']]
                df.columns = ['ds', 'y']
                df['ds'] = df['ds'].dt.tz_localize(None)

                model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                model.fit(df)
                future = model.make_future_dataframe(periods=PRED_DAYS)
                forecast = model.predict(future)

                with st.expander(f"📌 {ticker} | Cena: ${actual_price:.2f} | Akce: {action_rec.split(' ')[0]} {action_rec.split(' ')[1]} | Trend: {trend_text}"):
                    
                    st.markdown(f"### 🎯 Doporučení akce: **{action_rec}**")
                    st.markdown(f"📈 **Analýza trendu:** {trend_text}")
                    st.progress(int((trend_pct + 100) / 2))
                    
                    st.markdown("---")

                    col1, col2, col3 = st.columns(3)
                    
                    col1.markdown("**Cena & Momentum**")
                    col1.metric("Aktuální cena", f"${actual_price:.2f}")
                    col1.metric("RSI (14)", f"{rsi_val:.1f}")
                    
                    col2.markdown("**Objem & Potenciál**")
                    vol_status_text = "🔥 Objemová špička" if is_vol_spike else "⚖️ Normální objem"
                    col2.metric("Poměr objemu", f"{vol_ratio:.2f}x průměru", delta=vol_status_text, delta_color="off")
                    col2.metric("Zisk / 1 $ investice", f"+${gain_per_1_usd:.2f}")
                    
                    col3.markdown("**Benchmark & Vzorce**")
                    rs_color = "🟢 Nadprůměrný" if rs_vs_sp500 > 0 else "🔴 Podprůměrný"
                    col3.metric("vs S&P 500 (30d)", f"{rs_vs_sp500:+.2f}%", delta=rs_color, delta_color="off")
                    col3.write(f"**Vzorec:** {pattern_label}")

                    st.markdown("---")
                    
                    if st.button(f"🤖 AI Analytik: Posoudit techniku i zprávy pro {ticker}", key=f"ai_summary_btn_{ticker}"):
                        with st.spinner("AI propojuje technické ukazatele s mediálním sentimentem..."):
                            st.markdown("### 🧠 AI hodnocení trhu a souvislostí:")
                            
                            news_explanation = ""
                            if headlines_list:
                                news_explanation = f"Aktuální mediální ohlasy (např. *\"{latest_headline}\"*) naznačují, že trh reaguje na zprávy s **{news_sentiment.lower()}** podtónem. "
                            else:
                                news_explanation = "Žádné výrazné čerstvé titulky v hlavních médiích nebyly detekovány, pohyb tak vychází primárně z technických nákupů/prodejů institucí. "

                            vol_explanation = f"Objemová aktivita dosahuje **{vol_ratio:.2f}násobku** běžného průměru ({'což potvrzuje silný zájem nebo paniku' if is_vol_spike else 'při běžné likviditě'})."

                            st.write(
                                f"Aktivum **{ticker}** vykazuje relativní výkonnost **{rs_vs_sp500:+.2f}%** vůči S&P 500 za posledních 30 dní. "
                                f"Technický stav s RSI **{rsi_val:.1f}** a vzorcem **{pattern_label}** signalizuje: {trend_text}. \n\n"
                                f"📰 **Proč se akcie takto chová (AI vysvětlení zpráv):**\n"
                                f"{news_explanation} {vol_explanation}\n\n"
                                f"💡 **Doporučený další krok:** **{action_rec}**."
                            )

                    st.markdown("---")
                    st.markdown("#### 🟢 SPOT SWING NASTAVENÍ")
                    col_entry, col_sl, col_tp = st.columns(3)
                    col_entry.success(f"**Ideální vstup:**\n${long_entry:.2f}")
                    col_sl.warning(f"**Stop Loss:**\n${long_stop_loss:.2f}")
                    col_tp.info(f"**Take Profit:**\n${long_take_profit:.2f}")

                    st.markdown("---")
                    st.markdown("#### 💰 Kalkulačka velikosti pozice (bez páky)")
                    col_cap1, col_cap2 = st.columns(2)
                    with col_cap1:
                        user_capital = st.number_input(f"Celkový kapitál ($) pro {ticker}:", value=5000.0, step=500.0, key=f"cap_{ticker}")
                    with col_cap2:
                        risk_pct = st.slider(f"Riziko na obchod (% kapitálu):", 0.5, 3.0, 1.0, key=f"risk_{ticker}")

                    allowed_risk_usd = user_capital * (risk_pct / 100.0)
                    risk_per_share = 1.5 * atr_val
                    shares_to_buy = int(allowed_risk_usd / risk_per_share) if risk_per_share > 0 else 0
                    total_position_value = shares_to_buy * actual_price

                    if total_position_value > user_capital:
                        shares_to_buy = int(user_capital / actual_price)
                        total_position_value = shares_to_buy * actual_price
                        st.warning("⚠️ Úvodní výpočet překročil dostupnou hotovost. Upraveno na maximální možný počet kusů.")

                    st.info(f"👉 **Provedení:** Koupit **{shares_to_buy} ks** | **Celková hodnota:** `${total_position_value:.2f}` | **Max. riziko:** `${allowed_risk_usd:.2f}`")

                    st.markdown("---")
                    st.markdown("#### 📰 Přehled nejnovějších zpráv")
                    st.write(f"**Sentiment:** {news_sentiment}")
                    if headlines_list:
                        for h in headlines_list[:3]:
                            st.markdown(f"- *{h}*")
                    else:
                        st.write("Žádné zprávy k zobrazení.")
                    
                    if earnings_days != 999 and earnings_days <= 7:
                        st.error(f"⚠️ **VÝSLEDKY ZA {earnings_days} DNŮ:** ({earnings_date_str}). Vysoké riziko mezer v grafu (gap risk)!")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    model.plot(forecast, ax=ax)
                    ax.set_title(f"20denní cenová předpověď: {ticker}")
                    st.pyplot(fig)

        if analyzed_count == 0 or not valid_results:
            st.warning("⚠️ Žádná aktiva neodpovídají vašim aktuálním vzorcům a kritériím filtrů.")

elif app_mode == "🤖 Klondike Agent Hub":
    st.subheader("🤖 Klondike Spot Agent Hub")
    st.markdown("Sledování automatizovaných detektorů objemových anomálií a protokolů pro průrazy konsolidací.")
    agent = KlondikeExecutionAgent()
    st.success(f"**Stav agenta:** {agent.status}")
    for proto in agent.protocols:
        st.markdown(f"- ✅ `{proto}`")

elif app_mode == "📘 Uživatelská příručka":
    st.subheader("📘 Uživatelská příručka & Průvodce strategiemi")
    st.markdown("Vítejte v uživatelské příručce aplikace **Klondike Spot Swing Skener**. Tento nástroj slouží k pokročilé technické a fundamentální analýze akciových trhů se zaměřením na swingové obchodování na spotovém trhu (bez finanční páky).")
    
    st.markdown("---")
    st.markdown("### 📊 Vysvětlení klíčových ukazatelů a metrik")
    
    st.markdown("""
    1. **Zisk / 1 $ investice:** 
       * Ukazuje matematický prostor (v dolarech) směrem k nedávným 20denním vrcholům na každý 1 dolar investovaného kapitálu. Pomáhá okamžitě identifikovat tituly s největším prostorem pro růst (upside potential).
       
    2. **Překonání S&P 500 (Relativní síla - RS):** 
       * Porovnává 30denní výkonnost dané akcie s hlavním tržním indexem S&P 500 (`^GSPC`). Kladná hodnota znamená, že akcie trh poráží (institutionální zájem), záporná hodnota značí zaostávání.
       
    3. **RSI (Relative Strength Index, 14):** 
       * Měřič momentu a rychlosti cenových změn v rozmezí 0–100.
       * *Hodnoty nad 75* značí silné překoupení (riziko korekce).
       * *Hodnoty pod 30* značí přeprodanost. Pro swingové nákupy jsou ideální zdravé hodnoty mezi 40–65 v rostoucím trendu.
       
    4. **Objemový poměr (Volume Ratio) & Špičky:** 
       * Porovnává aktuální denní objem obchodů s 20denním průměrem. Pokud je poměr vyšší než `1.5x`, aplikace hlásí **objemovou špičku** (`🔥`), což potvrzuje vážný zájem velkých hráčů (institucí) o pohyb.
       
    5. **Bollinger Bands Squeeze (Konsolidace):** 
       * Indikátor stlačení volatility. Když se Bollingerova pásma k sobě výrazně přiblíží (squeeze), znamená to, že trh odpočívá a připravuje se na prudký výbušný pohyb (průraz jedním směrem).
       
    6. **ATR (Average True Range, 14):** 
       * Měřič aktuální tržní volatility. V aplikaci se používá pro dynamický výpočet **Stop Lossu** (1.5 násobek ATR pod vstupem) a **Take Profitu** (2.5 násobek ATR nad vstupem), což zajišťuje správný RRR (poměr risk/zisk).
    """)

    st.markdown("---")
    st.markdown("### 🤖 Funkce AI Analytika & Zpráv")
    st.markdown("""
    * **Mediální sentiment:** Aplikace automaticky stahuje nejnovější titulky zpráv pro daný ticker, vyhledává klíčová býčí či medvědí slova a určuje celkový tón zpráv.
    * **Riziko výsledků (Earnings Risk):** Pokud má společnost oznámit hospodářské výsledky za méně než 7 dnů, aplikace vás na to upozorní červeným varováním kvůli vysokému riziku mezer v grafu (gap risk).
    * **Prophet Predikce:** V dolní části detailu každého aktivního titulu naleznete 20denní cenovou předpověď vygenerovanou algoritmem časových řad od Meta (Prophet).
    """)

    st.markdown("---")
    st.markdown("### 💡 Doporučené postupy při obchodování")
    st.markdown("""
    * Využívejte postranní filtry k zacílení na konkrétní setupy (např. pouze průrazy s potvrzeným objemem).
    * Vždy dodržujte doporučený **Stop Loss** zobrazený v detailu akcie.
    * Přizpůsobte velikost pozice svému celkovému kapitálu pomocí integrované kalkulačky rizika.
    """)
