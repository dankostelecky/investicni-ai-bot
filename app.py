import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- PŮVODNÍ VIZUÁLNÍ KONFIGURACE (SVĚTLÝ DESIGN) ---
st.set_page_config(
    page_title="Klondike Spot Swing Skener", 
    page_icon="📈", 
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
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
        color: #111111;
    }
    [data-testid="stSidebar"] {
        background-color: #f1f3f5;
        border-right: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

class KlondikeExecutionAgent:
    def __init__(self):
        self.status = "Aktivní a připraveno (včetně zápisu predikcí do DB)"
        self.protocols = ["Sledování spotového trendu", "Analýza pre-market impulzů", "Zápis a vyhodnocení AI predikcí"]

st.title("📈 AI Spot Swing Skener & Predikční Modul")
st.markdown("<p style='font-size: 1.1em; color: #555555;'>Profesionální tržní analytika s AI vhledy, sledováním pre-marketu a učením se z historických predikcí.</p>", unsafe_allow_html=True)

# Inicializace Supabase databáze
supabase = None
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.sidebar.warning("⚠️ Databáze nepřipojena (zkontrolujte st.secrets)")

# Postranní panel pro vlastní tickery a filtry
st.sidebar.markdown("### 🔍 Vyhledávání aktiv")
custom_ticker_input = st.sidebar.text_input("Přidat ticker (např. AAPL, MSFT):", "").upper().strip()

DEFAULT_TICKERS = [
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", 
    "BRK-B", "WMT", "LLY", "MU", "JPM", "ORCL", "XOM", "V", "MA", 
    "AMD", "NFLX", "JNJ", "COST", "HD", "CRM", "UNH", "PG", "ABBV", 
    "BAC", "IBM", "DIS", "INTC", "KO", "PLTR", "UBER", "PYPL", "PFE", 
    "NKE", "BABA", "SPY"
]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Přidáno: {custom_ticker_input} do skeneru!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filtry a strategie")

filter_high_gain = st.sidebar.toggle("💵 Zisk / 1 $ >= 0.06$", value=False)
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

def get_premarket_data(ticker_obj):
    try:
        info = ticker_obj.info
        pre_price = info.get('preMarketPrice', None)
        prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', None))
        if pre_price and prev_close:
            change_pct = ((pre_price - prev_close) / prev_close) * 100
            return float(pre_price), float(change_pct)
    except Exception:
        pass
    return None, None

def evaluate_premarket_anomaly(pre_change):
    if pre_change is None:
        return "Pre-market data nejsou k dispozici (mimo obchodní hodiny nebo víkend)."
    
    if pre_change >= 3.0:
        return f"🚨 **VÝRAZNÝ BÝČÍ PRE-MARKET SKOK (+{pre_change:.2f}%):** Akcie vykazuje silný ranní nákupní tlak!"
    elif pre_change <= -3.0:
        return f"⚠️ **VÝRAZNÝ MEDVĚDÍ PRE-MARKET PROPAD ({pre_change:.2f}%):** Na pre-marketu probíhá silný výprodej!"
    elif pre_change > 0:
        return f"🟢 **Mírný ranní růst (+{pre_change:.2f}%):** Klidný pre-market."
    elif pre_change < 0:
        return f"🔴 **Mírný ranní pokles ({pre_change:.2f}%):** Lehký prodejní tlak."
    else:
        return "⚖️ Pre-market je bez pohybu (0 %). Žádná ranní anomálie."

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
    "🧠 Historie predikcí & Učení AI",
    "🤖 Klondike Agent Hub",
    "📘 Uživatelská příručka"
], horizontal=True)

if app_mode == "📊 Tržní skener & Vzorce":
    if st.button("🚀 Spustit sken, analýzu vzorců a zapsat predikce do DB", type="primary", use_container_width=True):
        st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        with st.spinner("Zpracovává se benchmark S&P 500, pre-market anomálie, modely a ukládání do DB..."):
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
                pre_price, pre_change = get_premarket_data(t_obj)
                
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

                risk_amount = long_entry - long_stop_loss
                reward_amount = long_take_profit - long_entry
                risk_reward_ratio = (reward_amount / risk_amount) if risk_amount > 0 else 0

                df = data.reset_index()[['Date', 'Close']]
                df.columns = ['ds', 'y']
                df['ds'] = df['ds'].dt.tz_localize(None)

                model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                model.fit(df)
                future = model.make_future_dataframe(periods=PRED_DAYS)
                forecast = model.predict(future)
                
                predicted_price_20d = float(forecast['yhat'].iloc[-1])
                target_date_val = (datetime.now() + timedelta(days=PRED_DAYS)).strftime('%Y-%m-%d')

                # ZÁPIS DO SUPABASE
                if supabase is not None:
                    try:
                        today_str = datetime.now().strftime('%Y-%m-%d')
                        existing = supabase.table("predictions").select("id").eq("ticker", ticker).gte("created_at", today_str).execute()
                        if not existing.data:
                            supabase.table("predictions").insert({
                                "ticker": ticker,
                                "entry_price": float(actual_price),
                                "predicted_price": float(predicted_price_20d),
                                "target_date": target_date_val,
                                "status": "PENDING"
                            }).execute()
                    except Exception as db_err:
                        pass

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

                if is_breakout and is_vol_spike and rsi_val < 70 and rs_vs_sp500 > 0:
                    action_rec = "🚀 SILNÝ NÁKUP (Okamžité potvrzení průrazu)"
                elif is_squeeze and rs_vs_sp500 >= 0:
                    action_rec = "📦 AKUMULACE / DCA (Konsolidace před expanzí)"
                elif rs_vs_sp500 > 0 and rsi_val <= 65:
                    action_rec = "🛒 NÁKUP NA PROPADECH (Překonává relativní sílu)"
                elif rsi_val > 75 or is_breakdown:
                    action_rec = "⚠️ REDUKOVAT / PRODAT (Překoupeno nebo riziko propadu)"
                else:
                    action_rec = "⏳ ČEKAT / SLEDOVAT (Neutrální nastavení)"

                score = rs_vs_sp500 + (vol_ratio * 10) if is_vol_spike else rs_vs_sp500

                valid_results.append({
                    "ticker": ticker,
                    "t_obj": t_obj,
                    "data": data,
                    "actual_price": actual_price,
                    "pre_price": pre_price,
                    "pre_change": pre_change,
                    "gain_per_1_usd": gain_per_1_usd,
                    "rs_vs_sp500": rs_vs_sp500,
                    "rsi_val": rsi_val,
                    "vol_ratio": vol_ratio,
                    "is_vol_spike": is_vol_spike,
                    "pattern_label": pattern_label,
                    "long_entry": long_entry,
                    "long_stop_loss": long_stop_loss,
                    "long_take_profit": long_take_profit,
                    "risk_reward_ratio": risk_reward_ratio,
                    "atr_val": atr_val,
                    "news_sentiment": news_sentiment,
                    "latest_headline": latest_headline,
                    "headlines_list": headlines_list,
                    "earnings_days": earnings_days,
                    "earnings_date_str": earnings_date_str,
                    "trend_pct": trend_pct,
                    "trend_text": trend_text,
                    "action_rec": action_rec,
                    "score": score,
                    "forecast": forecast,
                    "model": model
                })

            except Exception as e:
                pass
        
        if analyzed_count > 0 and valid_results:
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)

            st.markdown("### 📊 Výsledky filtrovaného skenu")

            for res in valid_results:
                ticker = res["ticker"]
                actual_price = res["actual_price"]
                pre_price = res["pre_price"]
                pre_change = res["pre_change"]
                gain_per_1_usd = res["gain_per_1_usd"]
                rs_vs_sp500 = res["rs_vs_sp500"]
                rsi_val = res["rsi_val"]
                vol_ratio = res["vol_ratio"]
                is_vol_spike = res["is_vol_spike"]
                pattern_label = res["pattern_label"]
                long_entry = res["long_entry"]
                long_stop_loss = res["long_stop_loss"]
                long_take_profit = res["long_take_profit"]
                risk_reward_ratio = res["risk_reward_ratio"]
                earnings_days = res["earnings_days"]
                earnings_date_str = res["earnings_date_str"]
                trend_pct = res["trend_pct"]
                trend_text = res["trend_text"]
                action_rec = res["action_rec"]
                forecast = res["forecast"]
                model = res["model"]
                atr_val = res["atr_val"]

                pre_str = f" | Pre-market: ${pre_price:.2f} ({pre_change:+.2f}%)" if pre_price is not None else ""

                with st.expander(f"📌 {ticker} | Cena: ${actual_price:.2f}{pre_str} | Akce: {action_rec.split(' ')[0]} {action_rec.split(' ')[1]}"):
                    
                    st.markdown(f"### 🎯 Doporučení akce: **{action_rec}**")
                    st.markdown(f"📈 **Analýza trendu:** {trend_text}")
                    st.progress(int((trend_pct + 100) / 2))
                    
                    st.markdown("---")
                    st.markdown("#### 🌅 Ranní Pre-market přehled")
                    
                    if pre_price is not None and pre_change is not None:
                        norm_progress = int(np.clip((pre_change + 5) / 10 * 100, 0, 100))
                        col_pg1, col_pg2 = st.columns([3, 1])
                        with col_pg1:
                            st.write(f"Pre-market kurz: **${pre_price:.2f}** (Změna: **{pre_change:+.2f}%**)")
                            st.progress(norm_progress)
                        with col_pg2:
                            if pre_change > 0: st.markdown("🟢 **Ranní růst**")
                            elif pre_change < 0: st.markdown("🔴 **Ranní pokles**")
                            else: st.markdown("⚖️ **Bez pohybu**")
                        
                        anomaly_text = evaluate_premarket_anomaly(pre_change)
                        if abs(pre_change) >= 3.0: st.warning(anomaly_text)
                        else: st.info(anomaly_text)
                    else:
                        st.caption("ℹ️ Pre-market data nejsou k dispozici.")

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.markdown("**Cena & Momentum**")
                    col1.metric("Závěrečná cena", f"${actual_price:.2f}")
                    if pre_price is not None:
                        col1.metric("Pre-market cena", f"${pre_price:.2f}", delta=f"{pre_change:+.2f}%")
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
                    st.markdown("#### 🟢 SPOT SWING NASTAVENÍ & RISK/REWARD")
                    col_entry, col_sl, col_tp, col_rr = st.columns(4)
                    col_entry.success(f"**Ideální vstup:**\n${long_entry:.2f}")
                    col_sl.warning(f"**Stop Loss:**\n${long_stop_loss:.2f}")
                    col_tp.info(f"**Take Profit:**\n${long_take_profit:.2f}")
                    
                    if risk_reward_ratio >= 1.5:
                        col_rr.success(f"**Risk/Reward (R:R):**\n1 : {risk_reward_ratio:.2f} 🟢")
                    else:
                        col_rr.error(f"**Risk/Reward (R:R):**\n1 : {risk_reward_ratio:.2f} ⚠️")

                    st.markdown("---")
                    st.markdown("#### 💰 Kalkulačka velikosti pozice")
                    col_cap1, col_cap2 = st.columns(2)
                    with col_cap1:
                        user_capital = st.number_input(f"Celkový kapitál ($) pro {ticker}:", value=5000.0, step=500.0, key=f"cap_{ticker}")
                    with col_cap2:
                        risk_pct = st.slider(f"Riziko na obchod (%):", 0.5, 3.0, 1.0, key=f"risk_{ticker}")

                    allowed_risk_usd = user_capital * (risk_pct / 100.0)
                    risk_per_share = 1.5 * atr_val
                    shares_to_buy = int(allowed_risk_usd / risk_per_share) if risk_per_share > 0 else 0
                    total_position_value = shares_to_buy * actual_price
                    if total_position_value > user_capital:
                        shares_to_buy = int(user_capital / actual_price)
                        total_position_value = shares_to_buy * actual_price

                    st.info(f"👉 **Provedení:** Koupit **{shares_to_buy} ks** | **Hodnota:** `${total_position_value:.2f}` | **Max. riziko:** `${allowed_risk_usd:.2f}`")

                    if earnings_days != 999 and earnings_days <= 7:
                        st.error(f"⚠️ **VÝSLEDKY ZA {earnings_days} DNŮ:** ({earnings_date_str}). Riziko mezery v grafu!")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    model.plot(forecast, ax=ax)
                    ax.set_title(f"20denní cenová předpověď: {ticker} (Zapsáno do DB)")
                    st.pyplot(fig)

        if analyzed_count == 0 or not valid_results:
            st.warning("⚠️ Žádná aktiva neodpovídají zvoleným filtrům.")

elif app_mode == "🧠 Historie predikcí & Učení AI":
    st.subheader("🧠 Vyhodnocení predikcí a učení se z minulosti")
    st.markdown("Tato sekce stahuje zapsané predikce z databáze Supabase, porovnává je s aktuální reálnou cenou a ukazuje úspěšnost AI modelů.")
    
    if supabase is not None:
        if st.button("🔄 Načíst a vyhodnotit predikce z databáze", type="primary"):
            with st.spinner("Stahuji predikce a ověřuji reálné ceny přes Yahoo Finance..."):
                try:
                    response = supabase.table("predictions").select("*").execute()
                    rows = response.data
                    
                    if not rows:
                        st.info("V databázi zatím nejsou uloženy žádné predikce. Spustťe nejdřív sken v záložce Tržní skener.")
                    else:
                        eval_data = []
                        success_count = 0
                        total_evaluated = 0

                        for row in rows:
                            pred_id = row["id"]
                            ticker = row.get("ticker", "N/A")
                            
                            raw_entry = row.get("entry_price")
                            entry_price = float(raw_entry) if raw_entry is not None else 0.0

                            raw_pred = row.get("predicted_price")
                            predicted_price = float(raw_pred) if raw_pred is not None else 0.0

                            target_date = row.get("target_date", datetime.now().strftime('%Y-%m-%d'))
                            status = row.get("status", "PENDING")
                            
                            try:
                                cur_t = yf.Ticker(ticker)
                                hist = cur_t.history(period="1d")
                                if not hist.empty:
                                    if isinstance(hist.columns, pd.MultiIndex):
                                        hist.columns = hist.columns.get_level_values(0)
                                    current_real_price = float(hist['Close'].iloc[-1])
                                else:
                                    current_real_price = entry_price
                            except:
                                current_real_price = entry_price

                            today_date_obj = datetime.now().date()
                            try:
                                target_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                            except:
                                target_date_obj = today_date_obj
                            
                            calculated_status = status
                            
                            if today_date_obj >= target_date_obj and entry_price > 0:
                                expected_dir = predicted_price > entry_price
                                actual_dir = current_real_price > entry_price
                                
                                if expected_dir == actual_dir:
                                    calculated_status = "SUCCESS 🟢"
                                    success_count += 1
                                else:
                                    calculated_status = "FAILED 🔴"
                                total_evaluated += 1
                                
                                try:
                                    supabase.table("predictions").update({
                                        "actual_price_at_target": current_real_price,
                                        "status": calculated_status
                                    }).eq("id", pred_id).execute()
                                except:
                                    pass
                            else:
                                calculated_status = "PENDING ⏳ (Probíhá)"

                            eval_data.append({
                                "Ticker": ticker,
                                "Vstup ($)": f"${entry_price:.2f}",
                                "Cíl ($)": f"${predicted_price:.2f}",
                                "Cílové datum": target_date,
                                "Aktuální/Reálná ($)": f"${current_real_price:.2f}",
                                "Stav": calculated_status
                            })
                        
                        if total_evaluated > 0:
                            winrate = (success_count / total_evaluated) * 100
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Vyhodnoceno predikcí", total_evaluated)
                            col_m2.metric("Úspěšné trefy", success_count)
                            col_m3.metric("Úspěšnost modelu (Winrate)", f"{winrate:.1f}%")

                        df_eval = pd.DataFrame(eval_data)
                        st.dataframe(df_eval, use_container_width=True)
                except Exception as db_ex:
                    st.error(f"Chyba při komunikaci s databází: {db_ex}")
        else:
            st.info("Stiskněte tlačítko výše pro aktualizaci a porovnání stavu predikcí oproti aktuálním tržním cenám.")
    else:
        st.warning("⚠️ Databáze Supabase není připojena. Nelze načítat historii predikcí.")

elif app_mode == "🤖 Klondike Agent Hub":
    st.subheader("🤖 Klondike Spot Agent Hub")
    st.markdown("Sledování automatizovaných detektorů ranních impulzů a zápisů do databáze.")
    agent = KlondikeExecutionAgent()
    st.success(f"**Stav agenta:** {agent.status}")
    for proto in agent.protocols:
        st.markdown(f"- ✅ `{proto}`")

elif app_mode == "📘 Uživatelská příručka":
    st.subheader("📘 Uživatelská příručka & Systém učení AI")
    st.markdown("""
    * **Automatické ukládání predikcí:** Při každém spuštění skenu se aktuální předpověď ceny na 20 dní dopředu odešle do tabulky `predictions` v Supabase.
    * **Učení a vyhodnocení:** V záložce **Historie predikcí & Učení AI** můžete sledovat, jak se modely trefují do reálného vývoje. Jakmile uplyne cílové datum, systém sám označí predikci jako Úspěch (SUCCESS) nebo Neúspěch (FAILED).
    * **Udržení aktivity projektu:** Díky pravidelným zápisům do databáze se eliminuje riziko, že Supabase projekt uspí kvůli 7denní neaktivitě.
    """)
