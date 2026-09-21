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
    /* Hlavní pozadí a typografie */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Elegantní karty / kontejnery */
    .stExpander {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Tlačítka */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid #1f2937;
    }
    
    /* Metriky */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #f9fafb !important;
    }
    
    /* Nadpisy */
    h1, h2, h3 {
        letter-spacing: -0.5px;
        color: #f9fafb;
    }
    
    /* Vylepšené tabulky */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
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

st.title("⚡ Klondike AI Investment & Swing Terminal")
st.markdown("<p style='font-size: 1.1em; color: #9ca3af;'>Institucionální analytika, predikční modely Prophet, sledování pre-marketu a zpětná validace úspěšnosti.</p>", unsafe_allow_html=True)

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

DEFAULT_TICKERS = [
    "NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "WMT", "LLY",
    "MU", "JPM", "ORCL", "XOM", "V", "MA", "AMD", "NFLX", "JNJ", "COST", "HD", "CRM", 
    "UNH", "PG", "ABBV", "BAC", "IBM", "DIS", "INTC", "KO", "PLTR", "UBER", "PYPL", "PFE", "NKE",
    "ASML", "TSM", "NVO", "BABA", "TM", "AZN", "SHEL", "NSRGY", "SAP", "TTE", "HSBC", "SONY", "MELI", "RIO", "BP",
    "SPY"
]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Přidáno: {custom_ticker_input}")

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

# --- POMOCNÉ FUNKCE PRO VÝPOČTY ---
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
        return "Pre-market data nejsou k dispozici (mimo obchodní hodiny)."
    if pre_change >= 3.0:
        return f"🚨 **BÝČÍ PRE-MARKET SKOK (+{pre_change:.2f}%):** Silný ranní nákupní tlak."
    elif pre_change <= -3.0:
        return f"⚠️ **MEDVĚDÍ PRE-MARKET PROPAD ({pre_change:.2f}%):** Ranní výprodej."
    elif pre_change > 0:
        return f"🟢 Mírný ranní růst (+{pre_change:.2f}%)."
    elif pre_change < 0:
        return f"🔴 Mírný ranní pokles ({pre_change:.2f}%)."
    else:
        return "⚖️ Pre-market bez pohybu (0%)."

def get_next_earnings_days(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if dates:
                next_date = pd.to_datetime(dates[0])
                return (next_date - pd.Timestamp.now()).days, next_date.strftime('%Y-%m-%d')
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            future_dates = ed[ed.index > pd.Timestamp.now()]
            if not future_dates.empty:
                next_date = future_dates.index[0]
                return (next_date - pd.Timestamp.now()).days, next_date.strftime('%Y-%m-%d')
    except Exception:
        pass
    return 999, "N/A"

def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (Žádné zprávy)", "Nadpis nenalezen"
        bearish_keywords = ["sue", "lawsuit", "fine", "drop", "plunge", "decline", "crash", "loss", "miss"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain"]
        score = 0
        latest_headline = "Neznámý nadpis"
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if title:
                if latest_headline == "Neznámý nadpis":
                    latest_headline = title
                title_lower = title.lower()
                for kw in bullish_keywords:
                    if kw in title_lower: score += 1
                for kw in bearish_keywords:
                    if kw in title_lower: score -= 1
        if score > 0: return "📈 BÝČÍ", latest_headline
        elif score < 0: return "📉 MEDVĚDÍ", latest_headline
        else: return "➖ NEUTRÁLNÍ", latest_headline
    except Exception:
        return "➖ (Zprávy nedostupné)", "Chyba"

# --- NAVIGAČNÍ ZÁLOŽKY ---
app_mode = st.radio("Navigace:", [
    "📊 Tržní skener & Signály", 
    "🧠 Historie predikcí & Úspěšnost AI",
    "🤖 Klondike Agent Hub",
    "📘 Uživatelská příručka"
], horizontal=True)

# ==========================================
# 1. ZÁLOŽKA: SKENER
# ==========================================
if app_mode == "📊 Tržní skener & Signály":
    st.markdown("### 🚀 Spuštění analýzy")
    if st.button("Spustit komplexní sken trhu & uložit predikce", type="primary", use_container_width=True):
        st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        with st.spinner("Analyzuji aktiva, S&P 500, počítám Prophet modely a zapisuji do databáze..."):
            try:
                sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                if isinstance(sp500.columns, pd.MultiIndex):
                    sp500.columns = sp500.columns.get_level_values(0)
                sp500_close = float(sp500['Close'].iloc[-1])
                sp500_30d_return = (sp500_close - float(sp500['Close'].iloc[-30])) / float(sp500['Close'].iloc[-30]) * 100 if len(sp500) >= 30 else 0
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
                gain_per_1_usd = (peak_20d - actual_price) / actual_price if actual_price > 0 else 0

                asset_30d_return = (actual_price - float(data['Close'].iloc[-30])) / float(data['Close'].iloc[-30]) * 100
                rs_vs_sp500 = asset_30d_return - sp500_30d_return

                if filter_high_gain and gain_per_1_usd < 0.06: continue
                is_breakout, is_breakdown, is_squeeze = detect_patterns(data)
                current_vol, avg_vol, is_vol_spike, vol_ratio = analyze_volume(data)
                
                if filter_breakout and not is_breakout: continue
                if filter_squeeze and not is_squeeze: continue
                if filter_volume and not is_vol_spike: continue
                if filter_outperforming and rs_vs_sp500 < 0: continue

                earnings_days, earnings_date_str = get_next_earnings_days(t_obj)
                if filter_safe_earnings and earnings_days <= 7: continue

                analyzed_count += 1
                news_sentiment, latest_headline = analyze_news_sentiment(t_obj)
                rsi_val = calculate_rsi(data)
                atr_val = calculate_atr(data)
                macd_line, macd_signal, macd_hist = calculate_macd(data)
                
                pattern_label = "⚖️ Standardní pohyb"
                if is_breakout and is_vol_spike: pattern_label = "🚀 Silný objemový průraz"
                elif is_breakout: pattern_label = "↗️ Cenový průraz"
                elif is_squeeze: pattern_label = "📦 Konsolidace (BB Squeeze)"
                elif is_breakdown: pattern_label = "📉 Riziko propadu"

                long_entry = actual_price
                long_stop_loss = actual_price - (1.5 * atr_val)
                long_take_profit = actual_price + (2.5 * atr_val)
                risk_reward_ratio = (long_take_profit - long_entry) / (long_entry - long_stop_loss)

                # Prophet Model
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
                    except:
                        pass

                trend_pct = float(np.clip(((rsi_val - 50) / 50 * 30) + (np.clip(rs_vs_sp500, -20, 20) / 20 * 40) + (30 if macd_hist > 0 else -30), -100, 100))
                trend_text = f"Býčí trend (+{trend_pct:.1f}%)" if trend_pct > 0 else f"Medvědí trend ({trend_pct:.1f}%)"

                if is_breakout and is_vol_spike and rsi_val < 70:
                    action_rec = "🚀 SILNÝ NÁKUP (Průraz s objemem)"
                elif is_squeeze:
                    action_rec = "📦 AKUMULACE / Squeeze"
                elif rs_vs_sp500 > 0 and rsi_val <= 65:
                    action_rec = "🛒 NÁKUP NA PROPADECH (Outperforms)"
                elif rsi_val > 75 or is_breakdown:
                    action_rec = "⚠️ REDUKOVAT / PRODAT"
                else:
                    action_rec = "⏳ ČEKAT / SLEDOVAT"

                score = rs_vs_sp500 + (vol_ratio * 10) if is_vol_spike else rs_vs_sp500

                valid_results.append({
                    "ticker": ticker, "actual_price": actual_price, "pre_price": pre_price, "pre_change": pre_change,
                    "gain_per_1_usd": gain_per_1_usd, "rs_vs_sp500": rs_vs_sp500, "rsi_val": rsi_val, "vol_ratio": vol_ratio,
                    "is_vol_spike": is_vol_spike, "pattern_label": pattern_label, "long_entry": long_entry,
                    "long_stop_loss": long_stop_loss, "long_take_profit": long_take_profit, "risk_reward_ratio": risk_reward_ratio,
                    "news_sentiment": news_sentiment, "latest_headline": latest_headline, "earnings_days": earnings_days,
                    "earnings_date_str": earnings_date_str, "trend_pct": trend_pct, "trend_text": trend_text,
                    "action_rec": action_rec, "score": score, "forecast": forecast, "model": model, "atr_val": atr_val
                })
            except Exception as e:
                pass

        if analyzed_count > 0 and valid_results:
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
            st.markdown(f"### 🎯 Nalezeno aktiv splňujících kritéria: {len(valid_results)}")

            for res in valid_results:
                ticker = res["ticker"]
                actual_price = res["actual_price"]
                pre_price = res["pre_price"]
                pre_change = res["pre_change"]
                action_rec = res["action_rec"]
                
                pre_str = f" | Pre-market: ${pre_price:.2f} ({pre_change:+.2f}%)" if pre_price is not None else ""

                with st.expander(f"📌 {ticker} — Cena: ${actual_price:.2f}{pre_str} | Signál: {action_rec}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Závěrečná cena", f"${actual_price:.2f}")
                        st.metric("RSI (14)", f"{res['rsi_val']:.1f}")
                    with col2:
                        st.metric("Relativní síla vs S&P", f"{res['rs_vs_sp500']:+.2f}%")
                        st.metric("Zisk / 1 $ investice", f"+${res['gain_per_1_usd']:.2f}")
                    with col3:
                        st.metric("Risk / Reward Racio", f"1 : {res['risk_reward_ratio']:.2f}")
                        st.write(f"**Vzorec:** {res['pattern_label']}")

                    st.markdown("---")
                    c_en, c_sl, c_tp = st.columns(3)
                    c_en.success(f"**Vstup:**\n${res['long_entry']:.2f}")
                    c_sl.warning(f"**Stop Loss:**\n${res['long_stop_loss']:.2f}")
                    c_tp.info(f"**Take Profit:**\n${res['long_take_profit']:.2f}")

                    if res['earnings_days'] <= 7:
                        st.error(f"⚠️ Výsledky firmy za {res['earnings_days']} dnů ({res['earnings_date_str']})!")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#111827')
                    ax.set_facecolor('#0b0f19')
                    res['model'].plot(res['forecast'], ax=ax)
                    ax.set_title(f"20denní Prophet predikce: {ticker}", color='white')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    st.pyplot(fig)
        else:
            st.warning("⚠️ Žádná aktiva nevyhovují zvoleným filtrům.")

# ==========================================
# 2. ZÁLOŽKA: HISTORIE PREDIKCÍ & ÚSPĚŠNOST
# ==========================================
elif app_mode == "🧠 Historie predikcí & Úspěšnost AI":
    st.subheader("🧠 Zpětná validace a učení modelů")
    st.markdown("Tento modul stahuje historické predikce ze Supabase, porovnává je s reálným vývojem a počítá celkovou úspěšnost (Winrate).")
    
    if supabase is not None:
        if st.button("🔄 Aktualizovat a vyhodnotit stav predikcí z DB", type="primary"):
            with st.spinner("Stahuji data a ověřuji tržní ceny..."):
                try:
                    response = supabase.table("predictions").select("*").execute()
                    rows = response.data
                    
                    if not rows:
                        st.info("V tabulce `predictions` zatím nejsou žádné záznamy.")
                    else:
                        eval_data = []
                        success_count = 0
                        total_evaluated = 0

                        for row in rows:
                            pred_id = row["id"]
                            ticker = row["ticker"]
                            entry_price = float(row["entry_price"])
                            predicted_price = float(row["predicted_price"])
                            target_date = row["target_date"]
                            status = row["status"]
                            
                            try:
                                cur_t = yf.Ticker(ticker)
                                hist = cur_t.history(period="1d")
                                current_real_price = float(hist['Close'].iloc[-1]) if not hist.empty else entry_price
                            except:
                                current_real_price = entry_price

                            today_obj = datetime.now().date()
                            target_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
                            
                            calculated_status = status
                            if today_obj >= target_obj:
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
                                calculated_status = "PENDING ⏳"

                            eval_data.append({
                                "Ticker": ticker,
                                "Vstup": f"${entry_price:.2f}",
                                "Predikce (Cíl)": f"${predicted_price:.2f}",
                                "Cílové datum": target_date,
                                "Reálná cena": f"${current_real_price:.2f}",
                                "Výsledek": calculated_status
                            })

                        # Zobrazení metrik úspěšnosti
                        if total_evaluated > 0:
                            winrate = (success_count / total_evaluated) * 100
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Vyhodnoceno predikcí", total_evaluated)
                            col_m2.metric("Úspěšné trefy", success_count)
                            col_m3.metric("Úspěšnost modelu (Winrate)", f"{winrate:.1f}%")

                        df_eval = pd.DataFrame(eval_data)
                        st.dataframe(df_eval, use_container_width=True)
                except Exception as ex:
                    st.error(f"Chyba při komunikaci s databází: {ex}")
        else:
            st.info("Stiskněte tlačítko pro načtení aktuálních výsledků úspěšnosti z databáze.")
    else:
        st.warning("⚠️ Databáze není připojena.")

# ==========================================
# 3. ZÁLOŽKA: AGENT HUB
# ==========================================
elif app_mode == "🤖 Klondike Agent Hub":
    st.subheader("🤖 Klondike Execution Hub")
    agent = KlondikeExecutionAgent()
    st.success(f"**Stav:** {agent.status}")
    st.markdown("#### Aktivní bezpečnostní protokoly:")
    for proto in agent.protocols:
        st.markdown(f"- ✅ `{proto}`")

# ==========================================
# 4. ZÁLOŽKA: PŘÍRUČKA
# ==========================================
elif app_mode == "📘 Uživatelská příručka":
    st.subheader("📘 Uživatelská příručka systému")
    st.markdown("""
    * **Automatická aktivita:** Každý spuštěný sken odešle aktuální Prophet predikci do Supabase, což spolehlivě zabrání pozastavení projektu kvůli nečinnosti.
    * **Zpětné učení:** Modely se vyhodnocují na základě toho, zda se trefily do správného **směru cenového pohybu** po uplynutí 20 dnů.
    """)
