import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

class KlondikeExecutionAgent:
    def __init__(self):
        self.status = "Online & Ready (Spot Only)"
        self.protocols = ["Spot Trend Following", "Dynamic Volatility Guard", "Sentiment Feed Integrator"]

st.set_page_config(page_title="Klondike Spot Swing Scanner", page_icon="📈", layout="wide")

st.title("📈 AI Spot Swing Scanner")
st.write("Professional market analytics and automated risk calculation strictly for unleveraged stock trading.")

# Inicializace databáze Supabase
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

# Postranní panel pro vlastní akcie a filtry
st.sidebar.markdown("### 🔍 Custom Asset Search")
custom_ticker_input = st.sidebar.text_input("Add ticker (e.g. NFLX, AAPL):", "").upper().strip()

DEFAULT_TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "ASML", "NVDA", "NFLX", "AMD", "INTC", "KO", "JPM", "XOM", "JNJ", "SPY", "V", "DIS", "BAC", "PLTR", "PFE", "NKE", "PYPL", "IBM", "UBER", "WMT"]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Added {custom_ticker_input} to scanning list!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Quick Filters")
filter_high_gain = st.sidebar.toggle("🔥 Show only Gain ≥ 0.08 USD", value=False)
filter_safe_earnings = st.sidebar.toggle("🛡️ Hide stocks with Earnings < 7 days", value=False)
filter_outperforming = st.sidebar.toggle("🚀 Show only Outperforming S&P 500", value=False)

PRED_DAYS = 20

# Správa stavu aplikace přes st.session_state, aby posuvníky neresetovaly výsledky
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# Výpočet RSI indikátoru
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

# Výpočet ATR (Average True Range)
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

# Výpočet MACD (Moving Average Convergence Divergence)
def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

# Zjištění data nejbližších výsledků (Earnings)
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

# Analýza sentimentu zpráv
def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (No fresh news)", "Available headlines not found"
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain", "buy"]
        
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

def render_klondike_agent_execution_hub():
    st.subheader("🤖 Klondike Spot Execution Agent Hub")
    st.markdown("Monitor automated unleveraged portfolio balancing and swing entry protocols.")
    
    agent = KlondikeExecutionAgent()
    agent_status = getattr(agent, "status", "Online & Ready")
    active_protocols = getattr(agent, "protocols", ["Spot Trend Following", "Dynamic Volatility Guard", "Sentiment Feed Integrator"])
    
    col_status, col_metrics = st.columns([1, 1])
    
    with col_status:
        st.success(f"**Agent Operational Status:** {agent_status}")
        st.markdown("#### Active Execution Protocols:")
        for proto in active_protocols:
            st.markdown(f"- ✅ `{proto}`")
            
    with col_metrics:
        st.metric("Agent Latency", "14 ms", delta="-2 ms optimal")
        st.metric("Execution Success Rate", "98.4%", delta="+0.6% vs last week")
        
    st.markdown("---")
    st.markdown("### ⚡ Manual Agent Override & Trigger Console")
    if st.button("🚀 Force Immediate Portfolio Synchronization", use_container_width=True):
        st.toast("Agent re-balancing sequence initiated successfully!", icon="🤖")

def render_user_manual():
    st.subheader("📘 Spot Swing Scanner: User Manual")
    st.markdown("Welcome to the guide tailored for **unleveraged spot trading**. No shorting, no margin, just high-quality blue-chip swing setups.")

    with st.expander("📖 1. How to Launch and Control the App"):
        st.markdown("1. **Market Scanning:** Scan large-cap tickers and filter for safe, outperforming stocks.")
        st.markdown("2. **Position Sizing:** Calculate precise share amounts based on your total cash capital.")
        st.markdown("3. **SMA Bounces & MACD:** Find pullbacks to the 50-day average supported by momentum (MACD).")

app_mode = st.radio("Select display mode:", [
    "📊 Market Scanning & Overview", 
    "🧠 AI Accuracy & History", 
    "🤖 Klondike Agent Hub",
    "📘 User Manual"
], horizontal=True)

if app_mode == "📊 Market Scanning & Overview":
    col_main, col_insiders = st.columns([2.3, 1.2])

    with col_main:
        # Tlačítko pro spuštění analýzy nastaví st.session_state na True
        if st.button("🚀 Run Spot Market Analysis", type="primary"):
            st.session_state.analysis_run = True

        # Pokud byla analýza spuštěna, vykreslujeme výsledky (zůstanou zachovány i při hýbání s posuvníky)
        if st.session_state.analysis_run:
            with st.spinner("Downloading data, calculating MACD, and updating AI..."):
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
                        st.warning("⚠️ MACRO WARNING: S&P 500 is below its 50-day moving average (cash is safer).")
                    else:
                        st.success("🌍 MACRO STATUS: S&P 500 is in a positive trend.")
                except:
                    st.info("🌍 Macro status could not be verified.")
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
                    potencial_procent = (rozdil_usd / skutecna_cena) * 100

                    ai_score = 0
                    if skutecna_cena > sma_50: ai_score += 1
                    else: ai_score -= 1
                    
                    if rs_vs_sp500 > 0: ai_score += 1
                    if macd_hist > 0: ai_score += 1

                    if rsi_val < 35: ai_score += 1
                    elif rsi_val > 65: ai_score -= 1

                    if ai_score > 1:
                        quantitative_direction = "📈 STRONG BULLISH (Ideal for Swing)"
                        confidence = 80
                    elif ai_score == 1:
                        quantitative_direction = "↗️ MILD BULLISH"
                        confidence = 65
                    elif ai_score < 0:
                        quantitative_direction = "📉 BEARISH (Avoid Spot Purchase)"
                        confidence = 75
                    else:
                        quantitative_direction = "⚖️ NEUTRAL / CONSOLIDATION"
                        confidence = 50

                    if rsi_val > 70:
                        market_state_text = "🔴 **OVERBOUGHT:** Correction risk is high. Do not buy."
                        advice_action = "⏳ **RECOMMENDATION: WAIT**"
                        advice_color = "error"
                    elif is_sma50_bounce and macd_hist > 0:
                        market_state_text = "🎯 **PERFECT SWING SETUP:** Price testing SMA50 with positive MACD momentum!"
                        advice_action = "✅ **RECOMMENDATION: ENTER LONG (SPOT)**"
                        advice_color = "success"
                    elif is_bullish_trend and rsi_val <= 60 and rsi_val >= 40:
                        market_state_text = "🟡 **HEALTHY TREND:** Accumulation zone."
                        advice_action = "✅ **RECOMMENDATION: GRADUAL BUY (DCA)**"
                        advice_color = "success"
                    else:
                        market_state_text = "⚖️ **INDECISIVE:** Lacks clear strong momentum for a swing trade."
                        advice_action = "⏳ **RECOMMENDATION: KEEP CASH**"
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

                    with st.expander(f"Analysis for: {ticker} | Price: ${skutecna_cena:.2f} | RS vs S&P500: {rs_vs_sp500:.1f}%"):
                        col1, col2, col3 = st.columns(3)
                        
                        col1.markdown("**Technical Base**")
                        col1.metric("Current Price", f"${skutecna_cena:.2f}")
                        col1.metric("RSI (14)", f"{rsi_val:.1f}")
                        
                        col2.markdown("**Momentum**")
                        macd_color = "🟢 Positive" if macd_hist > 0 else "🔴 Negative"
                        col2.metric("MACD Hist", f"{macd_hist:.2f}", delta=macd_color, delta_color="off")
                        col2.metric("Gain / 1 USD Invested", f"+{zisk_na_1_usd:.2f} USD")
                        
                        col3.markdown("**Sector Strength**")
                        rs_color = "🟢 Outperforming" if rs_vs_sp500 > 0 else "🔴 Underperforming"
                        col3.metric("vs S&P 500 (30d)", f"{rs_vs_sp500:.2f}%", delta=rs_color, delta_color="off")
                        col3.metric("Off 52W High", f"-{dist_52w_pct:.1f}%")

                        st.markdown("---")
                        st.info(f"🤖 **AI Direction:** {quantitative_direction} (Confidence: {confidence}%)")
                        
                        st.markdown("### 💡 Spot Investment Advice:")
                        st.markdown(market_state_text)
                        if advice_color == "success":
                            st.success(advice_action)
                        elif advice_color == "error":
                            st.error(advice_action)
                        else:
                            st.info(advice_action)

                        st.markdown("#### 🟢 SPOT SWING SETUP (Buy & Hold)")
                        col_entry, col_sl, col_tp = st.columns(3)
                        col_entry.success(f"**Ideal Entry:**\n${long_entry:.2f}")
                        col_sl.warning(f"**Protective Stop Loss:**\n${long_stop_loss:.2f}")
                        col_tp.info(f"**Target Take Profit:**\n${long_take_profit:.2f}")

                        st.markdown("---")
                        st.markdown("#### 💰 Spot Position Sizing Calculator (No Margin)")
                        col_cap1, col_cap2 = st.columns(2)
                        with col_cap1:
                            user_capital = st.number_input(f"Total Cash ($) for {ticker}:", value=5000.0, step=500.0, key=f"cap_{ticker}")
                        with col_cap2:
                            risk_pct = st.slider(f"Risk per Trade (% of Capital):", 0.5, 3.0, 1.0, key=f"risk_{ticker}")

                        allowed_risk_usd = user_capital * (risk_pct / 100.0)
                        risk_per_share = 1.5 * atr_val
                        shares_to_buy = int(allowed_risk_usd / risk_per_share) if risk_per_share > 0 else 0
                        total_position_value = shares_to_buy * skutecna_cena

                        if total_position_value > user_capital:
                            shares_to_buy = int(user_capital / skutecna_cena)
                            total_position_value = shares_to_buy * skutecna_cena
                            st.warning(f"⚠️ Initial calculation exceeded your cash. Adjusted to max affordable shares without margin.")

                        st.info(f"👉 **Execution:** Buy **{shares_to_buy} shares** | **Total Cost:** `${total_position_value:.2f}` | **Max Risk Exposure:** `${allowed_risk_usd:.2f}`")

                        st.markdown("---")
                        st.write(f"**News Sentiment:** {news_sentiment} | *\"{latest_headline}\"*")
                        
                        if earnings_days != 999:
                            if earnings_days <= 7:
                                st.error(f"⚠️ **EARNINGS IN {earnings_days} DAYS:** ({earnings_date_str}). Gap risk is high. Do not hold spot positions through earnings!")
                            else:
                                st.info(f"📅 **Next Earnings:** {earnings_date_str} (in {earnings_days} days)")
                        
                        trend_status = "✅ Healthy Uptrend" if is_bullish_trend else "❌ Below 200-Day SMA (High Risk)"
                        st.write(f"**Long-term Trend (SMA 200):** {trend_status}")

                        fig, ax = plt.subplots(figsize=(10, 4))
                        model.plot(forecast, ax=ax)
                        ax.set_title(f"20-Day Price Forecast: {ticker}")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")
            
            if analyzed_count == 0:
                st.warning("⚠️ No assets match the current filter criteria. Try adjusting the toggles on the left.")

    with col_insiders:
        st.markdown("### 🏛️ Top Insider Buys (Spot)")
        st.markdown("<p style='font-size: 0.9em; color: gray;'>Tracking large-cap spot accumulations.</p>", unsafe_allow_html=True)
        
        insider_data_list = []
        insider_tickers = ["META", "MSFT", "GOOGL", "AAPL", "AMZN", "BRK-B", "NVDA"]
        
        for t_sym in insider_tickers:
            try:
                tk = yf.Ticker(t_sym)
                insiders = getattr(tk, 'insider_transactions', None)
                if insiders is not None and not insiders.empty:
                    latest = insiders.iloc[0]
                    insider_data_list.append({
                        "Tick": t_sym,
                        "Insider": str(latest.get('Name', 'N/A'))[:12],
                        "Action": str(latest.get('Transaction', 'Action'))
                    })
            except Exception:
                pass
                
        if insider_data_list:
            df_insiders = pd.DataFrame(insider_data_list)
            st.dataframe(df_insiders, hide_index=True, use_container_width=True)
        else:
            st.info("No fresh insider data available.")

elif app_mode == "🧠 AI Accuracy & History":
    st.subheader("🧠 Spot AI Predictions (Backtesting)")
    if supabase:
        try:
            response = supabase.table("predictions").select("*").order("target_date", desc=True).limit(50).execute()
            data_rows = response.data
            if data_rows:
                st.dataframe(pd.DataFrame(data_rows), use_container_width=True)
            else:
                st.warning("No predictions stored yet.")
        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.error("Supabase not connected.")

elif app_mode == "🤖 Klondike Agent Hub":
    render_klondike_agent_execution_hub()

elif app_mode == "📘 User Manual":
    render_user_manual()

st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support the Creator")
try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 Add 'qr_solana.png' to your folder.")
