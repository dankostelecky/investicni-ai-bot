import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- VISUAL CONFIGURATION & MODERN GRAPHIC DESIGN (CSS) ---
st.set_page_config(
    page_title="Klondike Spot Swing Scanner", 
    page_icon="📈", 
    layout="wide"
)

st.markdown("""
    <style>
    /* Global Styles & Modern Font/Background */
    .main {
        background-color: #f4f6f9;
        color: #1e293b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        letter-spacing: -0.5px;
        color: #0f172a;
    }
    
    /* Custom Card Containers */
    .stExpander {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }
    
    /* Primary Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
        color: white;
    }
    
    /* Sidebar aesthetic */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metric styling adjustments */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

class KlondikeExecutionAgent:
    def __init__(self):
        self.status = "Active & Ready (Spot Only)"
        self.protocols = ["Spot Trend Tracking", "Dynamic Volatility Protection", "Volume & Pattern Confirmation"]

st.title("📈 AI Spot Swing Scanner & Pattern Analyst")
st.markdown("<p style='font-size: 1.1em; color: #475569;'>Professional market analytics featuring AI insights, volume spike detection, technical breakouts, and S&P 500 relative strength.</p>", unsafe_allow_html=True)

# Supabase Initialization
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning("⚠️ Database not connected")

# Sidebar - Tickers & Filters
st.sidebar.markdown("### 🔍 Asset Search")
custom_ticker_input = st.sidebar.text_input("Add custom ticker (e.g. AAPL, MSFT):", "").upper().strip()

DEFAULT_TICKERS = [
    # Original core tickers
    "META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "ASML", 
    "NVDA", "NFLX", "AMD", "INTC", "KO", "JPM", "XOM", "JNJ", "SPY", "V", 
    "DIS", "BAC", "PLTR", "PFE", "NKE", "PYPL", "IBM", "UBER", "WMT",
    # Added Mega-Cap Giants (Design 1 & 2)
    "AVGO",  # Broadcom
    "LLY",   # Eli Lilly
    "ORCL",  # Oracle
    "COST",  # Costco
    "MA",    # Mastercard
    "HD",    # Home Depot
    "CRM"    # Salesforce
]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Added: {custom_ticker_input} to scanner!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filters & Strategies")

filter_high_gain = st.sidebar.toggle("💵 Gain / 1 $ >= 0.06 $", value=False, help="Displays assets with growth potential to 20-day peak of $0.06 or more per 1 $ invested.")
filter_breakout = st.sidebar.toggle("🚀 Active Breakouts Only", value=False)
filter_squeeze = st.sidebar.toggle("📦 Consolidations Only (BB Squeeze)", value=False)
filter_volume = st.sidebar.toggle("📊 Confirmed Volume Only", value=False)
filter_outperforming = st.sidebar.toggle("⚡ Outperforming S&P 500 Only", value=False)
filter_safe_earnings = st.sidebar.toggle("🛡️ Hide Earnings < 7 Days", value=False)

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
            return "➖ (No fresh news)", "Headline not found", []
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss", "miss"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain", "buy"]
        
        score = 0
        latest_headline = "Unknown headline"
        headlines = []
        
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if title:
                headlines.append(title)
                if latest_headline == "Unknown headline":
                    latest_headline = title
                title_lower = title.lower()
                for kw in bullish_keywords:
                    if kw in title_lower: score += 1
                for kw in bearish_keywords:
                    if kw in title_lower: score -= 1
                
        if score > 0: sentiment_label = "📈 BULLISH"
        elif score < 0: sentiment_label = "📉 BEARISH"
        else: sentiment_label = "➖ NEUTRAL"
        
        return sentiment_label, latest_headline, headlines
    except Exception:
        return "➖ (News unavailable)", "Error loading news", []

app_mode = st.radio("Select View:", [
    "📊 Market Scanner & Patterns", 
    "🤖 Klondike Agent Hub",
    "📘 User Guide"
], horizontal=True)

if app_mode == "📊 Market Scanner & Patterns":
    if st.button("🚀 Run Scan & Pattern Analysis", type="primary", use_container_width=True):
        st.session_state.analysis_run = True

    if st.session_state.analysis_run:
        with st.spinner("Processing S&P 500 benchmark, volume profiles, and chart patterns..."):
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
                
                pattern_label = "⚖️ Standard Movement"
                if is_breakout and is_vol_spike:
                    pattern_label = "🚀 Strong Volume Breakout (Strongly Bullish)"
                elif is_breakout:
                    pattern_label = "↗️ Price Breakout"
                elif is_squeeze:
                    pattern_label = "📦 Consolidation (BB Squeeze)"
                elif is_breakdown:
                    pattern_label = "📉 Breakdown Warning / Selloff"

                long_entry = actual_price
                long_stop_loss = actual_price - (1.5 * atr_val)
                long_take_profit = actual_price + (2.5 * atr_val)

                # Trend score & percentage calculation
                rsi_score_component = (rsi_val - 50) / 50 * 30
                rs_score_component = np.clip(rs_vs_sp500, -20, 20) / 20 * 40
                macd_score_component = 30 if macd_hist > 0 else -30
                
                raw_trend_score = rsi_score_component + rs_score_component + macd_score_component
                trend_pct = float(np.clip(raw_trend_score, -100, 100))

                if trend_pct >= 40:
                    trend_text = f"Strong Bullish Trend (+{trend_pct:.1f}%)"
                elif trend_pct > 0:
                    trend_text = f"Mild Bullish Trend (+{trend_pct:.1f}%)"
                elif trend_pct <= -40:
                    trend_text = f"Strong Bearish Trend / Selloff ({trend_pct:.1f}%)"
                else:
                    trend_text = f"Mild Bearish Trend ({trend_pct:.1f}%)"

                # Action recommendation logic
                if is_breakout and is_vol_spike and rsi_val < 70 and rs_vs_sp500 > 0:
                    action_rec = "🚀 STRONG BUY (Immediate Breakout Confirmation)"
                elif is_squeeze and rs_vs_sp500 >= 0:
                    action_rec = "📦 ACCUMULATE / DCA (Consolidation before expansion)"
                elif rs_vs_sp500 > 0 and rsi_val <= 65:
                    action_rec = "🛒 BUY THE DIP (Outperforming Relative Strength)"
                elif rsi_val > 75 or is_breakdown:
                    action_rec = "⚠️ REDUCE / SELL (Overbought or Breakdown Risk)"
                else:
                    action_rec = "⏳ WAIT / WATCH (Neutral setup, stand by)"

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
                st.error(f"Error processing {ticker}: {e}")
        
        if analyzed_count > 0 and valid_results:
            valid_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)

            st.markdown("### 📊 Filtered Scan Results")

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

                with st.expander(f"📌 {ticker} | Price: ${actual_price:.2f} | Action: {action_rec.split(' ')[0]} {action_rec.split(' ')[1]} | Trend: {trend_text}"):
                    
                    st.markdown(f"### 🎯 Recommended Action: **{action_rec}**")
                    st.markdown(f"📈 **Trend Analysis:** {trend_text}")
                    st.progress(int((trend_pct + 100) / 2))
                    
                    st.markdown("---")

                    col1, col2, col3 = st.columns(3)
                    
                    col1.markdown("**Price & Momentum**")
                    col1.metric("Current Price", f"${actual_price:.2f}")
                    col1.metric("RSI (14)", f"{rsi_val:.1f}")
                    
                    col2.markdown("**Volume & Potential**")
                    vol_status_text = "🔥 Volume Spike" if is_vol_spike else "⚖️ Normal Volume"
                    col2.metric("Volume Ratio", f"{vol_ratio:.2f}x avg", delta=vol_status_text, delta_color="off")
                    col2.metric("Gain / 1 $ Invested", f"+${gain_per_1_usd:.2f}")
                    
                    col3.markdown("**Benchmark & Patterns**")
                    rs_color = "🟢 Outperforming" if rs_vs_sp500 > 0 else "🔴 Underperforming"
                    col3.metric("vs S&P 500 (30d)", f"{rs_vs_sp500:+.2f}%", delta=rs_color, delta_color="off")
                    col3.write(f"**Pattern:** {pattern_label}")

                    st.markdown("---")
                    
                    if st.button(f"🤖 AI Analyst: Evaluate Tech & News for {ticker}", key=f"ai_summary_btn_{ticker}"):
                        with st.spinner("AI linking technical indicators with media sentiment..."):
                            st.markdown("### 🧠 AI Market & Context Evaluation:")
                            
                            news_explanation = ""
                            if headlines_list:
                                news_explanation = f"Current media feedback (e.g. *\"{latest_headline}\"*) indicates the market is reacting to headlines with a **{news_sentiment.lower()}** undertone. "
                            else:
                                news_explanation = "No major fresh headlines were detected in primary news outlets, meaning price action is driven primarily by institutional order flow and technical buying/selling. "

                            vol_explanation = f"Volume activity stands at **{vol_ratio:.2f}x** the regular average ({'confirming aggressive institutional interest or panic' if is_vol_spike else 'under standard liquidity conditions'})."

                            st.write(
                                f"Asset **{ticker}** demonstrates a relative performance of **{rs_vs_sp500:+.2f}%** against the S&P 500 over the past 30 days. "
                                f"Technical condition with RSI at **{rsi_val:.1f}** and pattern **{pattern_label}** signals: {trend_text}. \n\n"
                                f"📰 **Why the stock is moving (AI News Explanation):**\n"
                                f"{news_explanation} {vol_explanation}\n\n"
                                f"💡 **Recommended Next Step:** **{action_rec}**."
                            )

                    st.markdown("---")
                    st.markdown("#### 🟢 SPOT SWING SETUP")
                    col_entry, col_sl, col_tp = st.columns(3)
                    col_entry.success(f"**Ideal Entry:**\n${long_entry:.2f}")
                    col_sl.warning(f"**Stop Loss:**\n${long_stop_loss:.2f}")
                    col_tp.info(f"**Take Profit:**\n${long_take_profit:.2f}")

                    st.markdown("---")
                    st.markdown("#### 💰 Position Sizing Calculator (No Leverage)")
                    col_cap1, col_cap2 = st.columns(2)
                    with col_cap1:
                        user_capital = st.number_input(f"Total Capital ($) for {ticker}:", value=5000.0, step=500.0, key=f"cap_{ticker}")
                    with col_cap2:
                        risk_pct = st.slider(f"Risk per Trade (% of Capital):", 0.5, 3.0, 1.0, key=f"risk_{ticker}")

                    allowed_risk_usd = user_capital * (risk_pct / 100.0)
                    risk_per_share = 1.5 * atr_val
                    shares_to_buy = int(allowed_risk_usd / risk_per_share) if risk_per_share > 0 else 0
                    total_position_value = shares_to_buy * actual_price

                    if total_position_value > user_capital:
                        shares_to_buy = int(user_capital / actual_price)
                        total_position_value = shares_to_buy * actual_price
                        st.warning("⚠️ Initial calculation exceeded available cash. Adjusted to maximum possible share count.")

                    st.info(f"👉 **Execution:** Buy **{shares_to_buy} shares** | **Total Value:** `${total_position_value:.2f}` | **Max Risk:** `${allowed_risk_usd:.2f}`")

                    st.markdown("---")
                    st.markdown("#### 📰 Recent News Feed")
                    st.write(f"**Sentiment:** {news_sentiment}")
                    if headlines_list:
                        for h in headlines_list[:3]:
                            st.markdown(f"- *{h}*")
                    else:
                        st.write("No news available to display.")
                    
                    if earnings_days != 999 and earnings_days <= 7:
                        st.error(f"⚠️ **EARNINGS IN {earnings_days} DAYS:** ({earnings_date_str}). High gap risk!")

                    fig, ax = plt.subplots(figsize=(10, 4))
                    model.plot(forecast, ax=ax)
                    ax.set_title(f"20-Day Price Forecast: {ticker}")
                    st.pyplot(fig)

        if analyzed_count == 0 or not valid_results:
            st.warning("⚠️ No assets match your current pattern definitions and filter criteria.")

elif app_mode == "🤖 Klondike Agent Hub":
    st.subheader("🤖 Klondike Spot Agent Hub")
    st.markdown("Monitoring automated volume anomaly detectors and consolidation breakout protocols.")
    agent = KlondikeExecutionAgent()
    st.success(f"**Agent Status:** {agent.status}")
    for proto in agent.protocols:
        st.markdown(f"- ✅ `{proto}`")

elif app_mode == "📘 User Guide":
    st.subheader("📘 User Guide & Strategy Manual")
    st.markdown("1. **Gain / 1 $ Invested:** Calculates room toward recent highs for every 1 dollar of deployed capital.")
    st.markdown("2. **S&P 500 Outperformance:** Filters stocks demonstrating positive relative strength against the primary market benchmark.")
    st.markdown("3. **Volume Spikes & Squeezes:** Combines high volume confirmation with Bollinger Bands (Squeeze) for explosive swing trades.")
    st.markdown("4. **AI News Evaluation:** Links technical patterns and volumes with current media headlines to explain *why* the stock is behaving a certain way.")
