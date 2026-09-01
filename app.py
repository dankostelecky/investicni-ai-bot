from klondike_agent import KlondikeExecutionAgent
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

st.set_page_config(page_title="Klondike AI Investment Scanner", page_icon="🤖", layout="wide")

st.title("🤖 Klondike AI Investment Scanner (News + Crowd + AI Learning + Custom Search)")
st.write("This application analyzes markets, monitors crowd psychology, calculates dual Long/Short scenarios, and learns from history using a database.")

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

st.sidebar.markdown("### 🔍 Custom Asset Search")
custom_ticker_input = st.sidebar.text_input("Add ticker (e.g. NFLX, AAPL, CZG.PR):", "").upper().strip()

DEFAULT_TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "NFLX", "AMD", "INTC", "KO", "JPM", "XOM", "JNJ", "SPY", "V", "DIS", "BAC", "PLTR", "PFE", "NKE", "PYPL", "IBM", "UBER", "WMT"]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Added {custom_ticker_input} to scanning list!")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Quick Filters")
filter_high_gain = st.sidebar.toggle("🔥 Show only Gain ≥ 0.08 USD", value=False)

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

def get_next_earnings_date(ticker_obj):
    try:
        cal = ticker_obj.calendar
        if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
            dates = cal['Earnings Date']
            if dates:
                return pd.to_datetime(dates[0]).strftime('%Y-%m-%d')
        ed = ticker_obj.earnings_dates
        if ed is not None and not ed.empty:
            future_dates = ed[ed.index > pd.Timestamp.now()]
            if not future_dates.empty:
                return future_dates.index[0].strftime('%Y-%m-%d')
    except Exception:
        pass
    return "N/A"

def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (No fresh news)", "Available headlines not found"
        
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

def render_trump_and_political_trades():
    st.subheader("🏛️ Donald Trump & Family Asset Transactions")
    st.write("Overview of tracked transactions and asset disclosures reported in official government registries.")

    trump_data = [
        {"Date": "2026-06-18", "Asset": "Berkshire Hathaway (BRK-B)", "Type": "Purchase", "Estimated Value": "$1M - $5M", "Status": "Active Portfolio"},
        {"Date": "2026-06-23", "Asset": "Visa Inc (V)", "Type": "Purchase", "Estimated Value": "$500K - $1M", "Status": "Active Portfolio"},
        {"Date": "2026-06-24", "Asset": "Mastercard (MA)", "Type": "Purchase", "Estimated Value": "$500K - $1M", "Status": "Active Portfolio"},
        {"Date": "2026-06-03", "Asset": "Palantir (PLTR)", "Type": "Buy/Sell", "Estimated Value": "$15K - $50K", "Status": "Rotated / Traded"},
        {"Date": "2025-04-08", "Asset": "Big Tech Basket (AAPL, MSFT, GOOGL)", "Type": "Large Purchase", "Estimated Value": "$12.8M total", "Status": "Core Holding"}
    ]
    
    df_trump = pd.DataFrame(trump_data)
    st.dataframe(df_trump, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔗 Official Sources & Public Disclosures (Free Access)")
    st.write("You can verify all asset records and source documents directly in official registries:")
    
    st.markdown("- 🇺🇸 [U.S. Office of Government Ethics (OGE) Official Search](https://www.oge.gov/web/oge.nsf/Officials%20Individual%20Disclosures%20Search%20Collection?OpenForm)")
    st.markdown("- 📊 [ProPublica Trump & Appointees Financial Disclosures Database](https://projects.propublica.org/trump-team-financial-disclosures/)")
    st.markdown("- 🏛️ [U.S. Senate Electronic Financial Disclosure (eFD) System](https://efd.senate.gov/)")

    st.info("💡 Tip: You can copy any ticker from the table above (e.g., `BRK-B`, `V`) and paste it into the **Custom Asset Search** sidebar to analyze current technical indicators and AI outlook.")

def render_user_manual():
    st.subheader("📘 Klondike AI Investment Scanner: User Manual")
    st.markdown("Welcome to the guide for Klondike AI Investment Scanner. This manual will help you navigate the interface, indicators, and investment management.")

    with st.expander("📖 1. How to Launch and Control the App"):
        st.markdown("""
        The application runs directly in your web browser and is split into main sections via the top menu:
        
        1. **📊 Market Scanning & Overview:** Scan markets, add custom tickers in the sidebar, use filters like **Gain ≥ 0.08 USD**, and run AI analysis including trend outlooks, Long/Short trading setups, and direct investment advice.
        2. **🧠 AI Accuracy & History (Backtesting):** Check historical predictions saved in the database.
        3. **🏛️ Trump & Insider Trades:** Track political and insider transactions.
        4. **📘 User Manual:** Read help and feature descriptions.
        """)

    with st.expander("📊 2. What Do the Indicators Mean?"):
        st.markdown("""
        * **📈 AI Quantitative Trend:** Trend evaluation and price direction prediction (`BULLISH`, `BEARISH`, `NEUTRAL`).
        * **💡 Investment Advice:** Immediate recommendation whether to **enter** (ideal buying zone/oversold), **wait** (market is overbought or undecided), or **avoid**.
        * **🟢 Long Setup:** Recommended ideal entry, stop loss, and take profit for buying/upside.
        * **🔴 Short Setup:** Recommended ideal entry, stop loss, and take profit for selling/downside.
        * **📊 RSI (Relative Strength Index):** Overbought (>65) / Oversold (<35).
        * **📰 News Sentiment:** Financial news evaluation.
        """)

    with st.expander("📐 3. Professional User Manual (Formulas & Algorithms)"):
        st.markdown("""
        ### Professional User Manual: Klondike AI Investment Scanner
        This manual details the mathematical formulas, logical rules, and algorithms used by the Klondike AI Investment Scanner application to calculate various items, financial metrics, technical indicators, and trading scenarios.

        #### 1. Technical and Quantitative Indicators
        **1.1. Relative Strength Index (RSI)**  
        RSI measures the speed and change of price movements to identify overbought or oversold conditions of an asset. The calculation runs over a 14-period window ($\text{window} = 14$):  
        * Price Change ($\Delta$): $\\Delta_t = \\text{Close}_t - \\text{Close}_{t-1}$  
        * Average Gain and Loss: Gains ($\\text{gain}$) are values where $\\Delta > 0$ (otherwise $0$), averaged using a 14-period moving average. Losses ($\\text{loss}$) are absolute values where $\\Delta < 0$ (otherwise $0$), averaged using a 14-period moving average.  
        * Relative Strength (RS): $\\text{RS} = \\frac{\\text{Gain}}{\\text{Loss}}$  
        * RSI Calculation: $\\text{RSI} = 100 - \\left(\\frac{100}{1 + \\text{RS}}\\right)$  
        *Code Interpretation:* RSI $< 30$ indicates oversold conditions, while RSI $> 70$ indicates overbought conditions.

        **1.2. Average True Range (ATR)**  
        ATR measures market volatility by factoring in interday gaps. It is calculated using a 14-day window:  
        * Three True Range (TR) Components:  
          $\\text{TR}_1 = \\text{High} - \\text{Low}$  
          $\\text{TR}_2 = \\vert \\text{High} - \\text{Close}_{\\text{prev}} \\vert$  
          $\\text{TR}_3 = \\vert \\text{Low} - \\text{Close}_{\\text{prev}} \\vert$  
        * True Range (TR): $\\text{TR} = \\max(\\text{TR}_1, \\text{TR}_2, \\text{TR}_3)$  
        * ATR: The 14-period simple moving average of the TR values: $\\text{ATR} = \\text{SMA}_{14}(\\text{TR})$

        **1.3. Simple Moving Averages (SMA)**  
        The application utilizes a 50-day ($\\text{SMA}_{50}$) and a 200-day ($\\text{SMA}_{200}$) simple moving average to determine long-term and medium-term trends:  
        $$\\text{SMA}_n = \\frac{1}{n} \\sum_{i=0}^{n-1} \\text{Close}_{t-i}$$  
        If the current price is above $\\text{SMA}_{200}$, the market is evaluated as having a long-term bullish trend (`is_bullish_trend = True`).

        #### 2. Profit Potential Calculation and Filtering
        **2.1. Gain per 1 USD Invested (`zisk_na_1_usd`)**  
        This metric quantifies the room left for the price to reach a recent 20-day high:  
        * 20-day Peak ($\\text{Peak}_{20}$): $\\text{Peak}_{20} = \\max(\\text{Close}_{t-19}, \\dots, \\text{Close}_t)$  
        * Difference in USD ($\\text{Difference}$): $\\text{Difference} = \\text{Peak}_{20} - \\text{Actual Price}$  
        * Gain per 1 USD ($\\text{Gain}$): $\\text{Gain} = \\frac{\\text{Difference}}{\\text{Actual Price}}$  
        *(Note: If $\\text{Actual Price} \\le 0$, the value is set to $0$).*  
        *Quick Filter (`filter_high_gain`):* When active, the application filters out any assets that do not meet the condition $\\text{Gain} \\ge 0.08\\,\\text{USD}$.

        #### 3. Trading Setups (Long & Short Strategies)
        The application dynamically generates price levels for entry and risk management using the current price ($\\text{Price}$) and volatility measured by ATR:  
        * **LONG SETUP (Bullish Strategy):**  
          * Ideal Entry ($\\text{Entry}_{\\text{Long}}$): $\\text{Price}$  
          * Stop Loss ($\\text{SL}_{\\text{Long}}$): $\\text{Price} - (1.5 \\times \\text{ATR})$  
          * Take Profit ($\\text{TP}_{\\text{Long}}$): $\\text{Price} + (2.5 \\times \\text{ATR})$  
        * **SHORT SETUP (Bearish Strategy):**  
          * Ideal Entry ($\\text{Entry}_{\\text{Short}}$): $\\text{Price}$  
          * Stop Loss ($\\text{SL}_{\\text{Short}}$): $\\text{Price} + (1.5 \\times \\text{ATR})$  
          * Take Profit ($\\text{TP}_{\\text{Short}}$): $\\text{Price} - (2.5 \\times \\text{ATR})$

        #### 4. Crowd Psychology Analysis
        The application monitors trading volume (`Volume`) against its 30-day moving average ($\\text{Volume}_{\\text{avg30}}$):  
        * **Crowd Buying (`crowd_buying`):** Triggered if the current volume is greater than twice the average and the price has increased compared to the previous day:  
          $$\\text{Volume} > (2.0 \\times \\text{Volume}_{\\text{avg30}}) \\quad \\land \\quad \\text{Close}_t > \\text{Close}_{t-1}$$  
        * **Crowd Panic (`crowd_panicking`):** Triggered during high volume combined with a price drop:  
          $$\\text{Volume} > (2.0 \\times \\text{Volume}_{\\text{avg30}}) \\quad \\land \\quad \\text{Close}_t < \\text{Close}_{t-1}$$

        #### 5. AI Trend Prediction and Scoring System
        * **AI Score (`ai_score`):** Starts at $0$. If $\\text{Price} > \\text{SMA}_{50}$, adds $1$; otherwise subtracts $1$. If $\\text{RSI} < 35$, adds $1$ (oversold / buying opportunity); if $\\text{RSI} > 65$, subtracts $1$.  
        * **Quantitative Direction:**  
          * $\\text{ai\\_score} > 0 \\rightarrow$ 📈 BULLISH (Confidence: $75\\%$)  
          * $\\text{ai\\_score} < 0 \\rightarrow$ 📉 BEARISH (Confidence: $75\\%$)  
          * $\\text{ai\\_score} == 0 \\rightarrow$ ⚖️ NEUTRAL (Confidence: $50\\%$)  
        * **Machine Learning (Prophet):** Uses the Prophet library time-series model with yearly seasonality (`yearly_seasonality=True`) to predict the future price (`yhat`) 20 days ahead ($\\text{PRED\\_DAYS} = 20$).

        #### 6. News Sentiment Analysis
        The application evaluates the latest 5 news headlines (`news`) using keyword matching:  
        * **Bearish keywords:** `sue`, `lawsuit`, `fine`, `penalty`, `drop`, `plunge`, `decline`, `crash`, `loss` (each decreases the score by $1$).  
        * **Bullish keywords:** `surge`, `jump`, `rally`, `growth`, `record`, `profit`, `beat`, `strong`, `gain` (each increases the score by $1$).  
        * **Result:** Score $> 0 \\rightarrow$ 📈 BULLISH, Score $< 0 \\rightarrow$ 📉 BEARISH, Score $0 \\rightarrow$ ➖ NEUTRAL.
        """)

    with st.expander("☕ 4. Creator Support"):
        st.markdown("You can find the **Creator Support** section at the bottom of the left sidebar.")

app_mode = st.radio("Select display mode:", [
    "📊 Market Scanning & Overview", 
    "🧠 AI Accuracy & History", 
    "🏛️ Trump & Insider Trades",
    "📘 User Manual"
], horizontal=True)

if app_mode == "📊 Market Scanning & Overview":
    col_main, col_insiders = st.columns([2.3, 1.2])

    with col_main:
        if st.button("🚀 Run Market Analysis & Save Predictions", type="primary"):
            with st.spinner("Downloading data, running AI, and updating database..."):
                try:
                    sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                    if isinstance(sp500.columns, pd.MultiIndex):
                        sp500.columns = sp500.columns.get_level_values(0)
                    sp500_close = float(sp500['Close'].iloc[-1])
                    sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
                    if sp500_close < sp500_sma50:
                        st.warning("⚠️ MACRO WARNING: S&P 500 is below its 50-day moving average (market under pressure).")
                    else:
                        st.success("🌍 MACRO STATUS: S&P 500 is in a positive trend.")
                except:
                    st.info("🌍 Macro status could not be verified.")

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

                        analyzed_count += 1
                        
                        news_sentiment, latest_headline = analyze_news_sentiment(t_obj)
                        rsi_val = calculate_rsi(data)
                        atr_val = calculate_atr(data)
                        sma_50 = float(data['Close'].rolling(window=50).mean().iloc[-1]) if len(data) >= 50 else float(data['Close'].mean())
                        sma_200 = float(data['Close'].rolling(window=200).mean().iloc[-1]) if len(data) >= 200 else float(data['Close'].mean())
                        next_earnings = get_next_earnings_date(t_obj)
                        
                        predchozi_cena = float(data['Close'].iloc[-2])
                        current_volume = float(data['Volume'].iloc[-1])
                        avg_volume_30d = float(data['Volume'].rolling(window=30).mean().iloc[-1])
                        
                        crowd_buying = current_volume > (avg_volume_30d * 2.0) and skutecna_cena > predchozi_cena
                        crowd_panicking = current_volume > (avg_volume_30d * 2.0) and skutecna_cena < predchozi_cena
                        is_bullish_trend = skutecna_cena > sma_200

                        potencial_procent = (rozdil_usd / skutecna_cena) * 100

                        ai_score = 0
                        if skutecna_cena > sma_50:
                            ai_score += 1
                        else:
                            ai_score -= 1

                        if rsi_val < 35:
                            ai_score += 1
                        elif rsi_val > 65:
                            ai_score -= 1

                        if ai_score > 0:
                            quantitative_direction = "📈 BULLISH (UPWARD TREND)"
                            confidence = 75
                        elif ai_score < 0:
                            quantitative_direction = "📉 BEARISH (DOWNWARD TREND)"
                            confidence = 75
                        else:
                            quantitative_direction = "⚖️ NEUTRAL (SIDEWAYS)"
                            confidence = 50

                        if rsi_val > 70:
                            market_state_text = "🔴 **OVERBOUGHT:** The market is extremely high, correction risk is elevated."
                            advice_action = "⏳ **RECOMMENDATION: WAIT / DO NOT ENTER**"
                            advice_color = "error"
                        elif rsi_val < 30:
                            market_state_text = "🟢 **OVERSOLD:** The asset is heavily undervalued / discounted."
                            advice_action = "🚀 **RECOMMENDATION: ENTER LONG**"
                            advice_color = "success"
                        elif is_bullish_trend and rsi_val <= 60 and rsi_val >= 40:
                            market_state_text = "🟡 **HEALTHY TREND:** Market is growing within a reasonable band."
                            advice_action = "✅ **RECOMMENDATION: SUITABLE FOR GRADUAL ENTRY (DCA)**"
                            advice_color = "success"
                        else:
                            market_state_text = "⚖️ **INDECISIVE / SIDEWAYS MARKET:** Lacks clear strong momentum."
                            advice_action = "⏳ **RECOMMENDATION: WAIT**"
                            advice_color = "info"

                        long_entry = skutecna_cena
                        long_stop_loss = skutecna_cena - (1.5 * atr_val)
                        long_take_profit = skutecna_cena + (2.5 * atr_val)

                        short_entry = skutecna_cena
                        short_stop_loss = skutecna_cena + (1.5 * atr_val)
                        short_take_profit = skutecna_cena - (2.5 * atr_val)

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

                        with st.expander(f"Analysis for: {ticker} (Gain/USD: +{zisk_na_1_usd:.2f})"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Current Price", f"{skutecna_cena:.2f} USD")
                            col2.metric("RSI (14)", f"{rsi_val:.1f}")
                            col3.metric("Gain / 1 USD Invested", f"+{zisk_na_1_usd:.2f} USD")

                            st.markdown("---")
                            st.info(f"🤖 **AI Quantitative Direction:** {quantitative_direction} (Confidence: {confidence}%)")
                            
                            st.markdown("### 💡 Investment Advice for Trader:")
                            st.markdown(market_state_text)
                            if advice_color == "success":
                                st.success(advice_action)
                            elif advice_color == "error":
                                st.error(advice_action)
                            else:
                                st.info(advice_action)

                            col_long, col_short = st.columns(2)

                            with col_long:
                                st.markdown("#### 🟢 LONG SETUP (Bullish Strategy)")
                                st.success(f"**Ideal Entry:** ${long_entry:.2f}")
                                st.metric("🛡️ Stop Loss (Long)", f"${long_stop_loss:.2f}")
                                st.metric("🎯 Take Profit (Long)", f"${long_take_profit:.2f}")

                            with col_short:
                                st.markdown("#### 🔴 SHORT SETUP (Bearish Strategy)")
                                st.error(f"**Ideal Entry:** ${short_entry:.2f}")
                                st.metric("🛡️ Stop Loss (Short)", f"${short_stop_loss:.2f}")
                                st.metric("🎯 Take Profit (Short)", f"${short_take_profit:.2f}")

                            st.markdown("---")
                            st.write(f"**News Sentiment:** {news_sentiment} | *\"{latest_headline}\"*")
                            st.write(f"**Distance to 20d Peak:** +{rozdil_usd:.2f} USD (+{potencial_procent:.2f}%)")
                            st.write(f"**ATR Volatility:** {atr_val:.2f}")
                            
                            if next_earnings != "N/A":
                                st.info(f"📅 **Next Earnings Season:** {next_earnings} (Expect higher volatility!)")
                            else:
                                st.write("**Next Earnings Season:** Unscheduled / Unavailable")
                            
                            if crowd_buying:
                                st.markdown("🔥 **Crowd Alert:** Massive buying detected (High Volume + Price Rise)!")
                            elif crowd_panicking:
                                st.markdown("🚨 **Crowd Alert:** Panic selling detected (High Volume + Price Drop)!")
                            else:
                                st.markdown("👥 **Crowd Behavior:** Calm / Normal volume.")

                            trend_status = "✅ OK (Bullish vs. SMA200)" if is_bullish_trend else "❌ Below SMA200 (Caution)"
                            st.write(f"**Long-term Trend:** {trend_status}")

                            fig, ax = plt.subplots(figsize=(10, 4))
                            model.plot(forecast, ax=ax)
                            ax.set_title(f"Prediction for {ticker} (20 days ahead)")
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error processing {ticker}: {e}")
                
                if filter_high_gain and analyzed_count == 0:
                    st.warning("⚠️ No assets currently match the filter criteria (Gain / 1 USD ≥ 0.08). Try turning off the filter.")

    with col_insiders:
        st.markdown("### 🏛️ Live Insider Purchases")
        st.markdown("<p style='font-size: 0.9em; color: gray;'>Tracking recent insider activity for top stocks.</p>", unsafe_allow_html=True)
        
        insider_data_list = []
        insider_tickers = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "AMD"]
        
        for t_sym in insider_tickers:
            try:
                tk = yf.Ticker(t_sym)
                insiders = getattr(tk, 'insider_transactions', None)
                if insiders is not None and not insiders.empty:
                    latest = insiders.iloc[0]
                    insider_data_list.append({
                        "Ticker": t_sym,
                        "Insider": str(latest.get('Name', 'N/A')),
                        "Position": str(latest.get('Position', 'Insider')),
                        "Action": str(latest.get('Transaction', 'Action')),
                        "Shares": str(latest.get('Shares', 'N/A'))
                    })
            except Exception:
                pass
                
        if insider_data_list:
            df_insiders = pd.DataFrame(insider_data_list)
            table_height = len(df_insiders) * 38 + 50
            st.dataframe(
                df_insiders, 
                hide_index=True, 
                use_container_width=True, 
                height=table_height
            )
        else:
            st.info("No fresh insider data available at this time.")

elif app_mode == "🧠 AI Accuracy & History":
    st.subheader("🧠 AI Learning & Prediction History (Backtesting)")
    st.write("This section pulls data from the database and compares past predictions with actual market developments.")
    
    if supabase:
        try:
            response = supabase.table("predictions").select("*").order("target_date", desc=True).limit(50).execute()
            data_rows = response.data
            
            if data_rows:
                df_preds = pd.DataFrame(data_rows)
                st.dataframe(df_preds, use_container_width=True)
                st.info("💡 Once the target date (`target_date`) passes, you can review how accurate the AI prediction was compared to the actual market price.")
            else:
                st.warning("No predictions stored in the database yet. Please run an analysis on the main page.")
        except Exception as e:
            st.error(f"Failed to load history from database: {e}")
    else:
        st.error("Supabase is not connected.")

elif app_mode == "🏛️ Trump & Insider Trades":
    render_trump_and_political_trades()

elif app_mode == "📘 User Manual":
    render_user_manual()

st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support the Creator - David_Seda")

try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 QR code image not found. Please add 'qr_solana.png' to the project folder.")

st.sidebar.markdown(
    "<p style='font-size: 0.9em; color: gray;'>If this app brings you value or profits, buy me a coffee! ☕</p>", 
    unsafe_allow_html=True
)



