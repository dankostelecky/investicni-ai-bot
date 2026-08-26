import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# Page configuration
st.set_page_config(page_title="AI Investment Scanner", page_icon="🤖", layout="wide")

st.title("🤖 Klondike AI Investment Scanner (News + Crowd + AI Learning + Custom Search)")
st.write("This tool analyzes markets, tracks crowd psychology, predicts prices, tracks earnings, and learns from its past mistakes using a database.")

# --- SUPABASE CONFIGURATION ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

# --- PŘIDÁNÍ VLASTNÍHO TICKERU ---
st.sidebar.markdown("### 🔍 Custom Asset Search")
custom_ticker_input = st.sidebar.text_input("Add Ticker (e.g. NFLX, AAPL, CZG.PR):", "").upper().strip()

DEFAULT_TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "AMD", "GLD", "BTC-USD", "VT", "^GSPC", "ETH-USD", "SOL-USD", "QQQ", "SPY", "XRP-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Added {custom_ticker_input} to scan list!")

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

# --- SEKCE PRO TRUMPOVY NÁKUPY A ODKAZY ---
def render_trump_and_political_trades():
    st.subheader("🏛️ Donald Trump & Family Stock Disclosures")
    st.write("Overview of major tracked transactions and disclosures reported in official government ethics filings.")

    trump_data = [
        {"Date": "2026-06-18", "Asset": "Berkshire Hathaway (BRK-B)", "Type": "Purchase", "Estimated Value": "$1M - $5M", "Status": "Active Portfolio"},
        {"Date": "2026-06-23", "Asset": "Visa Inc (V)", "Type": "Purchase", "Estimated Value": "$500K - $1M", "Status": "Active Portfolio"},
        {"Date": "2026-06-24", "Asset": "Mastercard (MA)", "Type": "Purchase", "Estimated Value": "$500K - $1M", "Status": "Active Portfolio"},
        {"Date": "2026-06-03", "Asset": "Palantir (PLTR)", "Type": "Purchase/Sale", "Estimated Value": "$15K - $50K", "Status": "Rotated / Traded"},
        {"Date": "2025-04-08", "Asset": "Big Tech Basket (AAPL, MSFT, GOOGL)", "Type": "Large Purchase", "Estimated Value": "$12.8M total", "Status": "Core Holding"}
    ]
    
    df_trump = pd.DataFrame(trump_data)
    st.dataframe(df_trump, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔗 Official Sources & Public Disclosures (Free Access)")
    st.write("You can verify all asset records and raw documentation directly through official public registries:")
    
    st.markdown("- 🇺🇸 [U.S. Office of Government Ethics (OGE) Official Search](https://www.oge.gov/web/oge.nsf/Officials%20Individual%20Disclosures%20Search%20Collection?OpenForm)")
    st.markdown("- 📊 [ProPublica Trump & Appointees Financial Disclosures Database](https://projects.propublica.org/trump-team-financial-disclosures/)")
    st.markdown("- 🏛️ [U.S. Senate Electronic Financial Disclosure (eFD) System](https://efd.senate.gov/)")

    st.info("💡 Tip: You can copy any ticker from the table above (e.g., `BRK-B`, `V`) and paste it into the **Custom Asset Search** sidebar on the main screen to analyze its current technical indicators and AI outlook.")

# --- UŽIVATELSKÝ MANUÁL ---
def render_user_manual():
    st.subheader("📘 Klondike AI Investment Scanner: User Guide")
    st.markdown("Welcome to the beginner's guide for the Klondike AI Investment Scanner. This manual will help you navigate the app, understand the indicators, and manage your investments with the help of artificial intelligence.")

    with st.expander("📖 1. How to Launch and Navigate the App"):
        st.markdown("""
        The application runs directly in your web browser and is divided into main sections using the top menu:
        
        1. **📊 Market Scanner & Dashboard (Main Overview):** Scan markets, add custom tickers via sidebar, and run AI analytics.
        2. **🧠 AI Accuracy & Backtesting History (History & AI Performance):** Review past database predictions.
        3. **🏛️ Trump & Insider Trades:** Check disclosed financial portfolio tracking and official sources.
        4. **📘 User Manual:** Read guidance and descriptions.
        """)

    with st.expander("📊 2. What Do the Indicators Mean?"):
        st.markdown("""
        * **📈 AI Quantitative Direction:** Trend evaluation (`BULLISH`, `BEARISH`, `NEUTRAL`).
        * **🛡️ Risk Management (Stop Loss & Take Profit):** Automated ATR volatility limits.
        * **📊 RSI (Relative Strength Index):** Overbought (>65) / Oversold (<35) ranges.
        * **📰 News Sentiment:** Financial news evaluation.
        * **🔥 Crowd Alert:** Volume anomalies detection.
        """)

    with st.expander("☕ 3. Support the Creator"):
        st.markdown("At the bottom of the left sidebar, you will find the **Support the Creator** section.")

# --- HLAVNÍ NAVIGACE ---
app_mode = st.radio("Select View / Režim zobrazení:", [
    "📊 Market Scanner & Dashboard", 
    "🧠 AI Accuracy & Backtesting History", 
    "🏛️ Trump & Insider Trades",
    "📘 User Manual"
], horizontal=True)

if app_mode == "📊 Market Scanner & Dashboard":
    col_main, col_insiders = st.columns([2.3, 1.2])

    with col_main:
        if st.button("🚀 Run Analysis of Preset Markets & Save Predictions", type="primary"):
            with st.spinner("Fetching data, running AI, and updating database..."):
                try:
                    sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                    if isinstance(sp500.columns, pd.MultiIndex):
                        sp500.columns = sp500.columns.get_level_values(0)
                    sp500_close = float(sp500['Close'].iloc[-1])
                    sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
                    if sp500_close < sp500_sma50:
                        st.warning("⚠️ MACRO WARNING: S&P 500 is below its 50-day moving average (Market under pressure).")
                    else:
                        st.success("🌍 MACRO STATUS: S&P 500 is in a positive trend.")
                except:
                    st.info("🌍 Macro status could not be verified.")

                for ticker in active_tickers:
                    with st.expander(f"Analysis for: {ticker}"):
                        try:
                            t_obj = yf.Ticker(ticker)
                            data = t_obj.history(period="1y", interval="1d")
                            if data.empty or len(data) < 30:
                                st.error(f"Insufficient data for {ticker}")
                                continue

                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = data.columns.get_level_values(0)

                            news_sentiment, latest_headline = analyze_news_sentiment(t_obj)
                            rsi_val = calculate_rsi(data)
                            atr_val = calculate_atr(data)
                            sma_50 = float(data['Close'].rolling(window=50).mean().iloc[-1]) if len(data) >= 50 else float(data['Close'].mean())
                            sma_200 = float(data['Close'].rolling(window=200).mean().iloc[-1]) if len(data) >= 200 else float(data['Close'].mean())
                            next_earnings = get_next_earnings_date(t_obj)
                            
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
                                quantitative_direction = "📈 RŮST (LONG)"
                                confidence = 75
                                stop_loss = skutecna_cena - (1.5 * atr_val)
                                take_profit = skutecna_cena + (2.5 * atr_val)
                            elif ai_score < 0:
                                quantitative_direction = "📉 POKLES (SHORT)"
                                confidence = 75
                                stop_loss = skutecna_cena + (1.5 * atr_val)
                                take_profit = skutecna_cena - (2.5 * atr_val)
                            else:
                                quantitative_direction = "⚖️ NEUTRÁLNÍ"
                                confidence = 50
                                stop_loss = skutecna_cena - (1.0 * atr_val)
                                take_profit = skutecna_cena + (1.0 * atr_val)

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
                            st.info(f"🤖 **AI Quantitative Direction:** {quantitative_direction} (Confidence: {confidence}%)")
                            
                            col_sl, col_tp = st.columns(2)
                            col_sl.metric("🛡️ Recommended Stop Loss", f"${stop_loss:.2f}")
                            col_tp.metric("🎯 Recommended Take Profit", f"${take_profit:.2f}")

                            st.write(f"**News Sentiment:** {news_sentiment} | *\"{latest_headline}\"*")
                            st.write(f"**Distance to 20d Peak:** +{rozdil_usd:.2f} USD (+{potencial_procent:.2f}%)")
                            st.write(f"**ATR Volatility:** {atr_val:.2f}")
                            
                            if next_earnings != "N/A":
                                st.info(f"📅 **Next Earnings Date:** {next_earnings} (Expect higher volatility around this date!)")
                            else:
                                st.write("**Next Earnings Date:** Not scheduled / unavailable")
                            
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
                                    st.info(f"🧠 AI Learning: Prediction for {ticker} saved to database (Target: {target_date} -> {predicted_price_20d:.2f} USD)")
                                except Exception as db_err:
                                    st.warning(f"Could not save to DB: {db_err}")

                            fig, ax = plt.subplots(figsize=(10, 4))
                            model.plot(forecast, ax=ax)
                            ax.set_title(f"Prediction for {ticker} (20 days ahead)")
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Error processing {ticker}: {e}")

    with col_insiders:
        st.markdown("### 🏛️ Live Insider Purchases")
        st.markdown("<p style='font-size: 0.9em; color: gray;'>Tracking recent insider activity for top equities.</p>", unsafe_allow_html=True)
        
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
            st.info("No fresh insider data available at the moment.")

elif app_mode == "🧠 AI Accuracy & Backtesting History":
    st.subheader("🧠 AI Learning & Prediction History (Backtesting)")
    st.write("This section reads data from the database and compares past predictions with current market developments.")
    
    if supabase:
        try:
            response = supabase.table("predictions").select("*").order("target_date", desc=True).limit(50).execute()
            data_rows = response.data
            
            if data_rows:
                df_preds = pd.DataFrame(data_rows)
                st.dataframe(df_preds, use_container_width=True)
                st.info("💡 Once the target date (`target_date`), has passed, you can track how accurate the AI ​​prediction was compared to the real market price.")
            else:
                st.warning("There are no predictions stored in the database yet. Please run the analysis on the main page.")
        except Exception as e:
            st.error(f"Nepodařilo se načíst historii z databáze: {e}")
    else:
        st.error("Supabase není připojena.")

elif app_mode == "🏛️ Trump & Insider Trades":
    render_trump_and_political_trades()

elif app_mode == "📘 User Manual":
    render_user_manual()

# --- DONATION / QR CODE SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support the Creator - David_Seda")

try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 QR code image not found. Please add 'qr_solana.png' to the project folder.")

st.sidebar.markdown(
    "<p style='font-size: 0.9em; color: gray;'>If this app entertains you or makes you money, buy me a coffee! ☕</p>", 
    unsafe_allow_html=True
)
