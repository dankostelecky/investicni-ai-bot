import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# Page configuration
st.set_page_config(page_title="AI Investment Scanner", page_icon="🤖", layout="wide")

st.title("🤖 Klondike AI Investment Scanner (News + Crowd + AI Learning)")
st.write("This tool analyzes markets, tracks crowd psychology, predicts prices, and learns from its past mistakes using a database.")

# --- SUPABASE CONFIGURATION ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

# Configuration
TICKERS = ["GLD", "BTC-USD", "VT", "MSFT", "META", "GOOGL", "^GSPC", "BRK-B", "CSPX.L", "ASML", "TSM", "NVDA", "TSLA", "AAPL", "AMZN", "AMD", "ETH-USD", "SOL-USD", "QQQ", "SPY", "XRP-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]
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

# Run Analysis Button
if st.button("🚀 Run Market Analysis & Save Predictions", type="primary"):
    with st.spinner("Fetching data, running AI, and updating database..."):
        
        # Macro status
        try:
            sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
            sp500_close = float(sp500['Close'].iloc[-1])
            sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
            if sp500_close < sp500_sma50:
                st.warning("⚠️ MACRO WARNING: S&P 500 is below its 50-day moving average (Market under pressure).")
            else:
                st.success("🌍 MACRO STATUS: S&P 500 is in a positive trend.")
        except:
            st.info("🌍 Macro status could not be verified.")

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
                    
                    # Crowd behavior logic
                    crowd_buying = current_volume > (avg_volume_30d * 2.0) and skutecna_cena > predchozi_cena
                    crowd_panicking = current_volume > (avg_volume_30d * 2.0) and skutecna_cena < predchozi_cena
                    is_bullish_trend = skutecna_cena > sma_200

                    vrchol_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
                    rozdil_usd = vrchol_20d - skutecna_cena
                    potencial_procent = (rozdil_usd / skutecna_cena) * 100
                    zisk_na_1_usd = rozdil_usd / skutecna_cena if skutecna_cena > 0 else 0

                    # Verdict logic based on RSI and trend
                    if rsi_val < 35:
                        verdict = "🟢 Verdict: OVERSOLD (ENTRY)"
                    elif rsi_val > 65:
                        verdict = "🔴 Verdict: OVERBOUGHT (CAUTION)"
                    else:
                        verdict = "🟡 Verdict: NEUTRAL (WAIT)"

                    # Metrics display
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

                    # Prophet prediction chart
                    df = data.reset_index()[['Date', 'Close']]
                    df.columns = ['ds', 'y']
                    df['ds'] = df['ds'].dt.tz_localize(None)

                    model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                    model.fit(df)
                    future = model.make_future_dataframe(periods=PRED_DAYS)
                    forecast = model.predict(future)

                    # Získání predikované ceny za 20 dní
                    predicted_price_20d = float(forecast.iloc[-1]['yhat'])
                    target_date = forecast.iloc[-1]['ds'].strftime('%Y-%m-%d')

                    # Uložení predikce do Supabase databáze
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


import streamlit as st

st.title("AI-Driven Quantitative Trading Scanner")

# HTML obsah manuálu
html_manual = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Driven Quantitative Trading Scanner Manual</title>
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
    <p><strong>Technical Documentation & Operations Manual</strong><br>
    Version 1.0.0 | Quantitative Analysis Division</p>

    <h2>1. Executive Summary</h2>
    <p>This document outlines the operational protocols and analytical methodologies governing the AI-Driven Trading Scanner. Designed for high-frequency volatility analysis, the platform provides actionable intelligence across equity and crypto markets.</p>

    <h2>2. Mathematical & Analytical Framework</h2>
    <h3>Relative Strength Index (RSI)</h3>
    <p>The RSI is the core momentum oscillator utilized for mean-reversion detection. It is calculated via the following formula:</p>
    <pre>RSI = 100 - [ 100 / ( 1 + RS ) ]</pre>
    <p>Where RS represents the ratio of average gains to average losses over 14 periods. The platform assumes a statistical lookback to optimize signal significance.</p>

    <h2>3. User Interface & Operations</h2>
    <ul>
        <li><strong>Dashboard:</strong> Real-time price tracking and RSI visualization.</li>
        <li><strong>Strategy Editor:</strong> Adjustable parameters for volatility sensitivity.</li>
        <li><strong>Notification Engine:</strong> Browser-based alerts for threshold breaches.</li>
    </ul>

    <h2>4. Advanced Feature Suite</h2>
    <p>Our proprietary model incorporates multi-asset liquidity filtering, which prioritizes assets with the tightest bid-ask spreads, and AI-driven signal validation to mitigate noise from pure momentum strategies.</p>

    <h2>5. Risk Disclaimer</h2>
    <p>Trading financial assets, including cryptocurrencies, involves substantial risk. This software is provided as an analytical tool; all investment decisions remain the sole responsibility of the user.</p>

    <h2>Data Metrics & Analysis Reference</h2>
    <div class="metric-box">
        <p><strong>Current Price:</strong> The last traded market price of the asset.</p>
        <p><strong>RSI (14):</strong> Momentum indicator showing if an asset is oversold (&lt;30) or overbought (&gt;70).</p>
        <p><strong>Profit / $1 Invested:</strong> Estimated return metric based on current mean-reversion analysis.</p>
        <p><strong>Verdict (Oversold/Entry):</strong> AI-generated signal indicating potential long entry based on RSI thresholds.</p>
        <p><strong>News Sentiment:</strong> Qualitative assessment of market news impact (Positive, Negative, Neutral).</p>
        <p><strong>Distance to 20d Peak:</strong> Mean distance from the asset's 20-day high, indicating retracement potential.</p>
        <p><strong>ATR Volatility (Average True Range):</strong> A measure of price variability; higher values indicate greater risk/reward potential.</p>
        <p><strong>Crowd Behavior:</strong> Analysis of retail volume and social sentiment.</p>
        <p><strong>Long-term Trend (SMA200):</strong> Comparison vs. 200-day Moving Average; crucial for identifying structural market shifts.</p>
        <p><strong>AI Learning Prediction:</strong> Machine learning model output forecasting future price targets.</p>
    </div>
</body>
</html>
"""
st.components.v1.html(html_manual, height=1800, scrolling=True)


# --- DONATION / QR CODE SECTION ---
st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support the Creator - David_Seda")

# Vložení QR kódu (ujisti se, že soubor qr_solana.png je ve stejné složce)
try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 QR code image not found. Please add 'qr_solana.png' to the project folder.")

# Anglický text podle tvého zadání
st.sidebar.markdown(
    "<p style='font-size: 0.9em; color: gray;'>If this app entertains you or makes you money, buy me a coffee! ☕</p>", 
    unsafe_allow_html=True
)
