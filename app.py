import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(page_title="AI Investment Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI Investment Scanner (News + Crowd + ROI)")
st.write("This tool analyzes markets, tracks crowd psychology, news sentiment, and calculates potential returns per invested dollar.")

# Configuration
TICKERS = ["GLD", "BTC-USD", "VT", "MSFT", "META", "GOOGL", "^GSPC", "BRK-B", "CSPX.L", "ASML", "TSM"]
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
if st.button("🚀 Run Market Analysis", type="primary"):
    with st.spinner("Fetching data and running AI analysis..."):
        
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
                    
                    # Crowd psychology status
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

                    fig, ax = plt.subplots(figsize=(10, 4))
                    model.plot(forecast, ax=ax)
                    ax.set_title(f"Prediction for {ticker}")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error processing {ticker}: {e}")

                    # Sekce pro dary v postranním panelu (Sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("☕ Support the Project")
st.sidebar.write("If this AI scanner helps you, buy me a coffee via Solana (Phantom):")

try:
    st.sidebar.image("qr_solana.png", caption="Scan with Phantom (SOL / USDC)", width=180)
    # Zde nahraď text 'TvojeSolanaAdresaZde' svou reálnou adresou peněženky
    st.sidebar.code("TvojeSolanaAdresaZde", language="text")
except Exception:
    st.sidebar.info("Upload 'qr_solana.png' to GitHub to display the QR code.")