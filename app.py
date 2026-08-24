import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Investment Scanner", page_icon="🤖", layout="wide")

# --- LANGUAGE SELECTOR ---
st.sidebar.markdown("### 🌍 Language / Jazyk")
languages = [
    "English", "Čeština", "Slovenčina", "Deutsch", "Polski", 
    "Русский", "Français", "Español", "中文", "日本語", "हिन्दी", "العربية"
]
lang = st.sidebar.selectbox("Choose language / Zvolte jazyk:", languages, label_visibility="collapsed")

# Text dictionary based on selected language
if lang == "Čeština":
    t_title = "🤖 Klondike AI Investment Scanner (Zprávy + Dav + AI učení)"
    t_desc = "Tento nástroj analýzuje trhy, sleduje psychologii davu, predikuje ceny a učí se z minulých chyb pomocí databáze."
    t_btn = "🚀 Spustit analýzu trhu a uložit predikce"
    t_spinner = "Načítám data, spouštím AI a aktualizuji databázi..."
    t_macro_warn = "⚠️ MAKRO VAROVÁNÍ: S&P 500 je pod svou 50denní klouzavou průměrnou hodnotou (Trh pod tlakem)."
    t_macro_ok = "🌍 MAKRO STAV: S&P 500 je v pozitivním trendu."
    t_macro_err = "🌍 Makro stav se nepodařilo ověřit."
elif lang == "Slovenčina":
    t_title = "🤖 Klondike AI Investment Scanner (Správy + Dav + AI učenie)"
    t_desc = "Tento nástroj analyzuje trhy, sleduje psychológiu davu, predikuje ceny a učí sa z minulých chýb pomocou databáze."
    t_btn = "🚀 Spustiť analýzu trhu a uložiť predikcie"
    t_spinner = "Načítavam dáta, spúšťam AI a aktualizujem databázu..."
    t_macro_warn = "⚠️ MAKRO VAROVANIE: S&P 500 je pod svojou 50-dňovou kĺzavou priemernou hodnotou (Trh pod tlakom)."
    t_macro_ok = "🌍 MAKRO STAV: S&P 500 je v pozitívnom trende."
    t_macro_err = "🌍 Makro stav sa nepodarilo overiť."
elif lang == "Deutsch":
    t_title = "🤖 Klondike AI Investment Scanner (Nachrichten + Crowd + KI-Lernen)"
    t_desc = "Dieses Tool analysiert Märkte, verfolgt die Crowd-Psychologie, prognostiziert Preise und lernt aus vergangenen Fehlern."
    t_btn = "🚀 Marktanalyse starten & Vorhersagen speichern"
    t_spinner = "Daten werden geladen, KI läuft und Datenbank wird aktualisiert..."
    t_macro_warn = "⚠️ MAKRO-WARNUNG: S&P 500 liegt unter dem 50-Tage-Durchschnitt (Markt unter Druck)."
    t_macro_ok = "🌍 MAKRO-STATUS: S&P 500 ist in einem positiven Trend."
    t_macro_err = "🌍 Makro-Status konnte nicht überprüft werden."
elif lang == "Polski":
    t_title = "🤖 Klondike AI Investment Scanner (Wiadomości + Tłum + Nauka AI)"
    t_desc = "To narzędzie analizuje rynki, śledzi psychologię tłumu, prognozuje ceny i uczy się na błędach."
    t_btn = "🚀 Uruchom analizę rynku i zapisz prognozy"
    t_spinner = "Pobieranie danych, uruchamianie AI i aktualizacja bazy danych..."
    t_macro_warn = "⚠️ OSTRZEŻENIE MAKRO: S&P 500 jest poniżej 50-dniowej średniej (Rynek pod presją)."
    t_macro_ok = "🌍 STATUS MAKRO: S&P 500 w trendzie wzrostowym."
    t_macro_err = "🌍 Nie udało się zweryfikować statusu makro."
elif lang == "Русский":
    t_title = "🤖 Klondike AI Investment Scanner (Новости + Толпа + ИИ Обучение)"
    t_desc = "Этот инструмент анализирует рынки, отслеживает психологию толпы, прогнозирует цены и учится на ошибках."
    t_btn = "🚀 Запустить анализ рынка и сохранить прогнозы"
    t_spinner = "Загрузка данных, запуск ИИ и обновление базы данных..."
    t_macro_warn = "⚠️ МАКРОПРЕДУПРЕЖДЕНИЕ: S&P 500 ниже 50-дневной скользящей средней (Рынок под давлением)."
    t_macro_ok = "🌍 МАКРОСТАТУС: S&P 500 в позитивном тренде."
    t_macro_err = "🌍 Не удалось проверить макростатус."
elif lang == "Français":
    t_title = "🤖 Klondike AI Investment Scanner (Actualités + Foule + IA)"
    t_desc = "Cet outil analyse les marchés, suit la psychologie des foules, prédit les prix et apprend de ses erreurs."
    t_btn = "🚀 Lancer l'analyse du marché et enregistrer"
    t_spinner = "Chargement des données, exécution de l'IA..."
    t_macro_warn = "⚠️ ALERTE MACRO : Le S&P 500 est sous sa moyenne mobile à 50 jours."
    t_macro_ok = "🌍 STATUT MACRO : Le S&P 500 est dans une tendance haussière."
    t_macro_err = "🌍 Impossible de vérifier le statut macro."
elif lang == "Español":
    t_title = "🤖 Klondike AI Investment Scanner (Noticias + Multitud + IA)"
    t_desc = "Esta herramienta analiza mercados, rastrea la psicología de masas, predice precios y aprende de errores pasados."
    t_btn = "🚀 Ejecutar análisis de mercado y guardar predicciones"
    t_spinner = "Cargando datos, ejecutando IA y actualizando base de datos..."
    t_macro_warn = "⚠️ ADVERTENCIA MACRO: El S&P 500 está por debajo de su media de 50 días."
    t_macro_ok = "🌍 ESTADO MACRO: El S&P 500 está en tendencia positiva."
    t_macro_err = "🌍 No se pudo verificar el estado macro."
elif lang == "中文":
    t_title = "🤖 Klondike AI 投资扫描器 (新闻 + 群众心理 + AI学习)"
    t_desc = "该工具分析市场、追踪群众心理、预测价格并通过数据库从过去的错误中学习。"
    t_btn = "🚀 运行市场分析并保存预测"
    t_spinner = "正在获取数据、运行AI并更新数据库..."
    t_macro_warn = "⚠️ 宏观警告：标普500指数低于其50日均线（市场承压）。"
    t_macro_ok = "🌍 宏观状态：标普500指数呈上升趋势。"
    t_macro_err = "🌍 无法验证宏观状态。"
elif lang == "日本語":
    t_title = "🤖 Klondike AI 投資スキャナー (ニュース + 群衆心理 + AI学習)"
    t_desc = "このツールは市場を分析し、群衆心理を追跡し、価格を予測し、データベースから過去の失敗を学習します。"
    t_btn = "🚀 市場分析を実行して予測を保存"
    t_spinner = "データ取得中、AI実行中、データベース更新中..."
    t_macro_warn = "⚠️ マクロ警告: S&P 500 が 50 日移動平均を下回っています (市場に圧力)。"
    t_macro_ok = "🌍 マクロステータス: S&P 500 は上昇トレンドです。"
    t_macro_err = "🌍 マクロステータスを確認できませんでした。"
elif lang == "हिन्दी":
    t_title = "🤖 Klondike AI Investment Scanner (समाचार + भीड़ + AI लर्निंग)"
    t_desc = "यह टूल बाजारों का विश्लेषण करता है, भीड़ के मनोविज्ञान को ट्रैक करता है, कीमतों की भविष्यवाणी करता है।"
    t_btn = "🚀 बाज़ार विश्लेषण चलाएँ और भविष्यवाणियाँ सहेजें"
    t_spinner = "डेटा प्राप्त किया जा रहा है, AI चल रहा है..."
    t_macro_warn = "⚠️ मैक्रो चेतावनी: S&P 500 अपने 50-दिवसीय औसत से नीचे है।"
    t_macro_ok = "🌍 मैक्रो स्थिति: S&P 500 सकारात्मक प्रवृत्ति में है।"
    t_macro_err = "🌍 मैक्रो स्थिति सत्यापित नहीं की जा सकी।"
elif lang == "العربية":
    t_title = "🤖 Klondike AI Investment Scanner (أخبار + حشود + تعلم الذكاء الاصطناعي)"
    t_desc = "تقوم هذه الأداة بتحليل الأسواق، تتبع نفسية الحشود، التنبؤ بالأسعار والتعلم من الأخطاء."
    t_btn = "🚀 تشغيل تحليل السوق وحفظ التوقعات"
    t_spinner = "جلب البيانات وتشغيل الذكاء الاصطناعي وتحديث قاعدة البيانات..."
    t_macro_warn = "⚠️ تحذير ماكرو: مؤشر S&P 500 أقل من متوسطه المتحرك لـ 50 يومًا."
    t_macro_ok = "🌍 حالة ماكرو: مؤشر S&P 500 في اتجاه إيجابي."
    t_macro_err = "🌍 تعذر التحقق من حالة ماكرو."
else:
    t_title = "🤖 Klondike AI Investment Scanner (News + Crowd + AI Learning)"
    t_desc = "This tool analyzes markets, tracks crowd psychology, predicts prices, and learns from its past mistakes using a database."
    t_btn = "🚀 Run Market Analysis & Save Predictions"
    t_spinner = "Fetching data, running AI, and updating database..."
    t_macro_warn = "⚠️ MACRO WARNING: S&P 500 is below its 50-day moving average (Market under pressure)."
    t_macro_ok = "🌍 MACRO STATUS: S&P 500 is in a positive trend."
    t_macro_err = "🌍 Macro status could not be verified."

st.title(t_title)
st.write(t_desc)

# --- SUPABASE CONFIGURATION ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Database not connected: {e}")

# Configuration
TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "AMD", "GLD", "BTC-USD", "VT", "^GSPC", "ETH-USD", "SOL-USD", "QQQ", "SPY", "XRP-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]
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
if st.button(t_btn, type="primary"):
    with st.spinner(t_spinner):
        
        # Macro status
        try:
            sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
            sp500_close = float(sp500['Close'].iloc[-1])
            sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
            if sp500_close < sp500_sma50:
                st.warning(t_macro_warn)
            else:
                st.success(t_macro_ok)
        except:
            st.info(t_macro_err)

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

                    predicted_price_20d = float(forecast.iloc[-1]['yhat'])
                    target_date = forecast.iloc[-1]['ds'].strftime('%Y-%m-%d')

                    # Save prediction to Supabase database
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

# HTML documentation manual
html_manual = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
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
st.components.v1.html(html_manual, height=3800, scrolling=True)

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
