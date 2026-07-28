import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. INSTALACE A IMPORT ---
!pip install -q yfinance prophet matplotlib schedule

import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
from datetime import datetime
from google.colab import drive
import os
import schedule
import time
import IPython
import datetime as dt

# --- 2. KONFIGURACE ---
drive.mount('/content/drive')
LOG_FILE = "/content/drive/MyDrive/finance_log.csv"
TICKERS = ["GLD", "BTC-USD", "VT", "MSFT", "META", "GOOGL", "^GSPC", "BRK-B", "CSPX.L", "ASML", "TSM"]
PRED_DAYS = 20

# NASTAVENÍ E-MAILU
SENDER_EMAIL = "dankostelecky2@gmail.com"
APP_PASSWORD = "tqkavizhnvjfwatd"
RECEIVER_EMAIL = "dankostelecky2@gmail.com"

def poslat_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        print("✅ E-mail s chytrou analýzou byl odeslán.")
    except Exception as e:
        print(f"❌ Chyba při odesílání e-mailu: {e}")

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

# --- 3. CHYTŘEJŠÍ ANALÝZA SENTIMENTU A RIZIK ---
def analyze_news_sentiment(ticker_obj):
    try:
        news = getattr(ticker_obj, 'news', None)
        if not news:
            return "➖ (Žádné aktuální zprávy)", "Žádné dostupné titulky"

        bearish_keywords = [
            "sue", "lawsuit", "fine", "penalty", "antitrust", "monopoly", "probe", "investigation",
            "drop", "plunge", "decline", "slump", "cut", "warning", "risk", "fear", "loss", "crash", "fall", "selloff"
        ]
        bullish_keywords = [
            "surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "high", "gain", "buy", "upgrade", "success", "expand", "dividend"
        ]

        score = 0
        latest_headline = "Neznámý titulek"
        found_regulatory_risk = False

        for item in news[:5]:
            if isinstance(item, dict):
                title = item.get('title', '')
            else:
                title = getattr(item, 'title', '')

            if latest_headline == "Neznámý titulek" and title:
                latest_headline = title

            title_lower = title.lower()

            if any(term in title_lower for term in ["antitrust", "monopoly", "sue", "lawsuit", "fine", "penalty", "probe"]):
                found_regulatory_risk = True

            for kw in bullish_keywords:
                if kw in title_lower: score += 1
            for kw in bearish_keywords:
                if kw in title_lower: score -= 1

        if found_regulatory_risk:
            return "⚖️📉 REGULATORNÍ TLAK / ANTITRUST", latest_headline
        elif score > 0:
            return "📈 BÝČÍ (Růstové zprávy)", latest_headline
        elif score < 0:
            return "📉 MEDVĚDÍ (Poklesové zprávy)", latest_headline
        else:
            return "➖ NEUTRÁLNÍ (Možný skrytý tlak)", latest_headline
    except Exception:
        return "➖ (Zprávy nedostupné)", "Chyba při načítání zpráv"

# --- 4. HLAVNÍ LOGIKA ---
def run_analysis():
    print(f"\n--- Spouštím chytrou analýzu: {datetime.now().strftime('%d.%m.%Y %H:%M')} ---")

    macro_warning = ""
    try:
        sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
        if not sp500.empty:
            sp500_close = float(sp500['Close'].iloc[-1])
            sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
            if sp500_close < sp500_sma50:
                macro_warning = "⚠️ MAKRO VAROVÁNÍ: S&P 500 je pod 50denním průměrem (Trh pod tlakem)!\n"
            else:
                macro_warning = "🌍 MAKRO STAV: S&P 500 je v pozitivním trendu.\n"
    except Exception:
        macro_warning = "🌍 MAKRO STAV: Nelze ověřit.\n"

    vysledky_email = "=" * 65 + "\n"
    vysledky_email += " 🤖 CHYTRÁ INVESTIČNÍ ANALÝZA (ZPRÁVY + DAV + VÝNOS/1USD) \n"
    vysledky_email += "=" * 65 + f"\nČas: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    vysledky_email += macro_warning + "-" * 65 + "\n\n"

    log_data = []

    for ticker in TICKERS:
        print(f"Analyzuji: {ticker}...")
        try:
            t_obj = yf.Ticker(ticker)
            data = t_obj.history(period="1y", interval="1d")
            if data.empty or len(data) < 30:
                continue

            news_sentiment, latest_headline = analyze_news_sentiment(t_obj)

            rsi_val = calculate_rsi(data)
            atr_val = calculate_atr(data)
            sma_200 = float(data['Close'].rolling(window=200).mean().iloc[-1]) if len(data) >= 200 else float(data['Close'].mean())

            skutecna_cena = float(data['Close'].iloc[-1])
            predchozi_cena = float(data['Close'].iloc[-2])

            current_volume = float(data['Volume'].iloc[-1])
            avg_volume_30d = float(data['Volume'].rolling(window=30).mean().iloc[-1])

            dav_nakupuje = current_volume > (avg_volume_30d * 2.0) and skutecna_cena > predchozi_cena
            dav_panikari = current_volume > (avg_volume_30d * 2.0) and skutecna_cena < predchozi_cena

            rsi_se_zotavuje = (rsi_val >= 30) and (rsi_val <= 45)
            objemy_klesaji = current_volume < avg_volume_30d
            min_prvni_polovina = float(data['Low'].iloc[-10:-5].min())
            min_druha_polovina = float(data['Low'].iloc[-5:].min())
            tvori_vyrazne_dno = min_druha_polovina >= min_prvni_polovina

            dav_se_uklidnuje = rsi_se_zotavuje and objemy_klesaji and tvori_vyrazne_dno

            vrchol_20d = float(data['Close'].rolling(window=20).max().iloc[-1])
            rozdil_usd = vrchol_20d - skutecna_cena
            potencial_procent = (rozdil_usd / skutecna_cena) * 100

            # --- NOVÝ VÝPOČET: Zisk na 1 investovaný dolar ---
            zisk_na_1_usd = rozdil_usd / skutecna_cena if skutecna_cena > 0 else 0

            dynamic_threshold = (atr_val / skutecna_cena) * 2.0

            df = data.reset_index()[['Date', 'Close']]
            df.columns = ['ds', 'y']
            df['ds'] = df['ds'].dt.tz_localize(None)

            model = Prophet(daily_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.3)
            model.fit(df)
            future = model.make_future_dataframe(periods=PRED_DAYS)
            forecast = model.predict(future)

            plt.figure(figsize=(10, 4))
            model.plot(forecast, ax=plt.gca())
            plt.title(f"{ticker} | Cena: {round(skutecna_cena, 2)} | RSI: {round(rsi_val, 2)} | ATR: {round(atr_val, 2)}")
            plt.show()

            zitra_predikce = round(float(forecast.iloc[-PRED_DAYS]['yhat']), 2)
            odchylka = (skutecna_cena - zitra_predikce) / zitra_predikce

            is_bullish_trend = skutecna_cena > sma_200

            if dav_se_uklidnuje:
                verdict = "🟢 DAV SE UKLIDŇUJE (IDEÁLNÍ VSTUP - DCA)"
            elif rsi_val < 30 and dav_panikari and "REGULATORNÍ" in news_sentiment:
                verdict = "🔴 PANICKÝ VÝPRODEJ (ANTITRUST / REGULACE - POZOR)"
            elif rsi_val < 30 and dav_panikari and is_bullish_trend:
                verdict = "🟢 PANICKÝ VÝPRODEJ V BÝČÍM TRENDU (SILNÁ SLEVA)"
            elif rsi_val < 30 and dav_panikari:
                verdict = "🟡 PANICKÝ VÝPRODEJ (POZOR - POD SMA200)"
            elif rsi_val > 70 and dav_nakupuje:
                verdict = "🔴 DAVOVÉ FOMO (BUBLINA)"
            elif rsi_val > 70:
                verdict = "⚠️ PŘEKOUPENO (ZAMKNOUT)"
            elif rsi_val < 30:
                verdict = "🟢 PŘEPRODÁNO (VSTUP)"
            elif odchylka > dynamic_threshold:
                verdict = "📊 NAD TRENDEM (DRŽET)"
            elif odchylka < -dynamic_threshold:
                verdict = "🟢 POD TRENDEM (DYNAMICKÝ KUPUJ)"
            else:
                verdict = "⚪ STABILNÍ"

            vysledky_email += f"🔹 Ticker: {ticker:<8} | RSI: {rsi_val:>5.1f} | ATR: {atr_val:>5.2f}\n"
            vysledky_email += f"   Zprávy  : {news_sentiment}\n"
            vysledky_email += f"   Titulek : \"{str(latest_headline)[:55]}...\"\n"
            vysledky_email += f"   Verdikt : {verdict}\n"
            vysledky_email += f"   Cena    : {skutecna_cena:>8.2f} USD  (Trend SMA200: {'✅ OK' if is_bullish_trend else '❌ POD'})\n"
            vysledky_email += f"   Na vrch : +{rozdil_usd:>7.2f} USD  (+{potencial_procent:>5.2f}%)\n"
            vysledky_email += f"   Zisk/1$ : +{zisk_na_1_usd:>6.2f} USD  (Návratnost na 1 investovaný dolar)\n"
            vysledky_email += "-" * 65 + "\n"

            log_data.append({'Date': datetime.now(), 'Ticker': ticker, 'Verdict': verdict, 'Price': skutecna_cena})

        except Exception as e:
            print(f"❌ Chyba při zpracování tickeru {ticker}: {e}")
            continue

    if log_data:
        df_log = pd.DataFrame(log_data)
        if not os.path.isfile(LOG_FILE):
            df_log.to_csv(LOG_FILE, index=False)
        else:
            df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)

    print("\n--- PŘEHLED ---")
    print(vysledky_email)
    poslat_email("Denní investiční analýza (Zprávy + Dav + Výnos)", vysledky_email)

# --- 5. PLÁNOVAČ A ANTI-IDLE ---

js_code = '''
function ClickConnect(){
    console.log("Working");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
'''
display(IPython.display.Javascript(js_code))

posledni_odeslani = None

def job():
    global posledni_odeslani
    dnes = datetime.now().date()

    if datetime.now().weekday() < 5:
        if posledni_odeslani == dnes:
            print("⏳ Dnešní analýza už byla odeslána, přeskakuji.")
            return

        run_analysis()
        posledni_odeslani = dnes
    else:
        print("Víkend - analýza vynechána.")

print("🚀 Spouštím úvodní okamžitou analýzu...")
job()

schedule.clear()
schedule.every().day.at("13:30").do(job)

print(f"✅ Plánovač aktivován. E-mail bude chodit v pracovní dny v 15:30 SEČ.")

while True:
    schedule.run_pending()
    time.sleep(60)
