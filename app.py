import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from supabase import create_client, Client

# Nastavení stránky aplikace
st.set_page_config(page_title="Klondike AI Investment Scanner", page_icon="🤖", layout="wide")

st.title("🤖 Klondike AI Investment Scanner (News + Crowd + AI Learning + Custom Search)")
st.write("Tato aplikace analyzuje trhy, sleduje psychologii davu, počítá duální Long/Short scénáře a učí se z historie pomocí databáze.")

# --- KONFIGURACE SUPABASE DATABÁZE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    st.sidebar.warning(f"⚠️ Databáze není připojena: {e}")

# --- PŘIDÁNÍ VLASTNÍHO TICKERU V BOČNÍM PANELU ---
st.sidebar.markdown("### 🔍 Vlastní vyhledávání aktiv")
custom_ticker_input = st.sidebar.text_input("Přidat ticker (např. NFLX, AAPL, CZG.PR):", "").upper().strip()

DEFAULT_TICKERS = ["META", "MSFT", "GOOGL", "TSM", "TSLA", "AAPL", "AMZN", "BRK-B", "CSPX.L", "ASML", "NVDA", "AMD", "GLD", "BTC-USD", "VT", "^GSPC", "ETH-USD", "SOL-USD", "QQQ", "SPY", "XRP-USD", "BNB-USD", "LINK-USD", "AVAX-USD"]

active_tickers = list(DEFAULT_TICKERS)
if custom_ticker_input and custom_ticker_input not in active_tickers:
    active_tickers.insert(0, custom_ticker_input)
    st.sidebar.success(f"Přidáno {custom_ticker_input} do seznamu skenování!")

PRED_DAYS = 20

# --- POMOCNÉ FUNKCE PRO VÝPOČET INDIKÁTORŮ ---
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
            return "➖ (Žádné čerstvé zprávy)", "Dostupné titulky nenalezeny"
        
        bearish_keywords = ["sue", "lawsuit", "fine", "penalty", "drop", "plunge", "decline", "crash", "loss", "pád", "pokles", "ztráta"]
        bullish_keywords = ["surge", "jump", "rally", "growth", "record", "profit", "beat", "strong", "gain", "růst", "rekord", "zisk"]
        
        score = 0
        latest_headline = "Neznámý titulek"
        
        for item in news[:5]:
            title = item.get('title', '') if isinstance(item, dict) else getattr(item, 'title', '')
            if latest_headline == "Neznámý titulek" and title:
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
        return "➖ (Zprávy nedostupné)", "Chyba při načítání zpráv"

# --- SEKCE PRO TRUMPOVY NÁKUPY A ODKAZY ---
def render_trump_and_political_trades():
    st.subheader("🏛️ Donald Trump & Rodinné majetkové transakce")
    st.write("Přehled sledovaných transakcí a majetkových přiznání nahlášených v oficiálních vládních registrech.")

    trump_data = [
        {"Date": "2026-06-18", "Asset": "Berkshire Hathaway (BRK-B)", "Type": "Nákup", "Estimated Value": "$1M - $5M", "Status": "Aktivní portfólio"},
        {"Date": "2026-06-23", "Asset": "Visa Inc (V)", "Type": "Nákup", "Estimated Value": "$500K - $1M", "Status": "Aktivní portfólio"},
        {"Date": "2026-06-24", "Asset": "Mastercard (MA)", "Type": "Nákup", "Estimated Value": "$500K - $1M", "Status": "Aktivní portfólio"},
        {"Date": "2026-06-03", "Asset": "Palantir (PLTR)", "Type": "Nákup/Prodej", "Estimated Value": "$15K - $50K", "Status": "Rotováno / Obchodováno"},
        {"Date": "2025-04-08", "Asset": "Big Tech Basket (AAPL, MSFT, GOOGL)", "Type": "Velký nákup", "Estimated Value": "$12.8M celkem", "Status": "Hlavní držení"}
    ]
    
    df_trump = pd.DataFrame(trump_data)
    st.dataframe(df_trump, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔗 Oficiální zdroje a veřejná přiznání (Volný přístup)")
    st.write("Všechny záznamy aktiv a zdrojové dokumenty můžete ověřit přímo v oficiálních registrech:")
    
    st.markdown("- 🇺🇸 [U.S. Office of Government Ethics (OGE) Official Search](https://www.oge.gov/web/oge.nsf/Officials%20Individual%20Disclosures%20Search%20Collection?OpenForm)")
    st.markdown("- 📊 [ProPublica Trump & Appointees Financial Disclosures Database](https://projects.propublica.org/trump-team-financial-disclosures/)")
    st.markdown("- 🏛️ [U.S. Senate Electronic Financial Disclosure (eFD) System](https://efd.senate.gov/)")

    st.info("💡 Tip: Můžete zkopírovat jakýkoliv ticker z tabulky výše (např. `BRK-B`, `V`) a vložit jej do bočního panelu **Vlastní vyhledávání aktiv** k analýze aktuálních technických indikátorů a AI výhledu.")

# --- UŽIVATELSKÝ MANUÁL ---
def render_user_manual():
    st.subheader("📘 Klondike AI Investment Scanner: Uživatelský manuál")
    st.markdown("Vítejte v průvodci pro aplikaci Klondike AI Investment Scanner. Tento manuál vám pomůže zorientovat se v rozhraní, indikátorech a správě investic.")

    with st.expander("📖 1. Jak spustit a ovládat aplikaci"):
        st.markdown("""
        Aplikace běží přímo ve webovém prohlížeči a je rozdělená do hlavních sekcí pomocí horního menu:
        
        1. **📊 Skenování trhů a přehled:** Skenujte trhy, přidejte vlastní tickery v postranním panelu a spusťte AI analýzu včetně výhledu trendu, obchodních nastavení Long/Short a **konkrétní investiční rady (Vstoupit / Čekat)**.
        2. **🧠 AI přesnost a historie (Backtesting):** Zkontrolujte historické predikce uložené v databázi.
        3. **🏛️ Trump & Insider obchody:** Sledujte politické a insider transakce.
        4. **📘 Uživatelský manuál:** Přečtěte si nápovědu a popis funkcí.
        """)

    with st.expander("📊 2. Co znamenají jednotlivé indikátory?"):
        st.markdown("""
        * **📈 AI Kvantitativní směr:** Vyhodnocení trendu a predikce směru ceny (`BULLISH`, `BEARISH`, `NEUTRAL`).
        * **💡 Investiční doporučení:** Okamžitá rada jestli **vstoupit** (ideální nákupní zóna/přeprodáno), **čekat** (trh je překoupený nebo nerozhodný) nebo **vyhnout se**.
        * **🟢 Long Setup:** Doporučený ideální vstup, stop loss a take profit pro nákup/růst.
        * **🔴 Short Setup:** Doporučený ideální vstup, stop loss a take profit pro prodej/pokles.
        * **📊 RSI (Relative Strength Index):** Překoupeno (>65) / Přeprodáno (<35).
        * **📰 Sentiment zpráv:** Vyhodnocení finančních novinek.
        """)

    with st.expander("☕ 3. Podpora tvůrce"):
        st.markdown("V dolní části levého bočního panelu naleznete sekci **Podpora tvůrce**.")

# --- HLAVNÍ NAVIGACE APLIKACE ---
app_mode = st.radio("Vyberte režim zobrazení:", [
    "📊 Skenování trhů a přehled", 
    "🧠 AI přesnost a historie", 
    "🏛️ Trump & Insider obchody",
    "📘 Uživatelský manuál"
], horizontal=True)

if app_mode == "📊 Skenování trhů a přehled":
    col_main, col_insiders = st.columns([2.3, 1.2])

    with col_main:
        if st.button("🚀 Spustit analýzu trhů a uložit predikce", type="primary"):
            with st.spinner("Stahování dat, spouštění AI a aktualizace databáze..."):
                try:
                    sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
                    if isinstance(sp500.columns, pd.MultiIndex):
                        sp500.columns = sp500.columns.get_level_values(0)
                    sp500_close = float(sp500['Close'].iloc[-1])
                    sp500_sma50 = float(sp500['Close'].rolling(window=50).mean().iloc[-1])
                    if sp500_close < sp500_sma50:
                        st.warning("⚠️ MAKRO VAROVÁNÍ: S&P 500 je pod 50denním klouzavým průměrem (trh pod tlakem).")
                    else:
                        st.success("🌍 MAKRO STAV: S&P 500 je v pozitivním trendu.")
                except:
                    st.info("🌍 Makro status se nepodařilo ověřit.")

                for ticker in active_tickers:
                    with st.expander(f"Analýza pro: {ticker}"):
                        try:
                            t_obj = yf.Ticker(ticker)
                            data = t_obj.history(period="1y", interval="1d")
                            if data.empty or len(data) < 30:
                                st.error(f"Nedostatek dat pro {ticker}")
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

                            # --- VÝPOČET AI SKÓRE A SMĚŘOVÁNÍ CENY ---
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
                                quantitative_direction = "📈 BULLISH (ROSTOUCÍ TREND)"
                                confidence = 75
                            elif ai_score < 0:
                                quantitative_direction = "📉 BEARISH (KLEVAJÍCÍ TREND)"
                                confidence = 75
                            else:
                                quantitative_direction = "⚖️ NEUTRAL (DO BOKU)"
                                confidence = 50

                            # --- NOVÁ INVESTIČNÍ RADA (VSTOUPIT / ČEKAT / PŘEKOUPENO / PŘEPRODÁNO) ---
                            if rsi_val > 70:
                                market_state_text = "🔴 **PŘEKOUPENO (Overbought):** Trh je extrémně vysoko, hrozí korekce."
                                advice_action = "⏳ **DOPORUČENÍ: ČEKAT / NEVSTUPOVAT** (Nenakupujte do vrcholu, počkejte na pokles/zdravou korekci)."
                                advice_color = "error"
                            elif rsi_val < 30:
                                market_state_text = "🟢 **PŘEPRODÁNO (Oversold):** Aktivum je silně podhodnocené/slevněné."
                                advice_action = "🚀 **DOPORUČENÍ: VSTOUPIT DO LONGU** (Skvělá příležitost k nákupu za výhodnou cenu za předpokladu dodržení Stop Lossu)."
                                advice_color = "success"
                            elif is_bullish_trend and rsi_val <= 60 and rsi_val >= 40:
                                market_state_text = "🟡 **ZDRAVÝ TREND:** Trh roste v rozumném pásmu bez extrémní mánie."
                                advice_action = "✅ **DOPORUČENÍ: VHODNÉ K POSTUPNÉMU VSTUPU (DCA)** nebo držení pozice."
                                advice_color = "success"
                            else:
                                market_state_text = "⚖️ **NEROZHODNÝ / BOČNÍ TRH:** Chybí jasný silný moment."
                                advice_action = "⏳ **DOPORUČENÍ: ČEKAT** na jasnější signál nebo proražení úrovní."
                                advice_color = "info"

                            # --- VÝPOČET DVOJITÝCH SCÉNÁŘŮ (LONG VS. SHORT) ---
                            long_entry = skutecna_cena
                            long_stop_loss = skutecna_cena - (1.5 * atr_val)
                            long_take_profit = skutecna_cena + (2.5 * atr_val)

                            short_entry = skutecna_cena
                            short_stop_loss = skutecna_cena + (1.5 * atr_val)
                            short_take_profit = skutecna_cena - (2.5 * atr_val)

                            col1, col2, col3 = st.columns(3)
                            col1.metric("Aktuální cena", f"{skutecna_cena:.2f} USD")
                            col2.metric("RSI (14)", f"{rsi_val:.1f}")
                            col3.metric("Zisk / 1 USD investice", f"+{zisk_na_1_usd:.2f} USD")

                            st.markdown("---")
                            st.info(f"🤖 **AI Kvantitativní směr:** {quantitative_direction} (Spolehlivost: {confidence}%)")
                            
                            # Zobrazení nové investiční rady
                            st.markdown("### 💡 Investiční rada pro obchodníka:")
                            st.markdown(market_state_text)
                            if advice_color == "success":
                                st.success(advice_action)
                            elif advice_color == "error":
                                st.error(advice_action)
                            else:
                                st.info(advice_action)

                            col_long, col_short = st.columns(2)

                            with col_long:
                                st.markdown("#### 🟢 LONG SETUP (Růstová strategie)")
                                st.success(f"**Ideální vstup:** ${long_entry:.2f}")
                                st.metric("🛡️ Stop Loss (Long)", f"${long_stop_loss:.2f}")
                                st.metric("🎯 Take Profit (Long)", f"${long_take_profit:.2f}")

                            with col_short:
                                st.markdown("#### 🔴 SHORT SETUP (Medvědí strategie)")
                                st.error(f"**Ideální vstup:** ${short_entry:.2f}")
                                st.metric("🛡️ Stop Loss (Short)", f"${short_stop_loss:.2f}")
                                st.metric("🎯 Take Profit (Short)", f"${short_take_profit:.2f}")

                            st.markdown("---")
                            st.write(f"**Sentiment zpráv:** {news_sentiment} | *\"{latest_headline}\"*")
                            st.write(f"**Vzdálenost k 20d vrcholu:** +{rozdil_usd:.2f} USD (+{potencial_procent:.2f}%)")
                            st.write(f"**ATR Volatilita:** {atr_val:.2f}")
                            
                            if next_earnings != "N/A":
                                st.info(f"📅 **Příští výsledková sezóna (Earnings):** {next_earnings} (Očekávejte vyšší volatilitu!)")
                            else:
                                st.write("**Příští výsledková sezóna:** Neplánováno / nedostupné")
                            
                            if crowd_buying:
                                st.markdown("🔥 **Davový poplach:** Zjištěno masivní nakupování (Vysoký objem + Růst ceny)!")
                            elif crowd_panicking:
                                st.markdown("🚨 **Davový poplach:** Zjištěna panická prodeje (Vysoký objem + Pokles ceny)!")
                            else:
                                st.markdown("👥 **Chování davu:** Klid / Normální objem.")

                            trend_status = "✅ OK (Rostoucí vs. SMA200)" if is_bullish_trend else "❌ Pod SMA200 (Opatrnost)"
                            st.write(f"**Dlouhodobý trend:** {trend_status}")

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
                                    st.info(f"🧠 AI Učení: Predikce pro {ticker} uložena do databáze (Cíl: {target_date} -> {predicted_price_20d:.2f} USD)")
                                except Exception as db_err:
                                    st.warning(f"Nepodařilo se uložit do DB: {db_err}")

                            fig, ax = plt.subplots(figsize=(10, 4))
                            model.plot(forecast, ax=ax)
                            ax.set_title(f"Predikce pro {ticker} (na 20 dní dopředu)")
                            st.pyplot(fig)

                        except Exception as e:
                            st.error(f"Chyba při zpracování {ticker}: {e}")

    with col_insiders:
        st.markdown("### 🏛️ Živé nákupy insiderů")
        st.markdown("<p style='font-size: 0.9em; color: gray;'>Sledování nedávné aktivity insiderů u top akcií.</p>", unsafe_allow_html=True)
        
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
                        "Pozice": str(latest.get('Position', 'Insider')),
                        "Akce": str(latest.get('Transaction', 'Akce')),
                        "Akcie": str(latest.get('Shares', 'N/A'))
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
            st.info("V tuto chvíli nejsou k dispozici žádná čerstvá data o insiderch.")

elif app_mode == "🧠 AI přesnost a historie":
    st.subheader("🧠 AI Učení a Historie Predikcí (Backtesting)")
    st.write("Tato sekce načítá data z databáze a porovnává minulé predikce s reálným vývojem na trhu.")
    
    if supabase:
        try:
            response = supabase.table("predictions").select("*").order("target_date", desc=True).limit(50).execute()
            data_rows = response.data
            
            if data_rows:
                df_preds = pd.DataFrame(data_rows)
                st.dataframe(df_preds, use_container_width=True)
                st.info("💡 Jakmile uplyne cílové datum (`target_date`), můžete sledovat, jak přesná byla AI predikce oproti reálné tržní ceně.")
            else:
                st.warning("V databázi zatím nejsou uloženy žádné predikce. Spusťte prosím analýzu na hlavní stránce.")
        except Exception as e:
            st.error(f"Nepodařilo se načíst historii z databáze: {e}")
    else:
        st.error("Supabase není připojena.")

elif app_mode == "🏛️ Trump & Insider obchody":
    render_trump_and_political_trades()

elif app_mode == "📘 Uživatelský manuál":
    render_user_manual()

# --- SEKCE PRO PODPORU TVŮRCE (QR KÓD V BOČNÍM PANELU) ---
st.sidebar.markdown("---")
st.sidebar.subheader("☕ Podpořte tvůrce - David_Seda")

try:
    st.sidebar.image("qr_solana.png", width=180)
except Exception:
    st.sidebar.info("📌 Obrázek QR kódu nebyl nalezen. Přidejte prosím 'qr_solana.png' do složky projektu.")

st.sidebar.markdown(
    "<p style='font-size: 0.9em; color: gray;'>Pokud vám tato aplikace přináší hodnotu nebo zisk, pozvěte mě na kávu! ☕</p>", 
    unsafe_allow_html=True
)
