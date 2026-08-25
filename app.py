import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def run_ai_quantitative_scanner(ticker_symbol):
    st.subheader(f"🤖 AI / Kvantitativní analýza pro: {ticker_symbol}")
    
    with st.spinner("Stahuji data a počítám kvantitativní model..."):
        # Stažení historických dat (např. za posledních 6 měsíců)
        data = yf.download(ticker_symbol, period="6mo", interval="1d", progress=False)
        
    if data.empty:
        st.error("Nepodařilo se stáhnout data pro vybraný tiker.")
        return

    # Oprava MultiIndexu, pokud ho yfinance vrátí
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 1. Výpočet technických indikátorů (Kvantitativní základy)
    # Klouzavé průměry pro trend
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # RSI (Relative Strength Index) pro momentum
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range) pro měření volatility a výpočet Stop Lossu
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    # Aktuální hodnoty
    current_price = float(data['Close'].iloc[-1])
    current_rsi = float(data['RSI'].iloc[-1])
    current_atr = float(data['ATR'].iloc[-1])
    sma_50 = float(data['SMA_50'].iloc[-1])
    sma_200 = float(data['SMA_200'].iloc[-1])

    # 2. AI / Kvantitativní logický model rozhodování
    # Skóre trendu a momentum
    score = 0
    reasons = []

    if current_price > sma_50:
        score += 1
        reasons.append("Cena je nad 50denním klouzavým průměrem (Býčí trend).")
    else:
        score -= 1
        reasons.append("Cena je pod 50denním klouzavým průměrem (Medvědí trend).")

    if sma_50 > sma_200:
        score += 1
        reasons.append("Zlatý kříž / Dlhodobý trend je rostoucí (SMA50 > SMA200).")
    else:
        score -= 1
        reasons.append("Dlouhodobý trend je klesající (SMA50 < SMA200).")

    if current_rsi < 30:
        score += 1
        reasons.append(f"RSI je v přeprodané zóně ({current_rsi:.1f}) – možný odraz vzhůru.")
    elif current_rsi > 70:
        score -= 1
        reasons.append(f"RSI je v prekoupěné zóně ({current_rsi:.1f}) – riziko korekce.")
    else:
        reasons.append(f"RSI je v neutrální zóně ({current_rsi:.1f}).")

    # Celkový odhad směru
    if score > 0:
        direction = "📈 RŮST (LONG)"
        confidence = min(int(50 + (score * 20)), 95)
    elif score < 0:
        direction = "📉 POKLES (SHORT)"
        confidence = min(int(50 + (abs(score) * 20)), 95)
    else:
        direction = "⚖️ NEUTRÁLNÍ / BOCOVÝ TRH"
        confidence = 50

    # 3. Kvantitativní výpočet Stop Lossu a Take Profitu pomocí ATR (Volatility-based)
    # Standardně se SL dává na 1.5 až 2násobek ATR a TP na 2 až 3násobek ATR (RRR 1:1.5 až 1:2)
    if "RŮST" in direction:
        stop_loss = current_price - (1.5 * current_atr)
        take_profit = current_price + (2.5 * current_atr)
    else:
        stop_loss = current_price + (1.5 * current_atr)
        take_profit = current_price - (2.5 * current_atr)

    # Zobrazení výsledků v aplikaci
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Aktuální cena", value=f"${current_price:.2f}")
        st.markdown(f"### Doporučený směr: **{direction}**")
        st.progress(confidence / 100, text=f"AI Síla signálu / Důvěra: {confidence}%")

    with col2:
        st.metric(label="Doporučený Stop Loss (ATR model)", value=f"${stop_loss:.2f}")
        st.metric(label="Doporučený Take Profit (Risk/Reward 1:1.6)", value=f"${take_profit:.2f}")

    # Rozbalovací detail odůvodnění AI modelu
    with st.expander("🔍 Zobrazit podrobnosti AI / Kvantitativní analýzy"):
        st.write(f"- **Aktuální ATR (14):** ${current_atr:.2f}")
        for r in reasons:
            st.write(f"- {r}")

# Příklad použití:
# run_ai_quantitative_scanner("SPY")
