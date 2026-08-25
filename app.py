import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

def run_ai_quantitative_scanner(ticker_symbol):
    st.subheader(f"🤖 AI / Kvantitativní analýza pro: {ticker_symbol}")
    
    try:
        with st.spinner("Stahuji data a počítám kvantitativní model..."):
            data = yf.download(ticker_symbol, period="6mo", interval="1d", progress=False)
            
        if data is None or data.empty:
            st.warning(f"Pro tiker '{ticker_symbol}' se nepodařilo stáhnout data.")
            return

        # Oprava MultiIndexu, pokud ho yfinance vrátí
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Kontrola, zda máme sloupec 'Close'
        if 'Close' not in data.columns:
            st.error("Stažená data neobsahují sloupec 'Close'.")
            return

        # Výpočty
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14).mean()

        current_price = float(data['Close'].iloc[-1])
        current_rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50.0
        current_atr = float(data['ATR'].iloc[-1]) if not pd.isna(data['ATR'].iloc[-1]) else (current_price * 0.02)
        sma_50 = float(data['SMA_50'].iloc[-1]) if not pd.isna(data['SMA_50'].iloc[-1]) else current_price

        # Logika
        score = 0
        if current_price > sma_50:
            score += 1
        else:
            score -= 1

        if current_rsi < 30:
            score += 1
        elif current_rsi > 70:
            score -= 1

        if score > 0:
            direction = "📈 RŮST (LONG)"
            confidence = 70
            stop_loss = current_price - (1.5 * current_atr)
            take_profit = current_price + (2.5 * current_atr)
        else:
            direction = "📉 POKLES (SHORT)"
            confidence = 70
            stop_loss = current_price + (1.5 * current_atr)
            take_profit = current_price - (2.5 * current_atr)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Aktuální cena", value=f"${current_price:.2f}")
            st.markdown(f"### Doporučený směr: **{direction}**")
            st.progress(confidence / 100, text=f"AI Síla signálu: {confidence}%")

        with col2:
            st.metric(label="Doporučený Stop Loss", value=f"${stop_loss:.2f}")
            st.metric(label="Doporučený Take Profit", value=f"${take_profit:.2f}")

    except Exception as e:
        st.error(f"Chyba při zpracování dat: {e}")

# Spuštění funkce (pokud máte v app.py nadefinovaný tiker, např. 'SPY'):
# run_ai_quantitative_scanner("SPY")
