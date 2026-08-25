import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# 1. Základní nastavení stránky (musí být jako první příkaz Streamlitu)
st.set_page_config(
    page_title="AI Investment Scanner",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("🤖 AI Investment & Trading Scanner")
    st.write("Kvantitativní analýza trhů pro S&P 500, MSCI Europe a globální ETF.")

    # --- POSTranní PANEL (Sidebar) ---
    st.sidebar.header("⚙️ Nastavení analýzy")

    # Výběr kategorií a tikerů
    tickers_dict = {
        "S&P 500 (US Core)": {
            "SPY (SPDR S&P 500 ETF)": "SPY",
            "VOO (Vanguard S&P 500)": "VOO",
            "CSPX (iShares Core S&P 500 Acc)": "CSPX.L",
            "S&P 500 Index (^GSPC)": "^GSPC"
        },
        "MSCI Europe (European Core)": {
            "IEUR (iShares Core MSCI Europe)": "IEUR",
            "MEUD (Lyxor MSCI Europe UCITS ETF)": "MEUD.PA",
            "EXXT (iShares Core MSCI Europe EUR)": "EXXT.DE"
        },
        "Globální trhy": {
            "VT (Vanguard Total World Stock ETF)": "VT"
        }
    }

    category = st.sidebar.selectbox("Vyberte trh / region:", list(tickers_dict.keys()))
    selected_name = st.sidebar.selectbox("Vyberte instrument:", list(tickers_dict[category].keys()))
    ticker_symbol = tickers_dict[category][selected_name]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Zvolený symbol: **{ticker_symbol}**")

    # --- HLAVNÍ OBSAH APLIKACE ---
    st.subheader(f"📊 Analýza pro: {selected_name} (`{ticker_symbol}`)")

    try:
        with st.spinner("Stahuji historická data a počítám kvantitativní model..."):
            data = yf.download(ticker_symbol, period="6mo", interval="1d", progress=False)

        if data is None or data.empty:
            st.error(f"Nepodařilo se stáhnout data pro symbol {ticker_symbol}. Zkuste vybrat jiný.")
            return

        # Oprava MultiIndexu, pokud ho yfinance vrátí v novějších verzích
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if 'Close' not in data.columns or 'High' not in data.columns or 'Low' not in data.columns:
            st.error("Stažená data nemají správnou strukturu pro výpočet indikátorů.")
            return

        # Výpočet kvantitativních indikátorů
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # RSI výpočet
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # ATR (Average True Range) pro volatilitu
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14).mean()

        # Ošetření hodnot pro aktuální den
        current_price = float(data['Close'].iloc[-1])
        current_rsi = float(data['RSI'].iloc[-1]) if not pd.isna(data['RSI'].iloc[-1]) else 50.0
        current_atr = float(data['ATR'].iloc[-1]) if not pd.isna(data['ATR'].iloc[-1]) else (current_price * 0.02)
        sma_50 = float(data['SMA_50'].iloc[-1]) if not pd.isna(data['SMA_50'].iloc[-1]) else current_price

        # AI / Kvantitativní logika rozhodování
        score = 0
        reasons = []

        if current_price > sma_50:
            score += 1
            reasons.append("Cena je nad 50denním klouzavým průměrem (Rostoucí trend).")
        else:
            score -= 1
            reasons.append("Cena je pod 50denním klouzavým průměrem (Klesající trend).")

        if current_rsi < 30:
            score += 1
            reasons.append(f"RSI je v přeprodané zóně ({current_rsi:.1f}) – potenciál k odrazu.")
        elif current_rsi > 70:
            score -= 1
            reasons.append(f"RSI je v překoupené zóně ({current_rsi:.1f}) – riziko poklesu.")
        else:
            reasons.append(f"RSI se nachází v neutrální zóně ({current_rsi:.1f}).")

        # Určení směru a síly signálu
        if score > 0:
            direction = "📈 RŮST (LONG)"
            confidence = 75
            stop_loss = current_price - (1.5 * current_atr)
            take_profit = current_price + (2.5 * current_atr)
        elif score < 0:
            direction = "📉 POKLES (SHORT)"
            confidence = 75
            stop_loss = current_price + (1.5 * current_atr)
            take_profit = current_price - (2.5 * current_atr)
        else:
            direction = "⚖️ NEUTRÁLNÍ / BOCOVÝ TRH"
            confidence = 50
            stop_loss = current_price - (1.0 * current_atr)
            take_profit = current_price + (1.0 * current_atr)

        # Zobrazení výsledků v přehledném dashboardu
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Aktuální cena", value=f"${current_price:.2f}")
            st.markdown(f"### Doporučený odhad: **{direction}**")
            st.progress(confidence / 100, text=f"Síla signálu: {confidence}%")

        with col2:
            st.metric(label="Doporučený Stop Loss ( ATR model )", value=f"${stop_loss:.2f}")
            st.metric(label="Doporučený Take Profit ( RRR 1:1.6 )", value=f"${take_profit:.2f}")

        # Graf vývoje ceny
        st.markdown("---")
        st.subheader("📈 Vývoj zavírací ceny (6 měsíců)")
        st.line_chart(data['Close'])

        # Rozbalovací detaily analýzy
        with st.expander("🔍 Zobrazit podrobnosti AI / Kvantitativního modelu"):
            st.write(f"- **Vypočítané ATR (14):** ${current_atr:.2f}")
            st.write(f"- **Hodnota RSI (14):** {current_rsi:.2f}")
            st.write(f"- **50denní SMA:** ${sma_50:.2f}")
            st.markdown("**Důvody rozhodnutí modelu:**")
            for r in reasons:
                st.write(f"- {r}")

    except Exception as e:
        st.error(f"Při zpracování dat došlo k chybě: {e}")

if __name__ == "__main__":
    main()
