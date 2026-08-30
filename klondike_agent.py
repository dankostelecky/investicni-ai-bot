import pandas as pd
from ib_insync import IB, Stock, MarketOrder, Order
import math

class KlondikeExecutionAgent:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Inicializace agenta. Port 7497 je standardně pro Paper Trading (nanečisto).
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        """Připojení k lokální TWS nebo IB Gateway bráně."""
        if not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                print("[INFO] Úspěšně připojeno k Interactive Brokers přes API.")
            except Exception as e:
                print(f"[CHYBA] Nepodařilo se připojit k TWS/Gateway: {e}")
                raise

    def disconnect(self):
        """Ukončení spojení."""
        if self.ib.isConnected():
            self.ib.disconnect()
            print("[INFO] Spojení s brokerem bylo ukončeno.")

    def process_scanner_results(self, scan_results_df: pd.DataFrame):
        """
        Zpracuje DataFrame výsledků ze skeneru, vyfiltruje akcie 
        splňující podmínku zisk >= 0.08 USD a provede obchody.
        """
        # 1. Filtrování podle vašeho klíčového pravidla
        qualified_assets = scan_results_df[
            (scan_results_df["zisk_na_1_usd"] >= 0.08) & 
            (scan_results_df["ai_confidence_score"] >= 75.0)  # Příklad doplňkové AI podmínky
        ]

        if qualified_assets.empty:
            print("[INFO] Žádná aktivní akcie nesplňuje požadovaná kritéria (Gain >= 0.08 USD).")
            return

        print(f"Nalezeno {len(qualified_assets)} vhodných aktiv k obchodování. Spouštím exekuci...")

        self.connect()

        try:
            for _, row in qualified_assets.iterrows():
                ticker = row["ticker"]
                atr_value = row["atr_14"]
                target_quantity = row.get("suggested_shares", 5) # Výchozí množství např. 5 ks

                print(f"\n--- Zpracovávám: {ticker} ---")
                
                # 2. Definice kontraktu
                contract = Stock(ticker, 'SMART', 'USD')
                qualified = self.ib.qualifyContracts(contract)
                if not qualified:
                    print(f"[VAROVÁNÍ] Kontrakt pro {ticker} nebyl burzou ověřen. Přeskakuji.")
                    continue

                # 3. Zjištění aktuální tržní ceny
                [ticker_data] = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(2) # Krátká pauza pro načtení ticku
                current_price = ticker_data.marketPrice()
                
                if not current_price or math.isnan(current_price):
                    current_price = ticker_data.close # Záložní hodnota zavírací ceny
                
                print(f"Aktivum: {ticker} | Aktuální cena: {current_price} USD | ATR: {atr_value}")

                # 4. Odeslání nákupního příkazu (Market Order)
                buy_order = MarketOrder('BUY', target_quantity)
                buy_trade = self.ib.placeOrder(contract, buy_order)
                
                # Čekání na dokončení nákupu
                while not buy_trade.isDone():
                    self.ib.sleep(0.5)

                if buy_trade.orderStatus.status != 'Filled':
                    print(f"[CHYBA] Nákup pro {ticker} nebyl vyplněn. Stav: {buy_trade.orderStatus.status}")
                    continue

                executed_price = buy_trade.orderStatus.avgFillPrice
                print(f"[ÚSPĚCH] Nakoupeno {target_quantity}x {ticker} za průměrnou cenu {executed_price} USD.")

                # 5. Výpočet a nastavení Trailing Stop Sell (posouvá se automaticky NAHORU s růstem ceny)
                # Odchylka (Trail Amount) je odvozena z ATR (např. 1.5násobek ATR, min. 0.50 USD)
                trail_amount = round(max(atr_value * 1.5, 0.50), 2)

                trailing_stop_order = Order()
                trailing_stop_order.action = 'SELL'
                trailing_stop_order.orderType = 'TRAIL'
                trailing_stop_order.totalQuantity = target_quantity
                trailing_stop_order.auxPrice = trail_amount  # Velikost trailing odchylky v dolarech
                trailing_stop_order.outsideRth = False

                ts_trade = self.ib.placeOrder(contract, trailing_stop_order)
                print(f"[OCHRANA] Trailing Stop Sell aktivován pro {ticker} s odchylkou {trail_amount} USD (sleduje cenu směrem nahoru).")

        finally:
            self.disconnect()

# --- Příklad použití s vaším skenerem ---
if __name__ == "__main__":
    # Simulace výstupu z vašeho skeneru Klondike
    mock_scan_data = pd.DataFrame([
        {
            "ticker": "AAPL",
            "zisk_na_1_usd": 0.12,
            "ai_confidence_score": 82.5,
            "atr_14": 2.30,
            "suggested_shares": 2
        },
        {
            "ticker": "TSLA",
            "zisk_na_1_usd": 0.05, # Nesplňuje filtr >= 0.08
            "ai_confidence_score": 60.0,
            "atr_14": 5.10,
            "suggested_shares": 1
        }
    ])

    agent = KlondikeExecutionAgent()
    # Spuštění procesu (vyfiltruje AAPL, TSLA ignoruje)
    agent.process_scanner_results(mock_scan_data)
