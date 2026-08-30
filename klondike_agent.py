import asyncio
import sys

# Zajištění běžící smyčky pro eventkit a ib_insync v Pythonu 3.14
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

if loop.is_closed():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import pandas as pd
import math
from ib_insync import IB, Stock, MarketOrder, Order

class KlondikeExecutionAgent:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        if not self.ib.isConnected():
            try:
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                print("[INFO] Úspěšně připojeno k Interactive Brokers přes API.")
            except Exception as e:
                print(f"[CHYBA] Nepodařilo se připojit k TWS/Gateway: {e}")
                raise

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            print("[INFO] Spojení s brokerem bylo ukončeno.")

    def process_scanner_results(self, scan_results_df: pd.DataFrame):
        qualified_assets = scan_results_df[
            (scan_results_df["zisk_na_1_usd"] >= 0.08) & 
            (scan_results_df["ai_confidence_score"] >= 75.0)
        ]

        if qualified_assets.empty:
            print("[INFO] Žádná aktivní akcie nesplňuje požadovaná kritéria.")
            return

        print(f"Nalezeno {len(qualified_assets)} vhodných aktiv k obchodování. Spouštím exekuci...")
        self.connect()

        try:
            for _, row in qualified_assets.iterrows():
                ticker = row["ticker"]
                atr_value = row["atr_14"]
                target_quantity = row.get("suggested_shares", 5)

                print(f"\n--- Zpracovávám: {ticker} ---")
                contract = Stock(ticker, 'SMART', 'USD')
                qualified = self.ib.qualifyContracts(contract)
                if not qualified:
                    continue

                [ticker_data] = self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(2)
                current_price = ticker_data.marketPrice()
                
                if not current_price or math.isnan(current_price):
                    current_price = ticker_data.close

                buy_order = MarketOrder('BUY', target_quantity)
                buy_trade = self.ib.placeOrder(contract, buy_order)
                
                while not buy_trade.isDone():
                    self.ib.sleep(0.5)

                if buy_trade.orderStatus.status != 'Filled':
                    continue

                executed_price = buy_trade.orderStatus.avgFillPrice
                
                trail_amount = round(max(atr_value * 1.5, 0.50), 2)
                trailing_stop_order = Order()
                trailing_stop_order.action = 'SELL'
                trailing_stop_order.orderType = 'TRAIL'
                trailing_stop_order.totalQuantity = target_quantity
                trailing_stop_order.auxPrice = trail_amount

                self.ib.placeOrder(contract, trailing_stop_order)
                print(f"[OCHRANA] Trailing Stop aktivován pro {ticker} s odchylkou {trail_amount} USD.")

        finally:
            self.disconnect()
