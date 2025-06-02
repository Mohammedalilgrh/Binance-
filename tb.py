import os
import threading
import time
import logging

from flask import Flask, request
import pandas as pd
import numpy as np
from binance.client import Client
from binance import ThreadedWebsocketManager
import requests

# ===================== CONFIG =====================
API_KEY = 'cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct'
API_SECRET = 'GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC'

TELEGRAM_BOT_TOKEN = '7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg'
TELEGRAM_CHANNEL_ID = '@tradegrh'

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT", "NEIROUSDT", "FLOKIUSDT",
    "BONKUSDT", "PEPEUSDT", "FLMUSDT", "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT",
    "GMTUSDT", "ADAUSDT", "OSMOUSDT"
]
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK = 100  # Number of candles to keep in memory

app = Flask(__name__)
client = Client(API_KEY, API_SECRET)

# For each symbol, we'll store recent candles
candles_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}

logging.basicConfig(level=logging.INFO)

# ===================== HELPER FUNCTIONS =====================
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            logging.warning(f"Telegram send failed: {response.text}")
    except Exception as e:
        logging.warning(f"Telegram send exception: {e}")

def fetch_candles(symbol, limit=LOOKBACK):
    klines = client.get_klines(symbol=symbol, interval=INTERVAL, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
    return df

# ===================== SIGNAL DETECTION =====================
def detect_signals(symbol, df):
    """Detect BOS, CHoCH, OB, FVG, 161.8 Fibo. Returns a list of signals."""
    signals = []

    # BOS & CHoCH (very basic: new high or low)
    if len(df) < 3:
        return []
    last_close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    prev_close = df["close"].iloc[-2]

    if last_close > prev_high:
        signals.append("BOS (Bullish)")
    elif last_close < prev_low:
        signals.append("BOS (Bearish)")
    # CHoCH: price reverses direction
    if prev_close > df["close"].iloc[-3] and last_close < prev_close:
        signals.append("CHoCH (to Bearish)")
    elif prev_close < df["close"].iloc[-3] and last_close > prev_close:
        signals.append("CHoCH (to Bullish)")

    # OB (Order Block) detection -- simplistic: strong up/down candle
    body = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
    candle_range = df["high"].iloc[-2] - df["low"].iloc[-2]
    if candle_range > 0 and body / candle_range > 0.7:
        if df["close"].iloc[-2] > df["open"].iloc[-2]:
            signals.append("OB (Bullish)")
        else:
            signals.append("OB (Bearish)")

    # FVG (Fair Value Gap): gap between candles
    if df["low"].iloc[-1] > df["high"].iloc[-2]:
        signals.append("FVG (Bullish gap)")
    elif df["high"].iloc[-1] < df["low"].iloc[-2]:
        signals.append("FVG (Bearish gap)")

    # Fibo 161.8 projection
    swing_high = df["high"].iloc[-5:-1].max()
    swing_low = df["low"].iloc[-5:-1].min()
    direction = "Bullish" if last_close > prev_close else "Bearish"
    if direction == "Bullish":
        fib_target = swing_high + (swing_high - swing_low) * 0.618
        if last_close >= fib_target:
            signals.append("Fibo 161.8% reached (Bullish)")
    else:
        fib_target = swing_low - (swing_high - swing_low) * 0.618
        if last_close <= fib_target:
            signals.append("Fibo 161.8% reached (Bearish)")

    return signals

# ===================== MAIN WORKER =====================
def worker():
    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch_candles(symbol)
                candles_data[symbol] = df
                signals = detect_signals(symbol, df)
                if signals:
                    message = f"*{symbol}*\n" + "\n".join([f"- {s}" for s in signals])
                    send_telegram_alert(message)
                    logging.info(f"Alert sent for {symbol}: {signals}")
            time.sleep(60)  # Run every minute
        except Exception as e:
            logging.error(f"Worker error: {e}")
            time.sleep(10)

# ===================== FLASK APP =====================
@app.route('/')
def index():
    return "Crypto Signal Bot is running."

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    # You can implement webhook logic here if needed (e.g., for Render)
    return {"status": "ok"}

# ===================== LAUNCH =====================
if __name__ == '__main__':
    # Start the worker in a separate thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    # Run Flask app
    app.run(host='0.0.0.0', port=5000)
