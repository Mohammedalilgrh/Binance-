import os
import threading
import time
import logging

from flask import Flask, request
import pandas as pd
import numpy as np
import requests

# ===================== CONFIG =====================
API_KEY = 'cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct'
API_SECRET = 'GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC'

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '@yourchannel')

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT", "NEIROUSDT", "FLOKIUSDT",
    "BONKUSDT", "PEPEUSDT", "FLMUSDT", "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT",
    "GMTUSDT", "ADAUSDT", "OSMOUSDT"
]
INTERVAL = '1m'
LOOKBACK = 5000   # Now fetch 5000 candles per symbol

BINANCE_API_BASE = 'https://binance-bql7.onrender.com'

app = Flask(__name__)

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
    url = f'{BINANCE_API_BASE}/klines'
    params = {
        'symbol': symbol,
        'interval': INTERVAL,
        'limit': limit
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        klines = response.json()
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Failed to fetch candles for {symbol}: {e}")
        return pd.DataFrame()

# ===================== SIGNAL DETECTION =====================
def detect_signals(symbol, df):
    signals = []
    if len(df) < 10:
        return []

    # BOS & CHoCH (very basic: new high or low)
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
                if df.empty:
                    continue

                # Concatenate with previous memory (keep rolling window of 5000)
                if not candles_data[symbol].empty:
                    df = pd.concat([candles_data[symbol], df]).drop_duplicates(subset=['open_time'], keep='last')
                    df = df.tail(LOOKBACK)  # Keep only last 5000
                candles_data[symbol] = df

                signals = detect_signals(symbol, df)
                if signals:
                    message = f"*{symbol}*\n" + "\n".join([f"- {s}" for s in signals])
                    send_telegram_alert(message)
                    logging.info(f"Alert sent for {symbol}: {signals}")
            time.sleep(60)
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
    return {"status": "ok"}

# ===================== LAUNCH =====================
if __name__ == '__main__':
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
