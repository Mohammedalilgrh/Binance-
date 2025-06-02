import os
import threading
import time
import logging

from flask import Flask, request
import pandas as pd
import numpy as np
import requests

# ===================== CONFIG =====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID", "@yourchannel")

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT", "NEIROUSDT", "FLOKIUSDT",
    "BONKUSDT", "PEPEUSDT", "FLMUSDT", "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT",
    "GMTUSDT", "ADAUSDT", "OSMOUSDT"
]
INTERVAL = '1m'
LOOKBACK = 10000  # Store last 10000 candles

app = Flask(__name__)

# Store last 10,000 candles for each symbol
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
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Telegram send failed: {response.text}")
    except Exception as e:
        logging.warning(f"Telegram send exception: {e}")

def fetch_candles(symbol, limit=LOOKBACK):
    url = "https://binance-bql7.onrender.com/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        klines = response.json()
        if not klines or not isinstance(klines, list):
            logging.warning(f"Malformed klines for {symbol}: {klines}")
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Failed to fetch candles for {symbol}: {e}")
        return pd.DataFrame()

# ===================== SIGNAL DETECTION =====================
def detect_signals(symbol, df):
    signals = []
    if len(df) < 6:
        return []
    last_close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    prev_close = df["close"].iloc[-2]

    if last_close > prev_high:
        signals.append("BOS (Bullish)")
    elif last_close < prev_low:
        signals.append("BOS (Bearish)")

    if prev_close > df["close"].iloc[-3] and last_close < prev_close:
        signals.append("CHoCH (to Bearish)")
    elif prev_close < df["close"].iloc[-3] and last_close > prev_close:
        signals.append("CHoCH (to Bullish)")

    body = abs(df["close"].iloc[-2] - df["open"].iloc[-2])
    candle_range = df["high"].iloc[-2] - df["low"].iloc[-2]
    if candle_range > 0 and body / candle_range > 0.7:
        if df["close"].iloc[-2] > df["open"].iloc[-2]:
            signals.append("OB (Bullish)")
        else:
            signals.append("OB (Bearish)")

    if df["low"].iloc[-1] > df["high"].iloc[-2]:
        signals.append("FVG (Bullish gap)")
    elif df["high"].iloc[-1] < df["low"].iloc[-2]:
        signals.append("FVG (Bearish gap)")

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
                df_new = fetch_candles(symbol)
                if df_new.empty:
                    logging.warning(f"No data for {symbol}")
                    continue

                # Maintain rolling window of 10,000
                df_old = candles_data[symbol]
                if not df_old.empty:
                    # Append new, remove overlaps, keep only last 10000
                    combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["open_time"])
                    combined = combined.tail(LOOKBACK).reset_index(drop=True)
                    candles_data[symbol] = combined
                else:
                    candles_data[symbol] = df_new.tail(LOOKBACK).reset_index(drop=True)

                signals = detect_signals(symbol, candles_data[symbol])
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
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env variable
    app.run(host='0.0.0.0', port=port)
