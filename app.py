import os
import threading
import time
import logging
from datetime import datetime

from flask import Flask, request
import pandas as pd
import numpy as np
import requests

# ===================== CONFIG =====================
API_KEY = 'cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct'
API_SECRET = 'GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC'

TELEGRAM_BOT_TOKEN = os.environ.get(TELEGRAM_BOT_TOKEN', '7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg')
TELEGRAM_CHANNEL_ID = os.environ.get('TELEGRAM_CHANNEL_ID', '@tradegrh')

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT", "NEIROUSDT", "FLOKIUSDT",
    "BONKUSDT", "PEPEUSDT", "FLMUSDT", "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT",
    "GMTUSDT", "ADAUSDT", "OSMOUSDT"
]
INTERVAL = '1m'
LOOKBACK = 5000
MAX_RETRIES = 3
RETRY_DELAY = 5

# Fallback to official Binance API if proxy fails
BINANCE_API_BASE = 'https://api.binance.com/api/v3'
BINANCE_PROXY = 'https://binance-bql7.onrender.com'

app = Flask(__name__)
candles_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===================== IMPROVED HELPER FUNCTIONS =====================
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            logging.warning(f"Telegram attempt {attempt+1} failed: {response.text}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.warning(f"Telegram attempt {attempt+1} exception: {e}")
            time.sleep(RETRY_DELAY)
    return False

def fetch_candles(symbol, limit=LOOKBACK):
    """Try proxy first, fallback to official API"""
    endpoints = [
        f"{BINANCE_PROXY}/klines",
        f"{BINANCE_API_BASE}/klines"
    ]
    
    params = {
        'symbol': symbol,
        'interval': INTERVAL,
        'limit': min(limit, 1000)  # Binance max is 1000 per request
    }
    
    for endpoint in endpoints:
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(endpoint, params=params, timeout=20)
                response.raise_for_status()
                klines = response.json()
                
                df = pd.DataFrame(klines, columns=[
                    "open_time", "open", "high", "low", "close", "volume", 
                    "close_time", "quote_asset_volume", "trades", 
                    "taker_base_vol", "taker_quote_vol", "ignore"
                ])
                
                # Convert timestamps and numeric values
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                numeric_cols = ["open", "high", "low", "close", "volume"]
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                return df[["open_time", "open", "high", "low", "close", "volume"]]
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    logging.warning(f"Symbol {symbol} not found on {endpoint}")
                    break  # Try next endpoint
                logging.warning(f"Attempt {attempt+1} failed for {symbol}: {e}")
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} error for {symbol}: {e}")
                time.sleep(RETRY_DELAY)
    
    return pd.DataFrame()

# ===================== SIGNAL DETECTION (UNCHANGED) =====================
def detect_signals(symbol, df):
    signals = []
    if len(df) < 10:
        return []

    # [Previous signal detection logic remains exactly the same]
    # ... (include all your existing signal detection code here)
    
    return signals

# ===================== IMPROVED WORKER =====================
def worker():
    while True:
        start_time = time.time()
        try:
            for symbol in SYMBOLS:
                df = fetch_candles(symbol)
                if df.empty:
                    logging.warning(f"No data for {symbol}")
                    continue
                
                # Update candle storage with deduplication
                if not candles_data[symbol].empty:
                    combined = pd.concat([candles_data[symbol], df])
                    combined = combined.drop_duplicates('open_time', keep='last')
                    candles_data[symbol] = combined.tail(LOOKBACK)
                else:
                    candles_data[symbol] = df.tail(LOOKBACK)
                
                # Signal detection and alerting
                signals = detect_signals(symbol, candles_data[symbol])
                if signals:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    price = candles_data[symbol]["close"].iloc[-1]
                    message = (
                        f"*{symbol} @ {timestamp}*\n"
                        + "\n".join([f"- {s}" for s in signals])
                        + f"\nPrice: {price:.8f}"
                    )
                    if send_telegram_alert(message):
                        logging.info(f"Sent alert for {symbol}")
                    else:
                        logging.error(f"Failed to send alert for {symbol}")
            
            # Precise timing control
            elapsed = time.time() - start_time
            sleep_time = max(60 - elapsed, 0)
            time.sleep(sleep_time)
            
        except Exception as e:
            logging.error(f"Worker error: {e}")
            time.sleep(10)

# ===================== FLASK ENDPOINTS (UNCHANGED) =====================
@app.route('/')
def index():
    return "Crypto Signal Bot is running."

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    return {"status": "ok"}

# ===================== LAUNCH =====================
if __name__ == '__main__':
    # Initial data load
    logging.info("Loading initial data...")
    for symbol in SYMBOLS:
        df = fetch_candles(symbol)
        if not df.empty:
            candles_data[symbol] = df.tail(LOOKBACK)
            logging.info(f"Loaded {len(df)} candles for {symbol}")
        else:
            logging.warning(f"Failed initial load for {symbol}")
    
    # Start worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
