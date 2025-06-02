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
TELEGRAM_BOT_TOKEN = '7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg'
TELEGRAM_CHANNEL_ID = '@tradegrh'

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT", "NEIROUSDT", "FLOKIUSDT",
    "BONKUSDT", "PEPEUSDT", "FLMUSDT", "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT",
    "GMTUSDT", "ADAUSDT", "OSMOUSDT"
]
INTERVAL = '1m'  # 1-minute candles
LOOKBACK = 5000  # Keep last 5000 candles in memory
MAX_RETRIES = 3  # Max retries for API calls
RETRY_DELAY = 5  # Seconds between retries

app = Flask(__name__)

# For each symbol, we'll store recent candles
candles_data = {symbol: pd.DataFrame() for symbol in SYMBOLS}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ===================== HELPER FUNCTIONS =====================
def send_telegram_alert(message):
    """Send alert to Telegram channel with retry logic."""
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
            else:
                logging.warning(f"Telegram send attempt {attempt + 1} failed: {response.text}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.warning(f"Telegram send attempt {attempt + 1} exception: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return False

def fetch_candles(symbol, limit=LOOKBACK):
    """Fetch candles from Binance API with retry logic."""
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': INTERVAL,
        'limit': limit
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "trades", "taker_base_vol", "taker_quote_vol", "ignore"
            ])
            
            # Convert timestamp to datetime and numeric columns to float
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df[["open_time", "open", "high", "low", "close", "volume"]]
            
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed to fetch candles for {symbol}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return pd.DataFrame()

# ===================== SIGNAL DETECTION =====================
def detect_signals(symbol, df):
    """Detect BOS, CHoCH, OB, FVG, 161.8 Fibo. Returns a list of signals."""
    if len(df) < 10:  # Need at least 10 candles for reliable signals
        return []
    
    signals = []
    last_close = df["close"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    prev_low = df["low"].iloc[-2]
    prev_close = df["close"].iloc[-2]
    
    # 1. Break of Structure (BOS)
    if last_close > prev_high:
        signals.append("BOS (Bullish)")
    elif last_close < prev_low:
        signals.append("BOS (Bearish)")
    
    # 2. Change of Character (CHoCH)
    if len(df) >= 3:
        if prev_close > df["close"].iloc[-3] and last_close < prev_close:
            signals.append("CHoCH (to Bearish)")
        elif prev_close < df["close"].iloc[-3] and last_close > prev_close:
            signals.append("CHoCH (to Bullish)")
    
    # 3. Order Block (OB) detection
    body = abs(prev_close - df["open"].iloc[-2])
    candle_range = prev_high - prev_low
    if candle_range > 0 and body / candle_range > 0.7:  # Strong candle
        if prev_close > df["open"].iloc[-2]:
            signals.append("OB (Bullish)")
        else:
            signals.append("OB (Bearish)")
    
    # 4. Fair Value Gap (FVG)
    if df["low"].iloc[-1] > df["high"].iloc[-2]:
        signals.append("FVG (Bullish gap)")
    elif df["high"].iloc[-1] < df["low"].iloc[-2]:
        signals.append("FVG (Bearish gap)")
    
    # 5. Fibonacci 161.8% projection
    if len(df) >= 6:
        swing_high = df["high"].iloc[-6:-1].max()
        swing_low = df["low"].iloc[-6:-1].min()
        range_size = swing_high - swing_low
        
        if range_size > 0:  # Valid swing
            direction = "Bullish" if last_close > prev_close else "Bearish"
            if direction == "Bullish":
                fib_target = swing_high + range_size * 0.618  # 161.8% extension
                if last_close >= fib_target:
                    signals.append("Fibo 161.8% reached (Bullish)")
            else:
                fib_target = swing_low - range_size * 0.618  # 161.8% extension
                if last_close <= fib_target:
                    signals.append("Fibo 161.8% reached (Bearish)")
    
    return signals

# ===================== MAIN WORKER =====================
def worker():
    """Main worker thread that fetches data and detects signals."""
    while True:
        start_time = time.time()
        try:
            for symbol in SYMBOLS:
                try:
                    # Fetch new candles
                    df = fetch_candles(symbol)
                    if df.empty:
                        logging.warning(f"No data received for {symbol}")
                        continue
                    
                    # Update stored data
                    if not candles_data[symbol].empty:
                        # Merge new data with existing, removing duplicates
                        combined = pd.concat([candles_data[symbol], df]).drop_duplicates(subset=['open_time'])
                        # Keep only the last LOOKBACK candles
                        candles_data[symbol] = combined.tail(LOOKBACK)
                    else:
                        candles_data[symbol] = df.tail(LOOKBACK)
                    
                    # Detect signals
                    signals = detect_signals(symbol, candles_data[symbol])
                    if signals:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        message = (
                            f"*{symbol} @ {timestamp}*\n"
                            + "\n".join([f"- {s}" for s in signals])
                            + f"\nPrice: {last_close:.8f}"
                        )
                        if send_telegram_alert(message):
                            logging.info(f"Alert sent for {symbol}: {signals}")
                        else:
                            logging.error(f"Failed to send alert for {symbol}")
                
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Calculate sleep time to maintain 1-minute interval
            processing_time = time.time() - start_time
            sleep_time = max(60 - processing_time, 0)
            time.sleep(sleep_time)
            
        except Exception as e:
            logging.error(f"Worker error: {e}")
            time.sleep(10)

# ===================== FLASK APP =====================
@app.route('/')
def index():
    """Health check endpoint."""
    return {
        "status": "running",
        "symbols": SYMBOLS,
        "last_updated": datetime.now().isoformat()
    }

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook endpoint for potential integrations."""
    data = request.json
    logging.info(f"Webhook received: {data}")
    return {"status": "ok", "received_data": data}

# ===================== LAUNCH =====================
if __name__ == '__main__':
    # Initial data fetch
    logging.info("Starting initial data fetch...")
    for symbol in SYMBOLS:
        df = fetch_candles(symbol)
        if not df.empty:
            candles_data[symbol] = df.tail(LOOKBACK)
            logging.info(f"Initial data loaded for {symbol} ({len(df)} candles)")
        else:
            logging.warning(f"Failed initial fetch for {symbol}")
    
    # Start the worker thread
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    logging.info("Worker thread started")
    
    # Run Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
