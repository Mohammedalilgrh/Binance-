import os
import time
import threading
from datetime import datetime
from flask import Flask

from binance.client import Client
from telegram import Bot

# =================== CONFIG ===================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL", "@tradegrh")

SYMBOLS = [
    "BTCUSDT", "SHIBUSDT", "DOGEUSDT", "XRPUSDT", "BTTCUSDT",
    "NEARUSDT", "FLOKIUSDT", "BONKUSDT", "PEPEUSDT", "FLMUSDT",
    "ARBUSDT", "SUSHIUSDT", "XLMUSDT", "CFXUSDT", "GMTUSDT",
    "ADAUSDT", "OSMOUSDT"
]
CANDLE_LIMIT = 5000

# =================== INIT ===================
app = Flask(__name__)
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(TELEGRAM_BOT_TOKEN)

candle_cache = {symbol: [] for symbol in SYMBOLS}
last_signal_sent = {symbol: None for symbol in SYMBOLS}

# =================== MARKET STRUCTURE ===================
def fetch_klines(symbol, interval="1m", limit=CANDLE_LIMIT):
    try:
        klines = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
        result = []
        for k in klines:
            result.append({
                'time': k[0] // 1000,
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })
        return result
    except Exception as e:
        print(f"[{symbol}] Error fetching klines: {e}")
        return []

def detect_bos(klines, lookback=20):
    if len(klines) < lookback + 2:
        return None
    highs = [k['high'] for k in klines[-lookback-1:-1]]
    lows = [k['low'] for k in klines[-lookback-1:-1]]
    last_close = klines[-1]['close']
    swing_high = max(highs)
    swing_low = min(lows)
    if last_close > swing_high:
        return {'type': 'bull', 'level': swing_high}
    elif last_close < swing_low:
        return {'type': 'bear', 'level': swing_low}
    return None

def detect_choch(klines, lookback=7):
    if len(klines) < lookback + 2:
        return None
    highs = [k['high'] for k in klines[-lookback-1:-1]]
    lows = [k['low'] for k in klines[-lookback-1:-1]]
    prev_high = max(highs[:-2])
    prev_low = min(lows[:-2])
    if (klines[-2]['close'] > prev_high and klines[-1]['close'] < klines[-2]['low']):
        return {'type': 'bear', 'level': klines[-2]['low']}
    elif (klines[-2]['close'] < prev_low and klines[-1]['close'] > klines[-2]['high']):
        return {'type': 'bull', 'level': klines[-2]['high']}
    return None

def detect_order_blocks(klines, window=6):
    obs = []
    for i in range(window, len(klines)):
        # Bull OB
        if (klines[i-1]['close'] < klines[i-1]['open'] and
            klines[i]['close'] > klines[i-1]['high'] and
            klines[i]['volume'] > sum(k['volume'] for k in klines[i-window:i]) / window):
            obs.append({'type': 'bull', 'price': klines[i-1]['low'], 'index': i-1})
        # Bear OB
        if (klines[i-1]['close'] > klines[i-1]['open'] and
            klines[i]['close'] < klines[i-1]['low'] and
            klines[i]['volume'] > sum(k['volume'] for k in klines[i-window:i]) / window):
            obs.append({'type': 'bear', 'price': klines[i-1]['high'], 'index': i-1})
    return obs[-2:] if obs else []

def detect_fvg(klines):
    fvgs = []
    for i in range(2, len(klines)):
        prev = klines[i-2]
        curr = klines[i]
        if curr['low'] > prev['high']:
            fvgs.append({'type':'bull', 'zone': (prev['high'], curr['low']), 'index': i})
        if curr['high'] < prev['low']:
            fvgs.append({'type':'bear', 'zone': (curr['high'], prev['low']), 'index': i})
    return fvgs[-2:] if fvgs else []

def fibo_levels(high, low):
    diff = high - low
    return {
        '0': high,
        '0.236': high - diff * 0.236,
        '0.382': high - diff * 0.382,
        '0.5': high - diff * 0.5,
        '0.618': high - diff * 0.618,
        '0.786': high - diff * 0.786,
        '1': low,
        '1.618': high - diff * 1.618,
    }

def analyze(symbol, klines):
    bos = detect_bos(klines)
    choch = detect_choch(klines)
    obs = detect_order_blocks(klines)
    fvgs = detect_fvg(klines)
    recent_high = max(k['high'] for k in klines[-40:])
    recent_low = min(k['low'] for k in klines[-40:])
    fib = fibo_levels(recent_high, recent_low)
    msg = f"<b>📢 {symbol} 1m Market Structure</b>\n"
    if bos: msg += f"• BOS: <b>{bos['type'].upper()}</b> at {bos['level']:.6f}\n"
    if choch: msg += f"• CHoCH: <b>{choch['type'].upper()}</b> at {choch['level']:.6f}\n"
    if obs:
        for ob in obs:
            msg += f"• OB: <b>{ob['type'].upper()}</b> at {ob['price']:.6f}\n"
    if fvgs:
        for fvg in fvgs:
            msg += f"• FVG: <b>{fvg['type'].upper()}</b> {min(fvg['zone']):.6f}-{max(fvg['zone']):.6f}\n"
    msg += f"• Fibo 1.618 extension: <b>{fib['1.618']:.6f}</b>\n"
    msg += f"• Time: <code>{datetime.utcfromtimestamp(klines[-1]['time']).strftime('%Y-%m-%d %H:%M')}</code> UTC"
    return msg

def should_send(symbol, msg):
    if last_signal_sent[symbol] != msg:
        last_signal_sent[symbol] = msg
        return True
    return False

def send_signal(symbol, msg):
    try:
        bot.send_message(
            chat_id=TELEGRAM_CHANNEL,
            text=msg,
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[{symbol}] Telegram send error: {e}")

def update_and_signal():
    while True:
        for symbol in SYMBOLS:
            klines = fetch_klines(symbol)
            if klines and len(klines) >= 50:
                candle_cache[symbol] = klines[-CANDLE_LIMIT:]
                msg = analyze(symbol, candle_cache[symbol])
                if should_send(symbol, msg):
                    send_signal(symbol, msg)
        time.sleep(60)

# =================== FLASK APP ===================
@app.route('/')
def index():
    return "Institutional Sniper Bot (Binance 1m) is running!"

@app.route('/healthz')
def healthz():
    return "ok"

if __name__ == '__main__':
    threading.Thread(target=update_and_signal, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
