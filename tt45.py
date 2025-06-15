import os
import time
import math
import threading
import logging
from datetime import datetime
from flask import Flask, jsonify
from binance.client import Client
from telegram import Bot

# ================= CONFIGURATION =================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL", "@tradegrh")
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
CANDLE_LIMIT = 100

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# =================== UTILITIES ===================

def get_candles(symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT):
    """
    Fetch recent candlestick data from Binance API
    and transform it into a structured list of dicts.
    """
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    candles = []
    for r in raw:
        candle = {
            "open_time": r[0],
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
            "close_time": r[6],
            "quote_asset_volume": float(r[7]),
            "number_of_trades": r[8],
            "taker_buy_base_asset_volume": float(r[9]),
            "taker_buy_quote_asset_volume": float(r[10]),
        }
        candles.append(candle)
    return candles

def ema(values, period):
    """
    Calculate Exponential Moving Average (EMA) of a list of values.
    """
    if not values or period <= 0 or period > len(values):
        return 0
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def sma(values, period):
    """
    Calculate Simple Moving Average (SMA).
    """
    if not values or period <= 0 or period > len(values):
        return 0
    return sum(values[-period:]) / period

def rsi(candles, period=14):
    """
    Calculate Relative Strength Index (RSI) from candles.
    """
    if len(candles) < period + 1:
        return 50  # Neutral default
    gains = 0
    losses = 0
    for i in range(-period, -1):
        delta = candles[i+1]["close"] - candles[i]["close"]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    if losses == 0:
        return 100
    rs = gains / losses
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def atr(candles, period=14):
    """
    Calculate Average True Range (ATR) for volatility measurement.
    """
    if len(candles) < period + 1:
        return 0
    trs = []
    for i in range(-period, 0):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr_val = sum(trs) / len(trs)
    return atr_val

def donchian_channel(candles, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    """
    if len(candles) < period:
        return (0, 0)
    highs = [c["high"] for c in candles[-period:]]
    lows = [c["low"] for c in candles[-period:]]
    return max(highs), min(lows)

def heikin_ashi(candles):
    """
    Calculate Heikin-Ashi candle for the latest candle.
    """
    if len(candles) < 2:
        return None
    prev = candles[-2]
    curr = candles[-1]
    ha_close = (curr["open"] + curr["high"] + curr["low"] + curr["close"]) / 4
    ha_open = (prev["open"] + prev["close"]) / 2
    ha_high = max(curr["high"], ha_open, ha_close)
    ha_low = min(curr["low"], ha_open, ha_close)
    return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}

def timestamp_to_datetime(ts):
    """
    Convert timestamp in ms to human readable datetime string.
    """
    return datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

# ================= SHORT-TERM PRICE PREDICTION ================

def predict_next_15(candles):
    """
    Simple prediction of price movement over next 15 candles.
    Uses net price movement over last 16 closes.
    """
    if len(candles) < 16:
        return "Insufficient Data"
    closes = [c["close"] for c in candles[-16:]]
    net_move = sum(closes[i+1] - closes[i] for i in range(15))
    return "UP 📈" if net_move > 0 else "DOWN 📉"
# ================== CORE STRATEGIES ==================

# ---------- Trend Following Strategies ----------

def strategy_ema_cross(candles):
    """
    EMA Crossover: Bullish when 9 EMA crosses above 21 EMA, bearish otherwise.
    """
    closes = [c["close"] for c in candles]
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    return "🟢up" if ema9 > ema21 else "🔴down"

def strategy_macd(candles):
    """
    MACD simplified: Bullish if 12 EMA > 26 EMA, bearish otherwise.
    """
    closes = [c["close"] for c in candles]
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    return "🟢up" if ema12 > ema26 else "🔴down"

def strategy_adx(candles, period=14):
    """
    Average Directional Index (ADX) simplified calculation.
    Returns bullish if trend strong and positive, bearish if strong and negative.
    """
    if len(candles) < period + 1:
        return "⚪️neutral"

    plus_di = 0
    minus_di = 0
    tr_sum = 0

    for i in range(-period, -1):
        high = candles[i + 1]["high"]
        low = candles[i + 1]["low"]
        prev_close = candles[i]["close"]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_sum += tr

        plus_dm = high - candles[i]["high"]
        minus_dm = candles[i]["low"] - low

        if plus_dm > minus_dm and plus_dm > 0:
            plus_di += 100 * (plus_dm / tr)
        elif minus_dm > plus_dm and minus_dm > 0:
            minus_di += 100 * (minus_dm / tr)

    plus_di /= period
    minus_di /= period
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx / period

    if adx > 25 and plus_di > minus_di:
        return "🟢up"
    elif adx > 25 and minus_di > plus_di:
        return "🔴down"
    return "⚪️neutral"

def strategy_parabolic_sar(candles, accel=0.02, max_accel=0.2):
    """
    Parabolic SAR trend indicator.
    Returns bullish if price above SAR, bearish if below.
    """
    if len(candles) < 5:
        return "⚪️neutral"

    sar = candles[-3]["low"] if candles[-3]["close"] > candles[-4]["close"] else candles[-3]["high"]
    ep = candles[-3]["high"] if candles[-3]["close"] > candles[-4]["close"] else candles[-3]["low"]
    trend = "up" if candles[-3]["close"] > candles[-4]["close"] else "down"
    af = accel

    for i in range(-2, 0):
        if trend == "up":
            sar = sar + af * (ep - sar)
            if candles[i]["low"] < sar:
                trend = "down"
                sar = ep
                ep = candles[i]["low"]
                af = accel
            else:
                if candles[i]["high"] > ep:
                    ep = candles[i]["high"]
                    af = min(af + accel, max_accel)
        else:
            sar = sar + af * (ep - sar)
            if candles[i]["high"] > sar:
                trend = "up"
                sar = ep
                ep = candles[i]["high"]
                af = accel
            else:
                if candles[i]["low"] < ep:
                    ep = candles[i]["low"]
                    af = min(af + accel, max_accel)

    current_price = candles[-1]["close"]
    if current_price > sar and trend == "up":
        return "🟢up"
    elif current_price < sar and trend == "down":
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_keltner_channels(candles, ema_period=20, atr_period=10, multiplier=2):
    """
    Keltner Channels: bullish if price > upper band, bearish if price < lower band.
    """
    closes = [c["close"] for c in candles]
    ema_val = ema(closes, ema_period)
    atr_val = atr(candles, atr_period)
    upper = ema_val + multiplier * atr_val
    lower = ema_val - multiplier * atr_val
    current = candles[-1]["close"]

    if current > upper:
        return "🟢up"
    elif current < lower:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_trix(candles, period=15):
    """
    TRIX indicator: triple EMA of close prices, signal is positive or negative slope.
    """
    closes = [c["close"] for c in candles]
    if len(closes) < period * 3:
        return "⚪️neutral"

    def single_ema(data, p):
        k = 2 / (p + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = price * k + ema_val * (1 - k)
        return ema_val

    ema1_list = [single_ema(closes[i:i+period], period) for i in range(len(closes) - period + 1)]
    ema2_list = [single_ema(ema1_list[i:i+period], period) for i in range(len(ema1_list) - period + 1)]
    ema3_list = [single_ema(ema2_list[i:i+period], period) for i in range(len(ema2_list) - period + 1)]

    if len(ema3_list) < 2:
        return "⚪️neutral"

    trix_val = 100 * (ema3_list[-1] - ema3_list[-2]) / (ema3_list[-2] + 1e-9)
    if trix_val > 0:
        return "🟢up"
    elif trix_val < 0:
        return "🔴down"
    return "⚪️neutral"

def strategy_awesome_oscillator(candles):
    """
    Awesome Oscillator (AO): difference of 5-period and 34-period moving averages of midpoints.
    """
    if len(candles) < 34:
        return "⚪️neutral"
    mids = [(c["high"] + c["low"]) / 2 for c in candles[-34:]]
    short_ema = ema(mids[-5:], 5)
    long_ema = ema(mids, 34)
    prev_short_ema = ema(mids[-6:-1], 5)
    prev_long_ema = ema(mids[:-1], 34)
    ao = short_ema - long_ema
    prev_ao = prev_short_ema - prev_long_ema

    if ao > 0 and ao > prev_ao:
        return "🟢up"
    elif ao < 0 and ao < prev_ao:
        return "🔴down"
    else:
        return "⚪️neutral"

# ---------- Mean Reversion Strategies ----------

def strategy_rsi(candles, period=14):
    """
    RSI: Bullish if oversold (<30), Bearish if overbought (>70).
    """
    rsi_val = rsi(candles, period)
    if rsi_val < 30:
        return "🟢up"
    elif rsi_val > 70:
        return "🔴down"
    return "⚪️neutral"

def strategy_bollinger(candles):
    """
    Bollinger Bands: price outside upper band bearish, outside lower band bullish.
    """
    if len(candles) < 20:
        return "⚪️neutral"
    closes = [c["close"] for c in candles[-20:]]
    mean = sum(closes) / 20
    std = math.sqrt(sum((x - mean) ** 2 for x in closes) / 20)
    upper = mean + 2 * std
    lower = mean - 2 * std
    last = candles[-1]["close"]

    if last > upper:
        return "🔴down"
    elif last < lower:
        return "🟢up"
    else:
        return "⚪️neutral"

def strategy_stochastic(candles, k_period=14):
    """
    Stochastic Oscillator: bullish if %K < 20, bearish if > 80.
    """
    if len(candles) < k_period:
        return "⚪️neutral"
    highs = [c["high"] for c in candles[-k_period:]]
    lows = [c["low"] for c in candles[-k_period:]]
    closes = [c["close"] for c in candles[-k_period:]]
    lowest_low = min(lows)
    highest_high = max(highs)
    k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low + 1e-9)

    if k < 20:
        return "🟢up"
    elif k > 80:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_cci(candles, period=20):
    """
    Commodity Channel Index (CCI): bullish if < -100, bearish if > 100.
    """
    if len(candles) < period:
        return "⚪️neutral"
    tps = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles[-period:]]
    avg = sum(tps) / period
    mean_dev = sum(abs(tp - avg) for tp in tps) / period
    cci = (tps[-1] - avg) / (0.015 * mean_dev + 1e-9)

    if cci < -100:
        return "🟢up"
    elif cci > 100:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_mfi(candles, period=14):
    """
    Money Flow Index (MFI): bullish if oversold (<20), bearish if overbought (>80).
    """
    if len(candles) < period + 1:
        return "⚪️neutral"
    positive_flow = 0
    negative_flow = 0
    for i in range(-period, -1):
        tp = (candles[i]["high"] + candles[i]["low"] + candles[i]["close"]) / 3
        prev_tp = (candles[i-1]["high"] + candles[i-1]["low"] + candles[i-1]["close"]) / 3
        vol = candles[i]["volume"]

        if tp > prev_tp:
            positive_flow += tp * vol
        else:
            negative_flow += tp * vol

    if negative_flow == 0:
        return "🟢up"
    mfi_val = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-9)))

    if mfi_val < 20:
        return "🟢up"
    elif mfi_val > 80:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_vwap(candles):
    """
    Volume Weighted Average Price (VWAP) comparison.
    """
    total_vol = sum(c["volume"] for c in candles)
    if total_vol == 0:
        return "⚪️neutral"
    vwap = sum(c["close"] * c["volume"] for c in candles) / total_vol
    current = candles[-1]["close"]

    if current > vwap:
        return "🟢up"
    else:
        return "🔴down"

# ---------- Breakout Strategies ----------

def strategy_donchian_breakout(candles, period=20):
    """
    Donchian channel breakout: price above upper channel bullish, below lower channel bearish.
    """
    upper, lower = donchian_channel(candles, period)
    current = candles[-1]["close"]
    if current > upper:
        return "🟢up"
    elif current < lower:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_inside_bar(candles):
    """
    Inside bar breakout strategy.
    """
    if len(candles) < 2:
        return "⚪️neutral"
    curr = candles[-1]
    prev = candles[-2]

    if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
        return "⚪️neutral"
    elif curr["close"] > prev["high"]:
        return "🟢up"
    elif curr["close"] < prev["low"]:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_break_retest(candles):
    """
    Break and retest pattern.
    """
    if len(candles) < 3:
        return "⚪️neutral"
    prev = candles[-3]
    current = candles[-1]

    if current["close"] > prev["high"]:
        return "🟢up"
    elif current["close"] < prev["low"]:
        return "🔴down"
    else:
        return "⚪️neutral"
# =================== PRICE ACTION STRATEGIES ===================

def strategy_wick_rejection(candles):
    """
    Wick rejection candle: long lower wick bullish, long upper wick bearish.
    """
    c = candles[-1]
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    body = abs(c["close"] - c["open"])

    if lower_wick > body * 1.5:
        return "🟢up"
    elif upper_wick > body * 1.5:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_engulfing(candles):
    """
    Bullish/bearish engulfing pattern detection.
    """
    if len(candles) < 2:
        return "⚪️neutral"
    prev = candles[-2]
    curr = candles[-1]

    if prev["close"] < prev["open"] and curr["open"] < prev["close"] and curr["close"] > prev["open"]:
        return "🟢up"
    elif prev["close"] > prev["open"] and curr["open"] > prev["close"] and curr["close"] < prev["open"]:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_hammer(candles):
    """
    Hammer or Hanging Man candle pattern.
    """
    c = candles[-1]
    body = abs(c["close"] - c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    upper_wick = c["high"] - max(c["close"], c["open"])

    if lower_wick > body * 2 and upper_wick < body * 0.5 and c["close"] > c["open"]:
        return "🟢up"
    elif lower_wick > body * 2 and upper_wick < body * 0.5 and c["close"] < c["open"]:
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_doji(candles):
    """
    Doji candle detection: small body relative to average.
    """
    if len(candles) < 14:
        return "⚪️neutral"
    c = candles[-1]
    avg_body = sum(abs(candle["close"] - candle["open"]) for candle in candles[-14:]) / 14
    body = abs(c["close"] - c["open"])

    if body < avg_body * 0.1:
        return "⚪️neutral"
    else:
        return "⚪️neutral"

# ========== SMART MONEY CONCEPTS (SMC) ==========

def detect_order_block(candles):
    """
    Identify bullish/bearish order block in prior candles.
    """
    if len(candles) < 4:
        return None
    c = candles[-3]
    if c["close"] < c["open"]:
        return {"type": "Bullish OB", "price": c["open"]}
    else:
        return {"type": "Bearish OB", "price": c["close"]}

def detect_fvg(candles):
    """
    Detect Fair Value Gap between candles.
    """
    if len(candles) < 4:
        return None
    if candles[-3]["high"] < candles[-1]["low"]:
        return {"type": "FVG Buy", "zone": (candles[-3]["high"], candles[-1]["low"])}
    elif candles[-3]["low"] > candles[-1]["high"]:
        return {"type": "FVG Sell", "zone": (candles[-1]["high"], candles[-3]["low"])}
    return None

def strategy_fvg(candles):
    fvg = detect_fvg(candles)
    if fvg and fvg["type"] == "FVG Buy":
        return "🟢up"
    elif fvg and fvg["type"] == "FVG Sell":
        return "🔴down"
    return "⚪️neutral"

def strategy_ob(candles):
    ob = detect_order_block(candles)
    if ob:
        if ob["type"] == "Bullish OB":
            return "🟢up"
        elif ob["type"] == "Bearish OB":
            return "🔴down"
    return "⚪️neutral"

def strategy_liquidity_grab(candles):
    """
    Detect liquidity grab pattern (stop hunts).
    """
    if len(candles) < 5:
        return "⚪️neutral"
    if (candles[-2]["low"] < min(c["low"] for c in candles[-5:-2]) and
        candles[-1]["close"] > max(c["high"] for c in candles[-5:-2])):
        return "🟢up"
    elif (candles[-2]["high"] > max(c["high"] for c in candles[-5:-2]) and
          candles[-1]["close"] < min(c["low"] for c in candles[-5:-2])):
        return "🔴down"
    else:
        return "⚪️neutral"

def strategy_mitigation_block(candles):
    """
    Identify mitigation block reversal.
    """
    if len(candles) < 4:
        return "⚪️neutral"
    if (candles[-3]["close"] < candles[-3]["open"] and
        candles[-2]["low"] < candles[-3]["low"] and
        candles[-1]["close"] > candles[-3]["open"]):
        return "🟢up"
    elif (candles[-3]["close"] > candles[-3]["open"] and
          candles[-2]["high"] > candles[-3]["high"] and
          candles[-1]["close"] < candles[-3]["open"]):
        return "🔴down"
    else:
        return "⚪️neutral"

def ict(candles):
    """
    Inner Circle Trader (ICT) concept: market structure shift.
    """
    if len(candles) < 4:
        return "⚪️neutral"
    if candles[-3]["low"] > candles[-1]["low"] and candles[-1]["close"] > candles[-3]["high"]:
        return "🟢up"
    elif candles[-3]["high"] < candles[-1]["high"] and candles[-1]["close"] < candles[-3]["low"]:
        return "🔴down"
    else:
        return "⚪️neutral"

def smc(candles):
    """
    Smart Money Concept liquidity grab detection.
    """
    if len(candles) < 4:
        return "⚪️neutral"
    if (candles[-2]["low"] < candles[-3]["low"] and candles[-1]["close"] > candles[-3]["high"]):
        return "🟢up"
    elif (candles[-2]["high"] > candles[-3]["high"] and candles[-1]["close"] < candles[-3]["low"]):
        return "🔴down"
    else:
        return "⚪️neutral"

# ========== STRATEGY RUNNER & SIGNAL AGGREGATION ==========

def run_all_strategies(candles):
    """
    Run all implemented strategies and return their signals.
    """
    strategies = {
        # Trend Following
        "EMA Crossover": strategy_ema_cross(candles),
        "MACD": strategy_macd(candles),
        "ADX": strategy_adx(candles),
        "Parabolic SAR": strategy_parabolic_sar(candles),
        "Keltner Channels": strategy_keltner_channels(candles),
        "TRIX": strategy_trix(candles),
        "Awesome Oscillator": strategy_awesome_oscillator(candles),

        # Mean Reversion
        "RSI": strategy_rsi(candles),
        "Bollinger Bands": strategy_bollinger(candles),
        "Stochastic": strategy_stochastic(candles),
        "CCI": strategy_cci(candles),
        "MFI": strategy_mfi(candles),
        "VWAP": strategy_vwap(candles),

        # Breakout
        "Donchian Breakout": strategy_donchian_breakout(candles),
        "Inside Bar": strategy_inside_bar(candles),
        "Break+Retest": strategy_break_retest(candles),

        # Price Action
        "Wick Rejection": strategy_wick_rejection(candles),
        "Engulfing": strategy_engulfing(candles),
        "Hammer": strategy_hammer(candles),
        "Doji": strategy_doji(candles),

        # Smart Money Concepts
        "FVG": strategy_fvg(candles),
        "Order Block": strategy_ob(candles),
        "Liquidity Grab": strategy_liquidity_grab(candles),
        "Mitigation Block": strategy_mitigation_block(candles),
        "ICT": ict(candles),
        "SMC": smc(candles),
    }
    return strategies

def aggregate_signals(signals):
    """
    Aggregate individual strategy signals into a majority decision.
    """
    votes = {"🟢up": 0, "🔴down": 0, "⚪️neutral": 0}
    for signal in signals.values():
        if signal in votes:
            votes[signal] += 1
    if votes["🟢up"] > votes["🔴down"]:
        return "BUY ✅", votes
    elif votes["🔴down"] > votes["🟢up"]:
        return "SELL ❌", votes
    else:
        return "WAIT 🕒", votes

# ========== TELEGRAM MESSAGE GENERATION ==========

def fib_extension(entry, atr_value, direction):
    """
    Calculate Fibonacci extension target for take profit.
    """
    base = atr_value * 1.618
    if direction == "BUY":
        return round(entry + base, 2)
    else:
        return round(entry - base, 2)

def generate_telegram_message(candles, signals, votes, decision, prediction):
    """
    Compose detailed Telegram alert message summarizing analysis.
    """
    entry = candles[-1]["close"]
    atr_val = atr(candles, 14)
    stop_loss = round(entry - atr_val, 2) if decision == "BUY ✅" else round(entry + atr_val, 2)
    fib_tp = fib_extension(entry, atr_val, "BUY" if decision == "BUY ✅" else "SELL")

    ob = detect_order_block(candles)
    fvg = detect_fvg(candles)

    msg = f"📊 *BTCUSDT 1-min Multi-Strategy Analysis*\n\n"
    msg += f"🧠 *Majority Decision*: {decision} (🟢{votes['🟢up']} | 🔴{votes['🔴down']} | ⚪️{votes['⚪️neutral']})\n"
    msg += f"📈 *Next 15 Candle Forecast*: {prediction}\n"
    msg += f"🎯 *Entry*: `{entry:.2f}` | 🛑 Stop Loss: `{stop_loss}` | 🎯 Take Profit (Fib 1.618): `{fib_tp}`\n\n"

    msg += "*Signals Overview*\n"
    for name, signal in signals.items():
        if signal != "⚪️neutral":
            msg += f"• {name}: {signal}\n"

    if ob:
        msg += f"\n🧱 Order Block: {ob['type']} at {ob['price']:.2f}"
    if fvg:
        zone = fvg["zone"]
        msg += f"\n📐 Fair Value Gap: {fvg['type']} Zone {zone[0]:.2f} → {zone[1]:.2f}"

    msg += "\n\n📣 Powered by Multi-Strategy AI Bot 🤖"
    return msg

# ========== MASTER ANALYSIS FUNCTION ==========

def analyze():
    """
    Main function to fetch data, run strategies, aggregate signals,
    generate message, and send alert via Telegram.
    """
    candles = get_candles()
    if len(candles) < 30:
        logging.warning("Insufficient candle data for analysis.")
        return

    signals = run_all_strategies(candles)
    decision, votes = aggregate_signals(signals)
    prediction = predict_next_15(candles)
    message = generate_telegram_message(candles, signals, votes, decision, prediction)

    # Send message to Telegram channel
    try:
        bot.send_message(chat_id=TELEGRAM_CHANNEL, text=message, parse_mode="Markdown")
        logging.info(f"Sent alert to Telegram: {decision}")
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")

# ========== EXECUTION LOOP + FLASK SERVER ==========

def run_scheduler():
    """
    Continuously run analyze() every 60 seconds in background thread.
    """
    while True:
        try:
            analyze()
        except Exception as e:
            logging.error(f"Error in analyze loop: {e}")
        time.sleep(60)

@app.route('/')
def health_check():
    """
    Simple health check endpoint for monitoring.
    """
    return jsonify({"status": "BTCUSDT Strategy Bot is Running", "time": datetime.utcnow().isoformat()})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    threading.Thread(target=run_scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
