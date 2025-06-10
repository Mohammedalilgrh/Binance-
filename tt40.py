import os
import time
import math
import threading
import logging
from datetime import datetime
from flask import Flask
from binance.client import Client
from telegram import Bot

# ================= CONFIG =================
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

# ================ UTILITIES =================

def get_candles(symbol=SYMBOL, interval=INTERVAL, limit=CANDLE_LIMIT):
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    return [{
        "open": float(r[1]),
        "high": float(r[2]),
        "low": float(r[3]),
        "close": float(r[4]),
        "volume": float(r[5]),
        "time": r[0]
    } for r in raw]

def ema(values, period):
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

def rsi(candles, period=14):
    gains, losses = 0, 0
    for i in range(-period, -1):
        change = candles[i+1]["close"] - candles[i]["close"]
        if change > 0:
            gains += change
        else:
            losses -= change
    if losses == 0: return 100
    rs = gains / losses
    return 100 - (100 / (1 + rs))

def atr(candles, period=14):
    trs = []
    for i in range(1, period + 1):
        high = candles[-i]["high"]
        low = candles[-i]["low"]
        prev_close = candles[-i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / len(trs)

def donchian_channel(candles, period=20):
    highs = [c["high"] for c in candles[-period:]]
    lows = [c["low"] for c in candles[-period:]]
    return max(highs), min(lows)

def heikin_ashi(candles):
    ha_close = (candles[-1]["open"] + candles[-1]["high"] + candles[-1]["low"] + candles[-1]["close"]) / 4
    ha_open = (candles[-2]["open"] + candles[-2]["close"]) / 2
    ha_high = max(candles[-1]["high"], ha_open, ha_close)
    ha_low = min(candles[-1]["low"], ha_open, ha_close)
    return {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close}

# ============== TREND PREDICTOR ==============

def predict_next_15(candles):
    closes = [c["close"] for c in candles[-16:]]
    net_move = sum([closes[i+1] - closes[i] for i in range(15)])
    return "UP 📈" if net_move > 0 else "DOWN 📉"

# ============== SMART MONEY CORE ==============

def detect_order_block(candles):
    c = candles[-3]
    if c["close"] < c["open"]:
        return {"type": "Bullish OB", "price": c["open"]}
    else:
        return {"type": "Bearish OB", "price": c["close"]}

def detect_fvg(candles):
    if candles[-3]["high"] < candles[-1]["low"]:
        return {"type": "FVG Buy", "zone": (candles[-3]["high"], candles[-1]["low"])}
    elif candles[-3]["low"] > candles[-1]["high"]:
        return {"type": "FVG Sell", "zone": (candles[-1]["high"], candles[-3]["low"])}
    return None

def fib_extension(entry, atr_value, direction):
    base = atr_value * 1.618
    return round(entry + base, 2) if direction == "BUY" else round(entry - base, 2)

# ============== CHART PATTERN DETECTION ==============

def detect_horizontal_levels(candles):
    # Find significant highs and lows
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    
    # Cluster nearby levels (within 0.5% of price)
    def cluster_levels(levels):
        clusters = []
        for level in sorted(levels):
            found = False
            for cluster in clusters:
                if abs(level - cluster[0])/cluster[0] < 0.005:  # 0.5% threshold
                    cluster.append(level)
                    found = True
                    break
            if not found:
                clusters.append([level])
        return clusters
    
    high_clusters = cluster_levels(highs)
    low_clusters = cluster_levels(lows)
    
    # Sort clusters by importance (number of touches and recentness)
    def score_cluster(cluster):
        recent_weight = sum(1/(len(candles)-i) for i, price in enumerate(highs) if price in cluster)
        return len(cluster) * recent_weight
    
    high_clusters.sort(key=score_cluster, reverse=True)
    low_clusters.sort(key=score_cluster, reverse=True)
    
    # Get top 4 levels (2 from highs, 2 from lows)
    significant_levels = []
    if len(high_clusters) > 0:
        significant_levels.append(round(sum(high_clusters[0])/len(high_clusters[0]), 2))
    if len(high_clusters) > 1:
        significant_levels.append(round(sum(high_clusters[1])/len(high_clusters[1]), 2))
    if len(low_clusters) > 0:
        significant_levels.append(round(sum(low_clusters[0])/len(low_clusters[0]), 2))
    if len(low_clusters) > 1:
        significant_levels.append(round(sum(low_clusters[1])/len(low_clusters[1]), 2))
    
    # Sort all levels and take top 4 most significant
    significant_levels = sorted(significant_levels, reverse=True)
    return significant_levels[:4]

def analyze_chart_pattern(candles):
    levels = detect_horizontal_levels(candles)
    current_price = candles[-1]["close"]
    
    pattern = ""
    if len(levels) >= 4:
        # Check for ascending/descending channel
        if levels[0] > levels[1] > levels[2] > levels[3]:
            pattern = "Descending Channel (Lower Highs)"
        elif levels[0] < levels[1] < levels[2] < levels[3]:
            pattern = "Ascending Channel (Higher Lows)"
        else:
            # Check for rectangle pattern
            max_diff = max(levels) - min(levels)
            if max_diff < (max(levels) * 0.01):  # Within 1% range
                pattern = "Consolidation Rectangle"
            else:
                # Check for triangle patterns
                top_levels = [l for l in levels if l > current_price]
                bottom_levels = [l for l in levels if l < current_price]
                if len(top_levels) >= 2 and len(bottom_levels) >= 2:
                    if sorted(top_levels, reverse=True) == top_levels and sorted(bottom_levels) == bottom_levels:
                        pattern = "Symmetrical Triangle"
                    elif sorted(top_levels, reverse=True) == top_levels and len(set(bottom_levels)) == 1:
                        pattern = "Descending Triangle"
                    elif sorted(bottom_levels) == bottom_levels and len(set(top_levels)) == 1:
                        pattern = "Ascending Triangle"
        
        # Check breakout direction
        if pattern:
            if current_price > max(levels):
                pattern += " - Bullish Breakout"
            elif current_price < min(levels):
                pattern += " - Bearish Breakout"
            else:
                pattern += " - Inside Range"
    
    return {
        "levels": levels,
        "pattern": pattern if pattern else "No Clear Pattern",
        "current_position": f"Price is {'above' if current_price > sum(levels)/len(levels) else 'below'} mean level"
    }

# ============== ICT & SMC STRATEGIES ==============

def ict(candles):
    # ICT (Inner Circle Trader) strategy
    # Checks for market structure shift
    if candles[-3]["low"] > candles[-1]["low"] and candles[-1]["close"] > candles[-3]["high"]:
        return "🟢up"
    elif candles[-3]["high"] < candles[-1]["high"] and candles[-1]["close"] < candles[-3]["low"]:
        return "🔴down"
    return "⚪️neutral"

def smc(candles):
    # Smart Money Concept strategy
    # Checks for liquidity grabs and stop hunts
    if (candles[-2]["low"] < candles[-3]["low"] and 
        candles[-1]["close"] > candles[-3]["high"]):
        return "🟢up"
    elif (candles[-2]["high"] > candles[-3]["high"] and 
          candles[-1]["close"] < candles[-3]["low"]):
        return "🔴down"
    return "⚪️neutral"

# ============== EXPANDED STRATEGIES (70+) ==============

# Trend Following Strategies
def strategy_ema_cross(candles):
    closes = [c["close"] for c in candles[-21:]]
    return "🟢up" if ema(closes[-21:], 9) > ema(closes[-21:], 21) else "🔴down"

def strategy_macd(candles):
    closes = [c["close"] for c in candles[-26:]]
    return "🟢up" if ema(closes[-26:], 12) > ema(closes[-26:], 26) else "🔴down"

def strategy_adx(candles, period=14):
    plus_di = 0
    minus_di = 0
    tr_sum = 0
    for i in range(-period, -1):
        high = candles[i+1]["high"]
        low = candles[i+1]["low"]
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
    if len(candles) < 3:
        return "⚪️neutral"
    
    # Initialize
    ep = candles[-3]["high"] if candles[-3]["close"] > candles[-4]["close"] else candles[-3]["low"]
    sar = candles[-3]["low"] if candles[-3]["close"] > candles[-4]["close"] else candles[-3]["high"]
    trend = "up" if candles[-3]["close"] > candles[-4]["close"] else "down"
    accel_factor = accel
    
    # Calculate SAR
    for i in range(-2, 0):
        if trend == "up":
            sar = sar + accel_factor * (ep - sar)
            if candles[i]["low"] < sar:
                trend = "down"
                sar = ep
                ep = candles[i]["low"]
                accel_factor = accel
            else:
                if candles[i]["high"] > ep:
                    ep = candles[i]["high"]
                    accel_factor = min(accel_factor + accel, max_accel)
        else:
            sar = sar + accel_factor * (ep - sar)
            if candles[i]["high"] > sar:
                trend = "up"
                sar = ep
                ep = candles[i]["high"]
                accel_factor = accel
            else:
                if candles[i]["low"] < ep:
                    ep = candles[i]["low"]
                    accel_factor = min(accel_factor + accel, max_accel)
    
    current_price = candles[-1]["close"]
    return "🟢up" if current_price > sar and trend == "up" else "🔴down" if current_price < sar and trend == "down" else "⚪️neutral"

def strategy_keltner_channels(candles, ema_period=20, atr_period=10, multiplier=2):
    closes = [c["close"] for c in candles[-ema_period:]]
    ema_val = ema(closes, ema_period)
    atr_val = atr(candles, atr_period)
    
    upper = ema_val + multiplier * atr_val
    lower = ema_val - multiplier * atr_val
    
    current = candles[-1]["close"]
    return "🟢up" if current > upper else "🔴down" if current < lower else "⚪️neutral"

def strategy_trix(candles, period=15):
    closes = [c["close"] for c in candles]
    
    # Triple EMA
    ema1 = ema(closes[-period:], period)
    ema2 = ema([ema(c[:i], period) for i, c in enumerate(closes[-2*period:])], period)
    ema3 = ema([ema([ema(c[:j], period) for j, c in enumerate(closes[-2*period-i:])], period) 
               for i in range(period)], period)
    
    prev_ema3 = ema([ema([ema(c[:j], period) for j, c in enumerate(closes[-2*period-i-1:-1])], period) 
                    for i in range(period)], period)
    
    trix = 100 * (ema3 - prev_ema3) / prev_ema3
    return "🟢up" if trix > 0 else "🔴down" if trix < 0 else "⚪️neutral"

def strategy_awesome_oscillator(candles):
    midpoints = [(c["high"] + c["low"]) / 2 for c in candles[-34:]]
    ao = ema(midpoints[-5:], 5) - ema(midpoints[-34:], 34)
    
    prev_ao = ema(midpoints[-6:-1], 5) - ema(midlines[-35:-1], 34)
    
    if ao > 0 and ao > prev_ao:
        return "🟢up"
    elif ao < 0 and ao < prev_ao:
        return "🔴down"
    return "⚪️neutral"

# Mean Reversion Strategies
def strategy_rsi(candles, period=14):
    rsi_val = rsi(candles, period)
    if rsi_val < 30: return "🟢up"
    elif rsi_val > 70: return "🔴down"
    return "⚪️neutral"

def strategy_bollinger(candles):
    closes = [c["close"] for c in candles[-20:]]
    avg = sum(closes) / 20
    std = math.sqrt(sum((x - avg) ** 2 for x in closes) / 20)
    upper = avg + 2 * std
    lower = avg - 2 * std
    last = candles[-1]["close"]
    if last > upper: return "🔴down"
    elif last < lower: return "🟢up"
    return "⚪️neutral"

def strategy_stochastic(candles):
    highs = [c["high"] for c in candles[-14:]]
    lows = [c["low"] for c in candles[-14:]]
    closes = [c["close"] for c in candles[-14:]]
    k = 100 * (closes[-1] - min(lows)) / (max(highs) - min(lows) + 1e-9)
    return "🔴down" if k > 80 else "🟢up" if k < 20 else "⚪️neutral"

def strategy_cci(candles, period=20):
    tps = [(c["high"] + c["low"] + c["close"]) / 3 for c in candles[-period:]]
    avg = sum(tps) / len(tps)
    dev = sum(abs(tp - avg) for tp in tps) / len(tps)
    cci = (tps[-1] - avg) / (0.015 * dev + 1e-9)
    if cci > 100: return "🔴down"
    elif cci < -100: return "🟢up"
    return "⚪️neutral"

def strategy_mfi(candles, period=14):
    positive = 0
    negative = 0
    for i in range(-period, -1):
        tp = (candles[i]["high"] + candles[i]["low"] + candles[i]["close"]) / 3
        prev_tp = (candles[i-1]["high"] + candles[i-1]["low"] + candles[i-1]["close"]) / 3
        
        if tp > prev_tp:
            positive += tp * candles[i]["volume"]
        else:
            negative += tp * candles[i]["volume"]
    
    if negative == 0: return 100
    mfi = 100 - (100 / (1 + positive / negative))
    return "🟢up" if mfi < 20 else "🔴down" if mfi > 80 else "⚪️neutral"

def strategy_vwap(candles):
    price_volume = sum([c["close"] * c["volume"] for c in candles])
    total_volume = sum([c["volume"] for c in candles])
    vwap = price_volume / total_volume
    return "🟢up" if candles[-1]["close"] > vwap else "🔴down"

# Breakout Strategies
def strategy_donchian_breakout(candles, period=20):
    upper, lower = donchian_channel(candles, period)
    current = candles[-1]["close"]
    return "🟢up" if current > upper else "🔴down" if current < lower else "⚪️neutral"

def strategy_inside_bar(candles):
    if candles[-1]["high"] < candles[-2]["high"] and candles[-1]["low"] > candles[-2]["low"]:
        return "⚪️neutral"
    elif candles[-1]["close"] > candles[-2]["high"]:
        return "🟢up"
    elif candles[-1]["close"] < candles[-2]["low"]:
        return "🔴down"
    return "⚪️neutral"

def strategy_break_retest(candles):
    prev = candles[-3]
    curr = candles[-1]
    if curr["close"] > prev["high"]:
        return "🟢up"
    elif curr["close"] < prev["low"]:
        return "🔴down"
    return "⚪️neutral"

def strategy_high_low_breakout(candles, period=5):
    highs = [c["high"] for c in candles[-period:]]
    lows = [c["low"] for c in candles[-period:]]
    current = candles[-1]["close"]
    return "🟢up" if current > max(highs) else "🔴down" if current < min(lows) else "⚪️neutral"

def strategy_fractal_breakout(candles):
    # Bearish fractal: high with two lower highs on each side
    # Bullish fractal: low with two higher lows on each side
    if len(candles) < 5:
        return "⚪️neutral"
    
    bearish = False
    bullish = False
    
    # Check for bearish fractal
    if (candles[-3]["high"] > candles[-5]["high"] and 
        candles[-3]["high"] > candles[-4]["high"] and
        candles[-3]["high"] > candles[-2]["high"] and
        candles[-3]["high"] > candles[-1]["high"]):
        bearish = True
    
    # Check for bullish fractal
    if (candles[-3]["low"] < candles[-5]["low"] and 
        candles[-3]["low"] < candles[-4]["low"] and
        candles[-3]["low"] < candles[-2]["low"] and
        candles[-3]["low"] < candles[-1]["low"]):
        bullish = True
    
    current = candles[-1]["close"]
    if bullish and current > candles[-3]["high"]:
        return "🟢up"
    elif bearish and current < candles[-3]["low"]:
        return "🔴down"
    return "⚪️neutral"

# Momentum Strategies
def strategy_momentum(candles, period=10):
    current = candles[-1]["close"]
    past = candles[-period-1]["close"]
    change = (current - past) / past * 100
    return "🟢up" if change > 1 else "🔴down" if change < -1 else "⚪️neutral"

def strategy_roc(candles, period=12):
    current = candles[-1]["close"]
    past = candles[-period-1]["close"]
    roc = (current - past) / past * 100
    return "🟢up" if roc > 0 else "🔴down" if roc < 0 else "⚪️neutral"

def strategy_williams_r(candles, period=14):
    highs = [c["high"] for c in candles[-period:]]
    lows = [c["low"] for c in candles[-period:]]
    current = candles[-1]["close"]
    wr = (max(highs) - current) / (max(highs) - min(lows)) * -100
    return "🟢up" if wr < -80 else "🔴down" if wr > -20 else "⚪️neutral"

def strategy_ultimate_oscillator(candles):
    # Calculate buying pressure
    bp = []
    tr = []
    for i in range(-7, 0):
        bp.append(candles[i]["close"] - min(candles[i]["low"], candles[i-1]["close"]))
        tr.append(max(candles[i]["high"], candles[i-1]["close"]) - min(candles[i]["low"], candles[i-1]["close"]))
    
    avg7 = sum(bp) / sum(tr)
    
    bp = []
    tr = []
    for i in range(-14, 0):
        bp.append(candles[i]["close"] - min(candles[i]["low"], candles[i-1]["close"]))
        tr.append(max(candles[i]["high"], candles[i-1]["close"]) - min(candles[i]["low"], candles[i-1]["close"]))
    
    avg14 = sum(bp[-14:]) / sum(tr[-14:])
    avg28 = sum(bp) / sum(tr)
    
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    return "🟢up" if uo < 30 else "🔴down" if uo > 70 else "⚪️neutral"

def strategy_obv(candles):
    obv = 0
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i-1]["close"]:
            obv += candles[i]["volume"]
        elif candles[i]["close"] < candles[i-1]["close"]:
            obv -= candles[i]["volume"]
    
    # Simple 5-period EMA of OBV
    obv_ema = ema([obv] * 5, 5)  # Simplified for demo
    return "🟢up" if obv_ema > 0 else "🔴down" if obv_ema < 0 else "⚪️neutral"

# Volume-based Strategies
def strategy_volume_spike(candles, period=10):
    vol = [c["volume"] for c in candles[-period:]]
    avg = sum(vol[:-1]) / len(vol[:-1])
    if vol[-1] > avg * 1.8:
        return "🟢up" if candles[-1]["close"] > candles[-1]["open"] else "🔴down"
    return "⚪️neutral"

def strategy_vwap_macd_combo(candles):
    return "🟢up" if strategy_macd(candles) == "🟢up" and strategy_vwap(candles) == "🟢up" else "🔴down"

def strategy_volume_profile(candles, levels=4):
    # Simplified volume profile analysis
    price_ranges = []
    price_step = (max(c["high"] for c in candles[-50:]) - min(c["low"] for c in candles[-50:])) / levels
    for i in range(levels):
        low = min(c["low"] for c in candles[-50:]) + i * price_step
        high = low + price_step
        vol = sum(c["volume"] for c in candles[-50:] if low <= c["close"] <= high)
        price_ranges.append((low, high, vol))
    
    price_ranges.sort(key=lambda x: x[2], reverse=True)
    current = candles[-1]["close"]
    
    # If price is in high volume node
    for low, high, vol in price_ranges[:2]:
        if low <= current <= high:
            return "⚪️neutral"  # Likely to reverse from this level
    
    # If price is breaking out of low volume node
    for low, high, vol in price_ranges[-2:]:
        if current > high:
            return "🟢up"
        elif current < low:
            return "🔴down"
    
    return "⚪️neutral"

def strategy_accumulation_distribution(candles):
    ad = 0
    for c in candles[-14:]:
        clv = ((c["close"] - c["low"]) - (c["high"] - c["close"])) / (c["high"] - c["low"] + 1e-9)
        ad += clv * c["volume"]
    
    prev_ad = 0
    for c in candles[-15:-1]:
        clv = ((c["close"] - c["low"]) - (c["high"] - c["close"])) / (c["high"] - c["low"] + 1e-9)
        prev_ad += clv * c["volume"]
    
    return "🟢up" if ad > prev_ad else "🔴down" if ad < prev_ad else "⚪️neutral"

# Candlestick Pattern Strategies
def strategy_wick_rejection(candles):
    c = candles[-1]
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    body = abs(c["close"] - c["open"])
    if lower_wick > body * 1.5:
        return "🟢up"
    elif upper_wick > body * 1.5:
        return "🔴down"
    return "⚪️neutral"

def strategy_engulfing(candles):
    prev = candles[-2]
    curr = candles[-1]
    
    # Bullish engulfing
    if (prev["close"] < prev["open"] and 
        curr["open"] < prev["close"] and 
        curr["close"] > prev["open"]):
        return "🟢up"
    
    # Bearish engulfing
    elif (prev["close"] > prev["open"] and 
          curr["open"] > prev["close"] and 
          curr["close"] < prev["open"]):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_hammer(candles):
    c = candles[-1]
    body = abs(c["close"] - c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    upper_wick = c["high"] - max(c["close"], c["open"])
    
    # Hammer
    if (lower_wick > body * 2 and 
        upper_wick < body * 0.5 and
        c["close"] > c["open"]):
        return "🟢up"

    # Hanging man
    elif (lower_wick > body * 2 and 
          upper_wick < body * 0.5 and
          c["close"] < c["open"]):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_harami(candles):
    prev = candles[-2]
    curr = candles[-1]
    
    # Bullish harami
    if (prev["close"] < prev["open"] and 
        curr["open"] > prev["close"] and 
        curr["close"] < prev["open"] and
        curr["close"] > curr["open"]):
        return "🟢up"
    
    # Bearish harami
    elif (prev["close"] > prev["open"] and 
          curr["open"] < prev["close"] and 
          curr["close"] > prev["open"] and
          curr["close"] < curr["open"]):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_doji(candles):
    c = candles[-1]
    body = abs(c["close"] - c["open"])
    avg_range = sum(abs(c["close"] - c["open"]) for c in candles[-14:]) / 14
    
    # Standard doji
    if body < avg_range * 0.1:
        return "⚪️neutral"  # Indecision
    
    # Dragonfly doji (bullish)
    elif (body < avg_range * 0.1 and
          c["high"] - max(c["close"], c["open"]) < avg_range * 0.1 and
          min(c["close"], c["open"]) - c["low"] > avg_range * 0.5):
        return "🟢up"
    
    # Gravestone doji (bearish)
    elif (body < avg_range * 0.1 and
          min(c["close"], c["open"]) - c["low"] < avg_range * 0.1 and
          c["high"] - max(c["close"], c["open"]) > avg_range * 0.5):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_morning_star(candles):
    if len(candles) < 3:
        return "⚪️neutral"
    
    first = candles[-3]
    second = candles[-2]
    third = candles[-1]
    
    # Morning star pattern
    if (first["close"] < first["open"] and
        abs(second["close"] - second["open"]) < (first["close"] - first["open"]) * 0.3 and
        third["close"] > third["open"] and
        third["close"] > first["open"]):
        return "🟢up"
    
    return "⚪️neutral"

def strategy_evening_star(candles):
    if len(candles) < 3:
        return "⚪️neutral"
    
    first = candles[-3]
    second = candles[-2]
    third = candles[-1]
    
    # Evening star pattern
    if (first["close"] > first["open"] and
        abs(second["close"] - second["open"]) < (first["close"] - first["open"]) * 0.3 and
        third["close"] < third["open"] and
        third["close"] < first["open"]):
        return "🔴down"
    
    return "⚪️neutral"

# Price Action Strategies
def strategy_pin_bar(candles):
    c = candles[-1]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    
    # Bullish pin bar
    if (lower_wick > body * 2 and
        upper_wick < body * 0.5 and
        c["close"] > c["open"]):
        return "🟢up"
    
    # Bearish pin bar
    elif (upper_wick > body * 2 and
          lower_wick < body * 0.5 and
          c["close"] < c["open"]):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_fakey(candles):
    if len(candles) < 4:
        return "⚪️neutral"
    
    # Bullish fakey (false breakout below support)
    if (candles[-3]["close"] < candles[-3]["open"] and  # Bearish bar
        candles[-2]["high"] < candles[-3]["low"] and    # Break below
        candles[-1]["close"] > candles[-3]["high"]):    # Close back above
        return "🟢up"
    
    # Bearish fakey (false breakout above resistance)
    elif (candles[-3]["close"] > candles[-3]["open"] and  # Bullish bar
          candles[-2]["low"] > candles[-3]["high"] and     # Break above
          candles[-1]["close"] < candles[-3]["low"]):       # Close back below
        return "🔴down"
    
    return "⚪️neutral"

def strategy_three_white_soldiers(candles):
    if len(candles) < 3:
        return "⚪️neutral"
    
    first = candles[-3]
    second = candles[-2]
    third = candles[-1]
    
    # Three white soldiers pattern
    if (first["close"] > first["open"] and
        second["close"] > second["open"] and
        third["close"] > third["open"] and
        first["close"] > first["open"] * 1.01 and  # At least 1% body
        second["close"] > second["open"] * 1.01 and
        third["close"] > third["open"] * 1.01 and
        first["close"] < second["open"] and
        second["close"] < third["open"]):
        return "🟢up"
    
    return "⚪️neutral"

def strategy_three_black_crows(candles):
    if len(candles) < 3:
        return "⚪️neutral"
    
    first = candles[-3]
    second = candles[-2]
    third = candles[-1]
    
    # Three black crows pattern
    if (first["close"] < first["open"] and
        second["close"] < second["open"] and
        third["close"] < third["open"] and
        first["close"] < first["open"] * 0.99 and  # At least 1% body
        second["close"] < second["open"] * 0.99 and
        third["close"] < third["open"] * 0.99 and
        first["close"] > second["open"] and
        second["close"] > third["open"]):
        return "🔴down"
    
    return "⚪️neutral"

# Advanced Strategies
def strategy_heikin_ashi(candles):
    ha = heikin_ashi(candles)
    prev_ha = heikin_ashi(candles[:-1])
    
    # Bullish trend (green candles with little lower wick)
    if (ha["close"] > ha["open"] and
        (ha["close"] - ha["open"]) > (ha["high"] - ha["close"]) * 2):
        return "🟢up"
    
    # Bearish trend (red candles with little upper wick)
    elif (ha["close"] < ha["open"] and
          (ha["open"] - ha["close"]) > (ha["close"] - ha["low"]) * 2):
        return "🔴down"
    
    # Reversal signals
    elif (prev_ha["close"] < prev_ha["open"] and
          ha["close"] > ha["open"] and
          ha["close"] > prev_ha["open"]):
        return "🟢up"
    
    elif (prev_ha["close"] > prev_ha["open"] and
          ha["close"] < ha["open"] and
          ha["close"] < prev_ha["open"]):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_volume_weighted_ma(candles, period=20):
    vwma_numerator = 0
    vwma_denominator = 0
    for c in candles[-period:]:
        vwma_numerator += c["close"] * c["volume"]
        vwma_denominator += c["volume"]
    
    vwma = vwma_numerator / vwma_denominator
    return "🟢up" if candles[-1]["close"] > vwma else "🔴down"

def strategy_supertrend(candles, factor=3, period=10):
    a = atr(candles, period)
    mid = (candles[-1]['high'] + candles[-1]['low']) / 2
    close = candles[-1]['close']
    
    upper = mid + factor * a
    lower = mid - factor * a
    
    # Determine trend direction
    if close > upper:
        return "🟢up"
    elif close < lower:
        return "🔴down"
    return "⚪️neutral"

def strategy_ichimoku(candles):
    highs9 = max(c["high"] for c in candles[-9:])
    lows9 = min(c["low"] for c in candles[-9:])
    tenkan = (highs9 + lows9) / 2
    
    highs26 = max(c["high"] for c in candles[-26:])
    lows26 = min(c["low"] for c in candles[-26:])
    kijun = (highs26 + lows26) / 2
    
    # Senkou Span A (leading span A)
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (leading span B)
    highs52 = max(c["high"] for c in candles[-52:])
    lows52 = min(c["low"] for c in candles[-52:])
    senkou_b = (highs52 + lows52) / 2
    
    # Current price position
    current = candles[-1]["close"]
    
    # Cloud color
    cloud_green = senkou_a > senkou_b
    cloud_red = senkou_a < senkou_b
    
    # Bullish signals
    if (current > tenkan and 
        current > kijun and
        (cloud_green and current > senkou_a) or 
        (cloud_red and current > senkou_b)):
        return "🟢up"
    
    # Bearish signals
    elif (current < tenkan and 
          current < kijun and
          (cloud_green and current < senkou_b) or 
          (cloud_red and current < senkou_a)):
        return "🔴down"
    
    return "⚪️neutral"

# Smart Money Concepts
def strategy_fvg(candles):
    fvg = detect_fvg(candles)
    if fvg and fvg["type"] == "FVG Buy": return "🟢up"
    elif fvg and fvg["type"] == "FVG Sell": return "🔴down"
    return "⚪️neutral"

def strategy_ob(candles):
    ob = detect_order_block(candles)
    if ob["type"] == "Bullish OB": return "🟢up"
    elif ob["type"] == "Bearish OB": return "🔴down"
    return "⚪️neutral"

def strategy_liquidity_grab(candles):
    if len(candles) < 5:
        return "⚪️neutral"
    
    # Bullish liquidity grab (stop hunt below recent low)
    if (candles[-2]["low"] < min(c["low"] for c in candles[-5:-2]) and
        candles[-1]["close"] > max(c["high"] for c in candles[-5:-2])):
        return "🟢up"
    
    # Bearish liquidity grab (stop hunt above recent high)
    elif (candles[-2]["high"] > max(c["high"] for c in candles[-5:-2]) and
          candles[-1]["close"] < min(c["low"] for c in candles[-5:-2])):
        return "🔴down"
    
    return "⚪️neutral"

def strategy_mitigation_block(candles):
    if len(candles) < 4:
        return "⚪️neutral"
    
    # Bullish mitigation (price returns to break even after stop run)
    if (candles[-3]["close"] < candles[-3]["open"] and  # Bearish candle
        candles[-2]["low"] < candles[-3]["low"] and     # Stop run
        candles[-1]["close"] > candles[-3]["open"]):    # Back above open
        return "🟢up"
    
    # Bearish mitigation
    elif (candles[-3]["close"] > candles[-3]["open"] and  # Bullish candle
          candles[-2]["high"] > candles[-3]["high"] and    # Stop run
          candles[-1]["close"] < candles[-3]["open"]):     # Back below open
        return "🔴down"
    
    return "⚪️neutral"

# Multi-Timeframe Strategies
def strategy_higher_timeframe_confirmation(candles):
    # Get 5-minute candles for higher timeframe confirmation
    higher_tf_candles = get_candles(interval=Client.KLINE_INTERVAL_5MINUTE)
    
    if len(higher_tf_candles) < 10:
        return "⚪️neutral"
    
    # Check if higher timeframe is bullish
    higher_tf_bullish = (higher_tf_candles[-1]["close"] > higher_tf_candles[-1]["open"] and
                         higher_tf_candles[-1]["close"] > ema([c["close"] for c in higher_tf_candles[-20:]], 20))
    
    # Check if higher timeframe is bearish
    higher_tf_bearish = (higher_tf_candles[-1]["close"] < higher_tf_candles[-1]["open"] and
                         higher_tf_candles[-1]["close"] < ema([c["close"] for c in higher_tf_candles[-20:]], 20))
    
    # Only trade in direction of higher timeframe
    current_strategy = strategy_ema_cross(candles)
    if higher_tf_bullish and current_strategy == "🟢up":
        return "🟢up"
    elif higher_tf_bearish and current_strategy == "🔴down":
        return "🔴down"
    return "⚪️neutral"

# ========== STRATEGY RUNNER + DECISION ==========
def run_all_strategies(candles):
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
        "High/Low Breakout": strategy_high_low_breakout(candles),
        "Fractal Breakout": strategy_fractal_breakout(candles),
        
        # Momentum
        "Momentum": strategy_momentum(candles),
        "ROC": strategy_roc(candles),
        "Williams %R": strategy_williams_r(candles),
        "Ultimate Oscillator": strategy_ultimate_oscillator(candles),
        "OBV": strategy_obv(candles),
        
        # Volume
        "Volume Spike": strategy_volume_spike(candles),
        "VWAP+MACD Combo": strategy_vwap_macd_combo(candles),
        "Volume Profile": strategy_volume_profile(candles),
        "Accumulation/Distribution": strategy_accumulation_distribution(candles),
        
        # Candlestick Patterns
        "Wick Rejection": strategy_wick_rejection(candles),
        "Engulfing": strategy_engulfing(candles),
        "Hammer/Hanging Man": strategy_hammer(candles),
        "Harami": strategy_harami(candles),
        "Doji": strategy_doji(candles),
        "Morning Star": strategy_morning_star(candles),
        "Evening Star": strategy_evening_star(candles),
        "Pin Bar": strategy_pin_bar(candles),
        "Fakey": strategy_fakey(candles),
        "3 White Soldiers": strategy_three_white_soldiers(candles),
        "3 Black Crows": strategy_three_black_crows(candles),
        
        # Price Action
        "Heikin Ashi": strategy_heikin_ashi(candles),
        "VWMA": strategy_volume_weighted_ma(candles),
        "Supertrend": strategy_supertrend(candles),
        "Ichimoku": strategy_ichimoku(candles),
        
        # Smart Money Concepts
        "FVG": strategy_fvg(candles),
        "Order Block": strategy_ob(candles),
        "Liquidity Grab": strategy_liquidity_grab(candles),
        "Mitigation Block": strategy_mitigation_block(candles),
        
        # Multi-Timeframe
        "HTF Confirmation": strategy_higher_timeframe_confirmation(candles),
        
        # Core Strategies
        "ICT": ict(candles),
        "SMC": smc(candles)
    }
    return strategies

def aggregate_signals(signals):
    votes = {"🟢up": 0, "🔴down": 0, "⚪️neutral": 0}
    for decision in signals.values():
        votes[decision] += 1
    if votes["🟢up"] > votes["🔴down"]:
        return "BUY ✅", votes
    elif votes["🔴down"] > votes["🟢up"]:
        return "SELL ❌", votes
    else:
        return "WAIT 🕒", votes

# ========== ALERT GENERATOR ==========

def generate_telegram_message(candles, signals, votes, decision, prediction):
    entry = candles[-1]["close"]
    atr_val = atr(candles, 14)
    fib_tp = fib_extension(entry, atr_val, "BUY" if decision == "BUY ✅" else "SELL")
    stop_loss = round(entry - atr_val, 2) if decision == "BUY ✅" else round(entry + atr_val, 2)

    ob = detect_order_block(candles)
    fvg = detect_fvg(candles)
    chart_pattern = analyze_chart_pattern(candles)

    # Group signals by category for better readability
    trend_signals = {k:v for k,v in signals.items() if k in [
        "EMA Crossover", "MACD", "ADX", "Parabolic SAR", "Keltner Channels", 
        "TRIX", "Awesome Oscillator", "Supertrend", "Ichimoku"
    ]}
    
    reversal_signals = {k:v for k,v in signals.items() if k in [
        "RSI", "Bollinger Bands", "Stochastic", "CCI", "MFI", "VWAP"
    ]}
    
    pattern_signals = {k:v for k,v in signals.items() if k in [
        "Wick Rejection", "Engulfing", "Hammer/Hanging Man", "Harami", 
        "Doji", "Morning Star", "Evening Star", "Pin Bar", "Fakey",
        "3 White Soldiers", "3 Black Crows"
    ]}
    
    smart_money_signals = {k:v for k,v in signals.items() if k in [
        "FVG", "Order Block", "Liquidity Grab", "Mitigation Block", "ICT", "SMC"
    ]}

    msg = f"📊 *BTCUSDT 1-min Strategy Summary*\n\n"
    msg += f"🧠 *Majority Decision*: {decision} (🟢{votes['🟢up']} | 🔴{votes['🔴down']} | ⚪️{votes['⚪️neutral']})\n"
    msg += f"📈 *15-Candle Forecast*: {prediction}\n"
    msg += f"🎯 *Entry*: `{entry:.2f}` | 🛑 SL: `{stop_loss}` | 🎯 TP (Fib 1.618): `{fib_tp}`\n\n"
    
    # Add trend signals
    msg += "📈 *Trend Signals*\n"
    for name, signal in trend_signals.items():
        if signal != "⚪️neutral":
            msg += f"• {name}: {signal}\n"
    
    # Add reversal signals
    msg += "\n🔄 *Reversal Signals*\n"
    for name, signal in reversal_signals.items():
        if signal != "⚪️neutral":
            msg += f"• {name}: {signal}\n"
    
    # Add pattern signals
    msg += "\n🕯️ *Candle Patterns*\n"
    for name, signal in pattern_signals.items():
        if signal != "⚪️neutral":
            msg += f"• {name}: {signal}\n"
    
    # Add smart money signals
    msg += "\n💡 *Smart Money Signals*\n"
    for name, signal in smart_money_signals.items():
        if signal != "⚪️neutral":
            msg += f"• {name}: {signal}\n"
    
    if ob:
        msg += f"\n🧱 OB: {ob['type']} @ {ob['price']:.2f}"
    if fvg:
        zone = fvg["zone"]
        msg += f"\n📐 FVG: {fvg['type']} Zone {zone[0]:.2f} → {zone[1]:.2f}"
    
    # Add chart pattern analysis
    msg += f"\n\n📉 *Chart Pattern Analysis*:"
    if len(chart_pattern["levels"]) >= 4:
        msg += f"\n1️⃣ 1st Horizontal Ray: {chart_pattern['levels'][0]:.2f}"
        msg += f"\n2️⃣ 2nd Horizontal Ray: {chart_pattern['levels'][1]:.2f}"
        msg += f"\n3️⃣ 3rd Horizontal Ray: {chart_pattern['levels'][2]:.2f}"
        msg += f"\n4️⃣ 4th Horizontal Ray: {chart_pattern['levels'][3]:.2f}"
        msg += f"\n🔍 Pattern Detected: {chart_pattern['pattern']}"
        msg += f"\n📍 {chart_pattern['current_position']}"
    else:
        msg += "\n⚠️ Not enough levels detected for pattern analysis"

    msg += "\n\n📣 Powered by Smart Multi-Strategy AI 🔍"
    return msg

# ========== MASTER ANALYSIS FUNCTION ==========
def analyze():
    candles = get_candles()
    if len(candles) < 30:
        logging.warning("Not enough candles.")
        return

    strategies = run_all_strategies(candles)
    decision, votes = aggregate_signals(strategies)
    prediction = predict_next_15(candles)
    message = generate_telegram_message(candles, strategies, votes, decision, prediction)

    bot.send_message(chat_id=TELEGRAM_CHANNEL, text=message, parse_mode="Markdown")

# ========== EXECUTION LOOP + FLASK SERVER ==========

def run_scheduler():
    while True:
        try:
            analyze()
        except Exception as e:
            logging.error(f"[ERROR] {e}")
        time.sleep(60)

@app.route('/')
def health_check():
    return "🔁 BTCUSDT Strategy Bot is Running."

if __name__ == "__main__":
    threading.Thread(target=run_scheduler).start()
    app.run(host="0.0.0.0", port=5000)
