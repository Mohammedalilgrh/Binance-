import os
import time
import threading
import logging
import random
from datetime import datetime
from flask import Flask, jsonify
from binance.client import Client
from telegram import Bot, ParseMode

# =================== CONFIG ===================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg")
TELEGRAM_CHANNEL = os.getenv("TELEGRAM_CHANNEL", "@tradegrh")
SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LOOKBACK_PERIOD = 100  # Number of candles to analyze
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to take action

# Initialize clients
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# [Previous utility functions remain exactly the same...]

# =================== TRADE EXECUTION LOGIC ===================
def calculate_trade_parameters(data, strategy_results):
    """Calculate optimal entry, TP, SL based on strategy consensus"""
    current_price = data[-1]['close']
    atr = calculate_atr(data)
    fib_levels = calculate_fibonacci_levels(data)
    fvg = detect_fvg(data)
    ob = detect_order_blocks(data)
    
    # Determine direction
    up_count = sum(1 for result in strategy_results.values() if "up" in result[0].lower())
    down_count = sum(1 for result in strategy_results.values() if "down" in result[0].lower())
    
    if up_count > down_count and up_count >= len(strategy_results) * CONFIDENCE_THRESHOLD:
        direction = "LONG"
        # Entry: Current price or slightly above
        entry = current_price * 1.0005
        
        # Take Profit: Near next resistance or fib extension
        tp_candidates = [
            fib_levels['1.618'],
            detect_support_resistance(data)['resistance'],
            current_price + (3 * atr)
        ]
        tp = min(tp for tp in tp_candidates if tp > current_price)
        
        # Stop Loss: Below recent swing low or FVG/OB
        sl_candidates = [
            fib_levels['0.618'],
            detect_support_resistance(data)['support'],
            current_price - (1.5 * atr),
            ob['bullish'][0] if ob['bullish'] else None,
            fvg['bullish'][0] if fvg['bullish'] else None
        ]
        sl = max(sl for sl in sl_candidates if sl and sl < current_price)
        
    elif down_count > up_count and down_count >= len(strategy_results) * CONFIDENCE_THRESHOLD:
        direction = "SHORT"
        # Entry: Current price or slightly below
        entry = current_price * 0.9995
        
        # Take Profit: Near next support or fib extension
        tp_candidates = [
            fib_levels['1.618'],
            detect_support_resistance(data)['support'],
            current_price - (3 * atr)
        ]
        tp = max(tp for tp in tp_candidates if tp and tp < current_price)  # Ensure TP is below entry
        
        # Stop Loss: Above recent swing high or FVG/OB
        sl_candidates = [
            fib_levels['0.618'],
            detect_support_resistance(data)['resistance'],
            current_price + (1.5 * atr),
            ob['bearish'][1] if ob['bearish'] else None,
            fvg['bearish'][1] if fvg['bearish'] else None
        ]
        sl = min(sl for sl in sl_candidates if sl and sl > current_price)  # Ensure SL is above entry
        
        # Additional validation for BEARISH trades
        if tp >= entry:
            # If TP is above entry, adjust it to be at least 1 ATR below entry
            tp = entry - (1 * atr)
            logging.warning(f"Adjusted TP to be below entry price for BEARISH trade. New TP: {tp}")
            
        if sl <= entry:
            # If SL is below entry, adjust it to be at least 1 ATR above entry
            sl = entry + (1 * atr)
            logging.warning(f"Adjusted SL to be above entry price for BEARISH trade. New SL: {sl}")
            
    else:
        direction = "NEUTRAL"
        entry = tp = sl = None
    
    return direction, entry, tp, sl

# [Rest of the code remains exactly the same...]


# =================== UTILITY FUNCTIONS ===================
def get_historical_data():
    """Fetch historical candle data from Binance"""
    candles = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LOOKBACK_PERIOD)
    return [{
        'time': candle[0],
        'open': float(candle[1]),
        'high': float(candle[2]),
        'low': float(candle[3]),
        'close': float(candle[4]),
        'volume': float(candle[5])
    } for candle in candles]

def calculate_ema(data, period):
    """Calculate Exponential Moving Average"""
    multiplier = 2 / (period + 1)
    ema = data[0]['close']
    for candle in data[1:]:
        ema = (candle['close'] - ema) * multiplier + ema
    return ema

def calculate_vwap(data):
    """Calculate Volume Weighted Average Price"""
    total_pv = sum((candle['high'] + candle['low'] + candle['close']) / 3 * candle['volume'] for candle in data)
    total_volume = sum(candle['volume'] for candle in data)
    return total_pv / total_volume if total_volume != 0 else 0

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    gains = []
    losses = []
    for i in range(1, len(data)):
        change = data[i]['close'] - data[i-1]['close']
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    
    avg_gain = sum(gains[:period]) / period if len(gains) >= period else 0
    avg_loss = sum(losses[:period]) / period if len(losses) >= period else 0
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
    for i in range(period, len(losses)):
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    """Calculate MACD (12, 26, 9)"""
    if len(data) < 26:
        return 0, 0, 0
    
    ema12 = calculate_ema(data[-12:], 12)
    ema26 = calculate_ema(data[-26:], 26)
    macd_line = ema12 - ema26
    signal_line = calculate_ema([{'close': macd_line}]*9, 9)  # Simplified signal line
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    closes = [candle['close'] for candle in data[-period:]]
    if len(closes) < period:
        return 0, 0, 0
    
    middle_band = sum(closes) / period
    std = (sum((x - middle_band) ** 2 for x in closes) / period) ** 0.5
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std
    return upper_band, middle_band, lower_band

def calculate_stochastic_rsi(data, period=14, k_period=3, d_period=3):
    """Calculate Stochastic RSI"""
    if len(data) < period + k_period + d_period:
        return 0, 0
    
    rsi_values = []
    for i in range(len(data) - period + 1):
        rsi_values.append(calculate_rsi(data[i:i+period], period))
    
    k_values = []
    for i in range(len(rsi_values) - k_period + 1):
        current_rsi = rsi_values[i:i+k_period]
        lowest = min(current_rsi)
        highest = max(current_rsi)
        k = 100 * (rsi_values[i+k_period-1] - lowest) / (highest - lowest) if (highest - lowest) != 0 else 0
        k_values.append(k)
    
    d_values = []
    for i in range(len(k_values) - d_period + 1):
        d_values.append(sum(k_values[i:i+d_period]) / d_period)
    
    return k_values[-1] if k_values else 0, d_values[-1] if d_values else 0

def calculate_supertrend(data, period=7, multiplier=3):
    """Calculate Supertrend"""
    if len(data) < period:
        return []
    
    hl2 = [(candle['high'] + candle['low']) / 2 for candle in data]
    atr = sum(max(candle['high'] - candle['low'], 
                 abs(candle['high'] - data[i-1]['close']), 
                 abs(candle['low'] - data[i-1]['close'])) 
             for i, candle in enumerate(data[1:], 1)) / (len(data) - 1)
    
    upper_band = hl2[-1] + multiplier * atr
    lower_band = hl2[-1] - multiplier * atr
    
    trend = []
    for i in range(len(data)):
        if i == 0:
            trend.append('up' if data[i]['close'] > upper_band else 'down')
        else:
            if trend[-1] == 'up' and data[i]['close'] > lower_band:
                trend.append('up')
            elif trend[-1] == 'down' and data[i]['close'] < upper_band:
                trend.append('down')
            else:
                trend.append('up' if data[i]['close'] > upper_band else 'down')
    
    return trend[-1]

def calculate_ichimoku(data):
    """Calculate Ichimoku Cloud"""
    if len(data) < 26:
        return 0, 0, 0, 0, 0
    
    # Tenkan-sen (Conversion Line)
    period9_high = max(candle['high'] for candle in data[-9:])
    period9_low = min(candle['low'] for candle in data[-9:])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line)
    period26_high = max(candle['high'] for candle in data[-26:])
    period26_low = min(candle['low'] for candle in data[-26:])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B)
    period52_high = max(candle['high'] for candle in data[-52:]) if len(data) >= 52 else 0
    period52_low = min(candle['low'] for candle in data[-52:]) if len(data) >= 52 else 0
    senkou_span_b = (period52_high + period52_low) / 2 if period52_high and period52_low else 0
    
    # Chikou Span (Lagging Span)
    chikou_span = data[-26]['close'] if len(data) >= 26 else 0
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_cci(data, period=20):
    """Calculate Commodity Channel Index"""
    if len(data) < period:
        return 0
    
    typical_prices = [(candle['high'] + candle['low'] + candle['close']) / 3 for candle in data[-period:]]
    sma = sum(typical_prices) / period
    mean_deviation = sum(abs(tp - sma) for tp in typical_prices) / period
    return (typical_prices[-1] - sma) / (0.015 * mean_deviation) if mean_deviation != 0 else 0

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    if len(data) < period + 1:
        return 0
    
    true_ranges = []
    for i in range(1, len(data)):
        high = data[i]['high']
        low = data[i]['low']
        prev_close = data[i-1]['close']
        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(true_range)
    
    return sum(true_ranges[-period:]) / period

def detect_support_resistance(data):
    """Detect key support and resistance levels"""
    closes = [candle['close'] for candle in data]
    highs = [candle['high'] for candle in data]
    lows = [candle['low'] for candle in data]
    
    # Simple pivot points
    resistance = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    support = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    
    # Round numbers as psychological levels
    current_price = closes[-1]
    round_numbers = [round(current_price / 100) * 100,
                    round(current_price / 50) * 50,
                    round(current_price / 20) * 20]
    
    return {
        'support': support,
        'resistance': resistance,
        'round_numbers': round_numbers
    }

def detect_trendlines(data):
    """Detect simple trendlines"""
    if len(data) < 10:
        return {'up': None, 'down': None}
    
    # Simple trendline detection (last 10 candles)
    highs = [candle['high'] for candle in data[-10:]]
    lows = [candle['low'] for candle in data[-10:]]
    
    # Up trendline (connect two most recent higher lows)
    up_trendline = None
    if lows[-1] > lows[-2]:
        up_trendline = (lows[-2], lows[-1])
    
    # Down trendline (connect two most recent lower highs)
    down_trendline = None
    if highs[-1] < highs[-2]:
        down_trendline = (highs[-2], highs[-1])
    
    return {'up': up_trendline, 'down': down_trendline}

def detect_chart_patterns(data):
    """Detect basic chart patterns"""
    patterns = {
        'double_top': False,
        'double_bottom': False,
        'inside_bar': False,
        'wick_rejection': False
    }
    
    if len(data) < 5:
        return patterns
    
    # Double Top/Bottom
    if (data[-3]['high'] < data[-2]['high'] and 
        data[-2]['high'] > data[-1]['high'] and
        abs(data[-2]['high'] - data[-4]['high']) < 0.005 * data[-2]['high']):
        patterns['double_top'] = True
    
    if (data[-3]['low'] > data[-2]['low'] and 
        data[-2]['low'] < data[-1]['low'] and
        abs(data[-2]['low'] - data[-4]['low']) < 0.005 * data[-2]['low']):
        patterns['double_bottom'] = True
    
    # Inside Bar
    if (data[-1]['high'] < data[-2]['high'] and 
        data[-1]['low'] > data[-2]['low']):
        patterns['inside_bar'] = True
    
    # Wick Rejection
    if (data[-1]['close'] > data[-1]['open'] and 
        (data[-1]['high'] - data[-1]['close']) > 2 * (data[-1]['close'] - data[-1]['open'])):
        patterns['wick_rejection'] = True
    elif (data[-1]['close'] < data[-1]['open'] and 
          (data[-1]['low'] - data[-1]['close']) > 2 * (data[-1]['open'] - data[-1]['close'])):
        patterns['wick_rejection'] = True
    
    return patterns

def detect_volume_spike(data):
    """Detect volume spikes"""
    if len(data) < 20:
        return False
    
    current_volume = data[-1]['volume']
    avg_volume = sum(candle['volume'] for candle in data[-20:-1]) / 19
    return current_volume > 2 * avg_volume

def detect_liquidity_zones(data):
    """Detect liquidity zones (simplified)"""
    highs = [candle['high'] for candle in data[-50:]]
    lows = [candle['low'] for candle in data[-50:]]
    
    # Recent highs/lows that could be liquidity zones
    liquidity_zones = {
        'highs': sorted(list(set([h for h in highs if highs.count(h) > 1]))),
        'lows': sorted(list(set([l for l in lows if lows.count(l) > 1])))
    }
    
    return liquidity_zones

def detect_fvg(data):
    """Detect Fair Value Gaps (ICT concept)"""
    if len(data) < 3:
        return {'bullish': None, 'bearish': None}
    
    # Bullish FVG (current candle low > previous candle high)
    bullish_fvg = None
    if data[-1]['low'] > data[-2]['high']:
        bullish_fvg = (data[-2]['high'], data[-1]['low'])
    
    # Bearish FVG (current candle high < previous candle low)
    bearish_fvg = None
    if data[-1]['high'] < data[-2]['low']:
        bearish_fvg = (data[-2]['low'], data[-1]['high'])
    
    return {'bullish': bullish_fvg, 'bearish': bearish_fvg}

def detect_order_blocks(data):
    """Detect Order Blocks (simplified ICT concept)"""
    if len(data) < 5:
        return {'bullish': None, 'bearish': None}
    
    # Bullish OB (strong bullish candle with high volume)
    bullish_ob = None
    if (data[-2]['close'] > data[-2]['open'] * 1.01 and 
        data[-2]['volume'] > sum(candle['volume'] for candle in data[-5:-2]) / 3):
        bullish_ob = (data[-2]['low'], data[-2]['high'])
    
    # Bearish OB (strong bearish candle with high volume)
    bearish_ob = None
    if (data[-2]['close'] < data[-2]['open'] * 0.99 and 
        data[-2]['volume'] > sum(candle['volume'] for candle in data[-5:-2]) / 3):
        bearish_ob = (data[-2]['high'], data[-2]['low'])
    
    return {'bullish': bullish_ob, 'bearish': bearish_ob}

def calculate_fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    if len(data) < 20:
        return {}
    
    swing_high = max(candle['high'] for candle in data[-20:])
    swing_low = min(candle['low'] for candle in data[-20:])
    price_range = swing_high - swing_low
    
    return {
        '0': swing_high,
        '0.236': swing_high - price_range * 0.236,
        '0.382': swing_high - price_range * 0.382,
        '0.5': swing_high - price_range * 0.5,
        '0.618': swing_high - price_range * 0.618,
        '0.786': swing_high - price_range * 0.786,
        '1': swing_low,
        '1.618': swing_low - price_range * 0.618
    }

# =================== STRATEGY FUNCTIONS ===================
def ema_crossover_strategy(data):
    """9 EMA & 21 EMA Crossover Strategy"""
    ema9 = calculate_ema(data[-9:], 9)
    ema21 = calculate_ema(data[-21:], 21)
    
    if ema9 > ema21 and data[-2]['close'] <= calculate_ema(data[-22:-1], 21):
        return "🟢up", 0.8
    elif ema9 < ema21 and data[-2]['close'] >= calculate_ema(data[-22:-1], 21):
        return "🔴down", 0.8
    return "🟡neutral", 0.5

def vwap_strategy(data):
    """VWAP Bounce or Rejection Strategy"""
    vwap = calculate_vwap(data)
    current_price = data[-1]['close']
    
    if current_price > vwap and data[-1]['close'] > data[-1]['open']:
        return "🟢up", 0.75
    elif current_price < vwap and data[-1]['close'] < data[-1]['open']:
        return "🔴down", 0.75
    return "🟡neutral", 0.5

def bollinger_band_strategy(data):
    """Bollinger Band Breakout (Squeeze Strategy)"""
    upper, middle, lower = calculate_bollinger_bands(data)
    current_price = data[-1]['close']
    
    if current_price > upper:
        return "🟢up", 0.7
    elif current_price < lower:
        return "🔴down", 0.7
    
    # Squeeze detection
    if (upper - lower) < 0.005 * middle:  # Very tight bands
        if current_price > middle:
            return "🟢up", 0.65
        else:
            return "🔴down", 0.65
    
    return "🟡neutral", 0.5

def rsi_strategy(data):
    """RSI Quick Reversal Strategy"""
    rsi = calculate_rsi(data)
    current_price = data[-1]['close']
    
    if rsi < 30 and current_price > data[-1]['open']:
        return "🟢up", 0.75
    elif rsi > 70 and current_price < data[-1]['open']:
        return "🔴down", 0.75
    return "🟡neutral", 0.5

def macd_strategy(data):
    """MACD Line Crossover with Histogram Confirmation"""
    macd_line, signal_line, histogram = calculate_macd(data)
    
    if macd_line > signal_line and histogram > 0:
        return "🟢up", 0.8
    elif macd_line < signal_line and histogram < 0:
        return "🔴down", 0.8
    return "🟡neutral", 0.5

def stochastic_rsi_strategy(data):
    """Stochastic RSI Overbought/Oversold Flip"""
    k, d = calculate_stochastic_rsi(data)
    
    if k < 20 and d < 20 and k > d:
        return "🟢up", 0.7
    elif k > 80 and d > 80 and k < d:
        return "🔴down", 0.7
    return "🟡neutral", 0.5

def supertrend_strategy(data):
    """Supertrend Flip Strategy"""
    trend = calculate_supertrend(data)
    return ("🟢up", 0.75) if trend == 'up' else ("🔴down", 0.75)

def ichimoku_strategy(data):
    """Ichimoku Tenkan/Kijun Cross Strategy"""
    tenkan, kijun, _, _, _ = calculate_ichimoku(data)
    
    if tenkan > kijun:
        return "🟢up", 0.7
    elif tenkan < kijun:
        return "🔴down", 0.7
    return "🟡neutral", 0.5

def cci_strategy(data):
    """CCI Spike Strategy"""
    cci = calculate_cci(data)
    
    if cci > 100:
        return "🟢up", 0.7
    elif cci < -100:
        return "🔴down", 0.7
    return "🟡neutral", 0.5

def atr_strategy(data):
    """ATR-Based Dynamic Stop-Loss & Take-Profit Strategy"""
    atr = calculate_atr(data)
    current_price = data[-1]['close']
    volatility_ratio = atr / current_price
    
    if volatility_ratio < 0.005:  # Low volatility
        return "🟡neutral", 0.5
    elif data[-1]['close'] > data[-1]['open'] and volatility_ratio > 0.01:
        return "🟢up", 0.6
    elif data[-1]['close'] < data[-1]['open'] and volatility_ratio > 0.01:
        return "🔴down", 0.6
    return "🟡neutral", 0.5

def support_resistance_strategy(data):
    """Support & Resistance Scalping Strategy"""
    levels = detect_support_resistance(data)
    current_price = data[-1]['close']
    
    if current_price > levels['resistance'] * 0.999:  # Breaking resistance
        return "🟢up", 0.8
    elif current_price < levels['support'] * 1.001:  # Breaking support
        return "🔴down", 0.8
    return "🟡neutral", 0.5

def trendline_strategy(data):
    """Trendline Break + Retest Entry Strategy"""
    trendlines = detect_trendlines(data)
    current_price = data[-1]['close']
    
    if trendlines['up'] and current_price > trendlines['up'][1]:
        return "🟢up", 0.75
    elif trendlines['down'] and current_price < trendlines['down'][1]:
        return "🔴down", 0.75
    return "🟡neutral", 0.5

def inside_bar_strategy(data):
    """Inside Bar Breakout Strategy"""
    patterns = detect_chart_patterns(data)
    
    if patterns['inside_bar']:
        if data[-1]['close'] > data[-2]['high']:
            return "🟢up", 0.7
        elif data[-1]['close'] < data[-2]['low']:
            return "🔴down", 0.7
    return "🟡neutral", 0.5

def double_top_bottom_strategy(data):
    """Micro Double Top / Double Bottom Strategy"""
    patterns = detect_chart_patterns(data)
    
    if patterns['double_top']:
        return "🔴down", 0.7
    elif patterns['double_bottom']:
        return "🟢up", 0.7
    return "🟡neutral", 0.5

def wick_rejection_strategy(data):
    """Candle Wick Rejection Entry Strategy"""
    patterns = detect_chart_patterns(data)
    volume_spike = detect_volume_spike(data)
    
    if patterns['wick_rejection'] and volume_spike:
        if data[-1]['close'] > data[-1]['open']:  # Bullish rejection
            return "🟢up", 0.75
        else:  # Bearish rejection
            return "🔴down", 0.75
    return "🟡neutral", 0.5

def volume_spike_strategy(data):
    """Volume Spike Breakout Strategy"""
    volume_spike = detect_volume_spike(data)
    current_price = data[-1]['close']
    
    if volume_spike:
        if current_price > data[-1]['open']:
            return "🟢up", 0.7
        else:
            return "🔴down", 0.7
    return "🟡neutral", 0.5

def break_retest_strategy(data):
    """Break & Retest at Key Levels Strategy"""
    levels = detect_support_resistance(data)
    current_price = data[-1]['close']
    
    # Check if price retested a broken level
    if (current_price > levels['resistance'] * 0.998 and 
        any(candle['low'] < levels['resistance'] * 1.002 for candle in data[-3:])):
        return "🟢up", 0.8
    elif (current_price < levels['support'] * 1.002 and 
          any(candle['high'] > levels['support'] * 0.998 for candle in data[-3:])):
        return "🔴down", 0.8
    return "🟡neutral", 0.5

def liquidity_grab_strategy(data):
    """Liquidity Grab/Fakeout Reversal Strategy"""
    liquidity_zones = detect_liquidity_zones(data)
    current_price = data[-1]['close']
    
    # Check if price took liquidity (went beyond a zone and reversed)
    for zone in liquidity_zones['highs']:
        if any(candle['high'] > zone * 1.001 for candle in data[-5:-1]) and current_price < zone * 0.999:
            return "🔴down", 0.75
    
    for zone in liquidity_zones['lows']:
        if any(candle['low'] < zone * 0.999 for candle in data[-5:-1]) and current_price > zone * 1.001:
            return "🟢up", 0.75
    
    return "🟡neutral", 0.5

def risk_reward_strategy(data):
    """1:2 Risk-Reward Scalping Model"""
    # This is more about position sizing than direction
    return "🟡neutral", 0.5

def time_based_strategy(data):
    """Time-Based Exit Strategy"""
    # This is about exit timing rather than direction
    return "🟡neutral", 0.5

def fixed_pip_strategy(data):
    """Fixed Pip/Point Scalping Strategy"""
    # This is about position sizing rather than direction
    return "🟡neutral", 0.5

def ema_rsi_strategy(data):
    """EMA + RSI Confluence Entry Strategy"""
    ema20 = calculate_ema(data[-20:], 20)
    rsi = calculate_rsi(data)
    current_price = data[-1]['close']
    
    if current_price > ema20 and rsi > 50:
        return "🟢up", 0.7
    elif current_price < ema20 and rsi < 50:
        return "🔴down", 0.7
    return "🟡neutral", 0.5

def vwap_macd_strategy(data):
    """VWAP + MACD Combo Scalping Strategy"""
    vwap = calculate_vwap(data)
    macd_line, signal_line, _ = calculate_macd(data)
    current_price = data[-1]['close']
    
    if current_price > vwap and macd_line > signal_line:
        return "🟢up", 0.75
    elif current_price < vwap and macd_line < signal_line:
        return "🔴down", 0.75
    return "🟡neutral", 0.5

def bollinger_rsi_strategy(data):
    """Bollinger Band + RSI Divergence Reversal Strategy"""
    upper, _, lower = calculate_bollinger_bands(data)
    rsi = calculate_rsi(data)
    current_price = data[-1]['close']
    
    if current_price < lower and rsi > 30:
        return "🟢up", 0.7
    elif current_price > upper and rsi < 70:
        return "🔴down", 0.7
    return "🟡neutral", 0.5

def ict_strategy(data):
    """ICT (Inner Circle Trader) Strategy"""
    fvg = detect_fvg(data)
    ob = detect_order_blocks(data)
    current_price = data[-1]['close']
    
    # ICT Power of 3 (Market Structure Shift)
    mss_bullish = (current_price > max(candle['high'] for candle in data[-3:-1])) and (data[-1]['volume'] > sum(candle['volume'] for candle in data[-10:-1]) / 9)
    mss_bearish = (current_price < min(candle['low'] for candle in data[-3:-1])) and (data[-1]['volume'] > sum(candle['volume'] for candle in data[-10:-1]) / 9)
    
    # ICT Kill Zones (New York/London session)
    current_hour = datetime.now().hour
    ny_kill_zone = 8 <= current_hour <= 11  # NY AM
    london_kill_zone = 2 <= current_hour <= 5  # London AM
    
    # Decision
    if (mss_bullish or fvg['bullish'] or ob['bullish']) and (ny_kill_zone or london_kill_zone):
        return "🟢up", 0.85
    elif (mss_bearish or fvg['bearish'] or ob['bearish']) and (ny_kill_zone or london_kill_zone):
        return "🔴down", 0.85
    return "🟡neutral", 0.5

def smc_strategy(data):
    """Smart Money Concept (SMC) Strategy"""
    # Break of Structure (BOS)
    bos_bullish = (data[-1]['high'] > max(candle['high'] for candle in data[-5:-1])) and (data[-1]['close'] > data[-1]['open'])
    bos_bearish = (data[-1]['low'] < min(candle['low'] for candle in data[-5:-1])) and (data[-1]['close'] < data[-1]['open'])
    
    # Change of Character (CHOCH)
    choch_bullish = (data[-1]['low'] > min(candle['low'] for candle in data[-3:-1])) and (data[-1]['close'] > data[-1]['open'])
    choch_bearish = (data[-1]['high'] < max(candle['high'] for candle in data[-3:-1])) and (data[-1]['close'] < data[-1]['open'])
    
    # Liquidity Grab
    liquidity_bullish = (data[-1]['low'] < min(candle['low'] for candle in data[-10:-1])) and (data[-1]['close'] > data[-1]['open'])
    liquidity_bearish = (data[-1]['high'] > max(candle['high'] for candle in data[-10:-1])) and (data[-1]['close'] < data[-1]['open'])
    
    if bos_bullish or choch_bullish or liquidity_bullish:
        return "🟢up", 0.85
    elif bos_bearish or choch_bearish or liquidity_bearish:
        return "🔴down", 0.85
    return "🟡neutral", 0.5

# =================== PREDICTION FUNCTIONS ===================
def predict_next_15_candles(data, strategy_results):
    """Predict the next 15 candles based on strategy consensus"""
    current_price = data[-1]['close']
    direction = None
    confidence = 0
    
    # Calculate weighted direction based on all strategies
    up_score = 0
    down_score = 0
    total_weight = 0
    
    for strategy, (signal, conf) in strategy_results.items():
        if "up" in signal:
            up_score += conf
        elif "down" in signal:
            down_score += conf
        total_weight += conf
    
    if total_weight > 0:
        up_prob = up_score / total_weight
        down_prob = down_score / total_weight
        
        if up_prob > down_prob and up_prob > CONFIDENCE_THRESHOLD:
            direction = "up"
            confidence = up_prob
        elif down_prob > up_prob and down_prob > CONFIDENCE_THRESHOLD:
            direction = "down"
            confidence = down_prob
    
    # Generate prediction
    prediction = []
    atr = calculate_atr(data)
    
    if direction == "up":
        for i in range(1, 16):
            # Simple linear projection with some randomness
            price = current_price * (1 + (i * 0.001 * confidence)) + (atr * 0.1 * i)
            prediction.append({
                'time': datetime.now().timestamp() + (i * 60),
                'price': price,
                'trend': 'up',
                'confidence': confidence * (1 - (i * 0.05))  # Confidence decreases with time
            })
    elif direction == "down":
        for i in range(1, 16):
            price = current_price * (1 - (i * 0.001 * confidence)) - (atr * 0.1 * i)
            prediction.append({
                'time': datetime.now().timestamp() + (i * 60),
                'price': price,
                'trend': 'down',
                'confidence': confidence * (1 - (i * 0.05))
            })
    else:
        # No clear direction, predict sideways movement
        for i in range(1, 16):
            price = current_price * (1 + (random.uniform(-0.0005, 0.0005) * i))
            prediction.append({
                'time': datetime.now().timestamp() + (i * 60),
                'price': price,
                'trend': 'neutral',
                'confidence': 0.5 * (1 - (i * 0.05))
            })
    
    return prediction

# =================== TRADE EXECUTION LOGIC ===================
def calculate_trade_parameters(data, strategy_results):
    """Calculate optimal entry, TP, SL based on strategy consensus"""
    current_price = data[-1]['close']
    atr = calculate_atr(data)
    fib_levels = calculate_fibonacci_levels(data)
    fvg = detect_fvg(data)
    ob = detect_order_blocks(data)
    
    # Determine direction
    up_count = sum(1 for result in strategy_results.values() if "up" in result[0].lower())
    down_count = sum(1 for result in strategy_results.values() if "down" in result[0].lower())
    
    if up_count > down_count and up_count >= len(strategy_results) * CONFIDENCE_THRESHOLD:
        direction = "LONG"
        # Entry: Current price or slightly above
        entry = current_price * 1.0005
        
        # Take Profit: Near next resistance or fib extension
        tp_candidates = [
            fib_levels['1.618'],
            detect_support_resistance(data)['resistance'],
            current_price + (3 * atr)
        ]
        tp = min(tp for tp in tp_candidates if tp > current_price)
        
        # Stop Loss: Below recent swing low or FVG/OB
        sl_candidates = [
            fib_levels['0.618'],
            detect_support_resistance(data)['support'],
            current_price - (1.5 * atr),
            ob['bullish'][0] if ob['bullish'] else None,
            fvg['bullish'][0] if fvg['bullish'] else None
        ]
        sl = max(sl for sl in sl_candidates if sl and sl < current_price)
        
    elif down_count > up_count and down_count >= len(strategy_results) * CONFIDENCE_THRESHOLD:
        direction = "SHORT"
        # Entry: Current price or slightly below
        entry = current_price * 0.9995
        
        # Take Profit: Near next support or fib extension
        tp_candidates = [
            fib_levels['1.618'],
            detect_support_resistance(data)['support'],
            current_price - (3 * atr)
        ]
        tp = max(tp for tp in tp_candidates if tp < current_price)
        
        # Stop Loss: Above recent swing high or FVG/OB
        sl_candidates = [
            fib_levels['0.618'],
            detect_support_resistance(data)['resistance'],
            current_price + (1.5 * atr),
            ob['bearish'][1] if ob['bearish'] else None,
            fvg['bearish'][1] if fvg['bearish'] else None
        ]
        sl = min(sl for sl in sl_candidates if sl and sl > current_price)
        
    else:
        direction = "NEUTRAL"
        entry = tp = sl = None
    
    return direction, entry, tp, sl

# =================== ALERT GENERATION ===================
def generate_telegram_alert(data, strategy_results, prediction, trade_params):
    """Generate comprehensive trading alert for Telegram"""
    current_price = data[-1]['close']
    timestamp = datetime.fromtimestamp(data[-1]['time']/1000).strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare strategy results table
    strategy_table = "\n".join(
        f"{strategy[:20].ljust(20)}: {result[0]} (Conf: {result[1]:.0%})" 
        for strategy, result in sorted(strategy_results.items())
    )
    
    # Prepare prediction summary
    if prediction:
        pred_direction = prediction[0]['trend']
        pred_confidence = prediction[0]['confidence']
        pred_summary = (
            f"📈 Predicted Direction: {'🚀 BULLISH' if pred_direction == 'up' else '📉 BEARISH' if pred_direction == 'down' else '🟡 NEUTRAL'}\n"
            f"🔮 Confidence: {pred_confidence:.0%}\n"
            f"🎯 Projected High: {max(p['price'] for p in prediction):.2f}\n"
            f"🛑 Projected Low: {min(p['price'] for p in prediction):.2f}"
        )
    else:
        pred_summary = "🔮 No clear prediction available"
    
    # Prepare trade parameters if available
    if trade_params[0] != "NEUTRAL":
        trade_info = (
            f"\n\n💎 TRADE SETUP 💎\n"
            f"Direction: {'🚀 LONG' if trade_params[0] == 'LONG' else '📉 SHORT'}\n"
            f"Entry: {trade_params[1]:.2f}\n"
            f"Take Profit: {trade_params[2]:.2f}\n"
            f"Stop Loss: {trade_params[3]:.2f}\n"
            f"Risk/Reward: {(abs(trade_params[2] - trade_params[1]) / abs(trade_params[3] - trade_params[1])):.2f}:1"
        )
    else:
        trade_info = "\n\n🛑 No trade recommended (low confidence)"
    
    # Prepare key levels
    levels = detect_support_resistance(data)
    fib_levels = calculate_fibonacci_levels(data)
    fvg = detect_fvg(data)
    ob = detect_order_blocks(data)
    
    levels_info = (
        f"\n\n🔑 KEY LEVELS\n"
        f"Support: {levels['support']:.2f}\n"
        f"Resistance: {levels['resistance']:.2f}\n"
        f"Fib 0.618: {fib_levels['0.618']:.2f}\n"
        f"Fib 1.618: {fib_levels['1.618']:.2f}\n"
        f"Bullish FVG: {fvg['bullish'][0] if fvg['bullish'] else 'N/A'}-{fvg['bullish'][1] if fvg['bullish'] else 'N/A'}\n"
        f"Bearish FVG: {fvg['bearish'][0] if fvg['bearish'] else 'N/A'}-{fvg['bearish'][1] if fvg['bearish'] else 'N/A'}\n"
        f"Bullish OB: {ob['bullish'][0] if ob['bullish'] else 'N/A'}-{ob['bullish'][1] if ob['bullish'] else 'N/A'}\n"
        f"Bearish OB: {ob['bearish'][0] if ob['bearish'] else 'N/A'}-{ob['bearish'][1] if ob['bearish'] else 'N/A'}"
    )
    
    # Compose full message
    message = (
        f"📊 *BTC/USDT 1min Trading Alert* 📊\n"
        f"🕒 {timestamp}\n"
        f"💰 Current Price: {current_price:.2f}\n\n"
        f"📊 *STRATEGY ANALYSIS*\n"
        f"{strategy_table}\n\n"
        f"{pred_summary}"
        f"{trade_info}"
        f"{levels_info}\n\n"
        f"⚠️ *Disclaimer*: Trading involves risk. Past performance is not indicative of future results."
    )
    
    return message

# =================== MAIN TRADING LOOP ===================
def trading_loop():
    """Main trading analysis loop"""
    while True:
        try:
            # Get market data
            data = get_historical_data()
            current_price = data[-1]['close']
            
            # Run all strategies
            all_strategies = {
                'EMA Crossover': ema_crossover_strategy(data),
                'VWAP': vwap_strategy(data),
                'Bollinger Bands': bollinger_band_strategy(data),
                'RSI': rsi_strategy(data),
                'MACD': macd_strategy(data),
                'Stoch RSI': stochastic_rsi_strategy(data),
                'Supertrend': supertrend_strategy(data),
                'Ichimoku': ichimoku_strategy(data),
                'CCI': cci_strategy(data),
                'ATR': atr_strategy(data),
                'S/R': support_resistance_strategy(data),
                'Trendline': trendline_strategy(data),
                'Inside Bar': inside_bar_strategy(data),
                'Double Top/Bottom': double_top_bottom_strategy(data),
                'Wick Rejection': wick_rejection_strategy(data),
                'Volume Spike': volume_spike_strategy(data),
                'Break & Retest': break_retest_strategy(data),
                'Liquidity Grab': liquidity_grab_strategy(data),
                'EMA+RSI': ema_rsi_strategy(data),
                'VWAP+MACD': vwap_macd_strategy(data),
                'BB+RSI': bollinger_rsi_strategy(data),
                'ICT': ict_strategy(data),
                'SMC': smc_strategy(data)
            }
            
            # Filter only strategies with signals (remove neutral)
            strategy_results = {k: v for k, v in all_strategies.items() if "neutral" not in v[0].lower()}
            
            # Predict next 15 candles
            prediction = predict_next_15_candles(data, strategy_results)
            
            # Calculate trade parameters
            trade_params = calculate_trade_parameters(data, strategy_results)
            
            # Generate and send alert
            alert_message = generate_telegram_alert(data, all_strategies, prediction, trade_params)
            bot.send_message(
                chat_id=TELEGRAM_CHANNEL,
                text=alert_message,
                parse_mode=ParseMode.MARKDOWN
            )
            
            logging.info(f"Sent trading alert at {datetime.now()}")
            
        except Exception as e:
            logging.error(f"Error in trading loop: {str(e)}")
            time.sleep(10)
            continue
        
        # Wait for next candle
        time.sleep(60 - (time.time() % 60))

# =================== FLASK API ENDPOINTS ===================
@app.route('/')
def status():
    return jsonify({
        "status": "running",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "last_update": datetime.now().isoformat()
    })

@app.route('/strategies')
def strategies():
    data = get_historical_data()
    return jsonify({
        strategy: result[0]
        for strategy, result in {
            'EMA Crossover': ema_crossover_strategy(data),
            'VWAP': vwap_strategy(data),
            # ... [include all other strategies]
        }.items()
    })

# =================== START APPLICATION ===================
if __name__ == "__main__":
    # Start trading thread
    trading_thread = threading.Thread(target=trading_loop)
    trading_thread.daemon = True
    trading_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5000)
