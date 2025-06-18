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
INTERVAL_1M = Client.KLINE_INTERVAL_1MINUTE
INTERVAL_5M = Client.KLINE_INTERVAL_5MINUTE
CANDLE_LIMIT_1M = 68
CANDLE_LIMIT_5M = 68

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# ================ UTILITIES =================

def get_candles(symbol=SYMBOL, interval=INTERVAL_1M, limit=68):
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
    if not values or period <= 0:
        return 0
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
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    def cluster_levels(levels):
        clusters = []
        for level in sorted(levels):
            found = False
            for cluster in clusters:
                if abs(level - cluster[0])/cluster[0] < 0.005:
                    cluster.append(level)
                    found = True
                    break
            if not found:
                clusters.append([level])
        return clusters

    high_clusters = cluster_levels(highs)
    low_clusters = cluster_levels(lows)

    def score_cluster(cluster):
        recent_weight = sum(1/(len(candles)-i) for i, price in enumerate(highs) if price in cluster)
        return len(cluster) * recent_weight

    high_clusters.sort(key=score_cluster, reverse=True)
    low_clusters.sort(key=score_cluster, reverse=True)

    significant_levels = []
    if len(high_clusters) > 0:
        significant_levels.append(round(sum(high_clusters[0])/len(high_clusters[0]), 2))
    if len(high_clusters) > 1:
        significant_levels.append(round(sum(high_clusters[1])/len(high_clusters[1]), 2))
    if len(low_clusters) > 0:
        significant_levels.append(round(sum(low_clusters[0])/len(low_clusters[0]), 2))
    if len(low_clusters) > 1:
        significant_levels.append(round(sum(low_clusters[1])/len(low_clusters[1]), 2))

    significant_levels = sorted(significant_levels, reverse=True)
    return significant_levels[:4]

def analyze_chart_pattern(candles):
    levels = detect_horizontal_levels(candles)
    current_price = candles[-1]["close"]

    pattern = ""
    if len(levels) >= 4:
        if levels[0] > levels[1] > levels[2] > levels[3]:
            pattern = "Descending Channel (Lower Highs)"
        elif levels[0] < levels[1] < levels[2] < levels[3]:
            pattern = "Ascending Channel (Higher Lows)"
        else:
            max_diff = max(levels) - min(levels)
            if max_diff < (max(levels) * 0.01):
                pattern = "Consolidation Rectangle"
            else:
                top_levels = [l for l in levels if l > current_price]
                bottom_levels = [l for l in levels if l < current_price]
                if len(top_levels) >= 2 and len(bottom_levels) >= 2:
                    if sorted(top_levels, reverse=True) == top_levels and sorted(bottom_levels) == bottom_levels:
                        pattern = "Symmetrical Triangle"
                    elif sorted(top_levels, reverse=True) == top_levels and len(set(bottom_levels)) == 1:
                        pattern = "Descending Triangle"
                    elif sorted(bottom_levels) == bottom_levels and len(set(top_levels)) == 1:
                        pattern = "Ascending Triangle"

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
    if candles[-3]["low"] > candles[-1]["low"] and candles[-1]["close"] > candles[-3]["high"]:
        return "🟢up"
    elif candles[-3]["high"] < candles[-1]["high"] and candles[-1]["close"] < candles[-3]["low"]:
        return "🔴down"
    return "⚪️neutral"

def smc(candles):
    if (candles[-2]["low"] < candles[-3]["low"] and
        candles[-1]["close"] > candles[-3]["high"]):
        return "🟢up"
    elif (candles[-2]["high"] > candles[-3]["high"] and
          candles[-1]["close"] < candles[-3]["low"]):
        return "🔴down"
    return "⚪️neutral"

# ============== (STRATEGY FUNCTIONS) ==============
# For brevity, assume all your original strategy functions here unchanged...
# (strategy_ema_cross, strategy_macd, strategy_adx, ..., strategy_higher_timeframe_confirmation, etc.)
# Paste all your existing strategy functions here exactly as you have them.

# (For this example, let's keep them as is, unchanged)

# ========== STRATEGY RUNNER + DECISION ==========

def run_all_strategies(candles):
    # Paste your full strategy dictionary here (same as original, unchanged)
    # Keep all strategy calls as-is
    strategies = {
        "EMA Crossover": strategy_ema_cross(candles),
        "MACD": strategy_macd(candles),
        "ADX": strategy_adx(candles),
        "Parabolic SAR": strategy_parabolic_sar(candles),
        "Keltner Channels": strategy_keltner_channels(candles),
        "TRIX": strategy_trix(candles),
        "Awesome Oscillator": strategy_awesome_oscillator(candles),

        "RSI": strategy_rsi(candles),
        "Bollinger Bands": strategy_bollinger(candles),
        "Stochastic": strategy_stochastic(candles),
        "CCI": strategy_cci(candles),
        "MFI": strategy_mfi(candles),
        "VWAP": strategy_vwap(candles),

        "Donchian Breakout": strategy_donchian_breakout(candles),
        "Inside Bar": strategy_inside_bar(candles),
        "Break+Retest": strategy_break_retest(candles),
        "High/Low Breakout": strategy_high_low_breakout(candles),
        "Fractal Breakout": strategy_fractal_breakout(candles),

        "Momentum": strategy_momentum(candles),
        "ROC": strategy_roc(candles),
        "Williams %R": strategy_williams_r(candles),
        "Ultimate Oscillator": strategy_ultimate_oscillator(candles),
        "OBV": strategy_obv(candles),

        "Volume Spike": strategy_volume_spike(candles),
        "VWAP+MACD Combo": strategy_vwap_macd_combo(candles),
        "Volume Profile": strategy_volume_profile(candles),
        "Accumulation/Distribution": strategy_accumulation_distribution(candles),

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

        "Heikin Ashi": strategy_heikin_ashi(candles),
        "VWMA": strategy_volume_weighted_ma(candles),
        "Supertrend": strategy_supertrend(candles),
        "Ichimoku": strategy_ichimoku(candles),

        "FVG": strategy_fvg(candles),
        "Order Block": strategy_ob(candles),
        "Liquidity Grab": strategy_liquidity_grab(candles),
        "Mitigation Block": strategy_mitigation_block(candles),

        "HTF Confirmation": strategy_higher_timeframe_confirmation(candles),

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

def combine_multi_tf_votes(votes_1m, votes_5m):
    combined_votes = {"🟢up": 0, "🔴down": 0, "⚪️neutral": 0}
    for key in combined_votes.keys():
        combined_votes[key] = votes_1m.get(key,0) + votes_5m.get(key,0)
    if combined_votes["🟢up"] > combined_votes["🔴down"]:
        return "BUY ✅", combined_votes
    elif combined_votes["🔴down"] > combined_votes["🟢up"]:
        return "SELL ❌", combined_votes
    else:
        return "WAIT 🕒", combined_votes

# ========== ALERT GENERATOR ==========

def generate_telegram_message(candles_1m, signals_1m, votes_1m, decision_1m,
                              candles_5m, signals_5m, votes_5m, decision_5m,
                              combined_decision, combined_votes,
                              prediction_1m, prediction_5m):
    entry_1m = candles_1m[-1]["close"]
    atr_val_1m = atr(candles_1m, 14)
    fib_tp_1m = fib_extension(entry_1m, atr_val_1m, "BUY" if decision_1m == "BUY ✅" else "SELL")
    stop_loss_1m = round(entry_1m - atr_val_1m, 2) if decision_1m == "BUY ✅" else round(entry_1m + atr_val_1m, 2)

    entry_5m = candles_5m[-1]["close"]
    atr_val_5m = atr(candles_5m, 14)
    fib_tp_5m = fib_extension(entry_5m, atr_val_5m, "BUY" if decision_5m == "BUY ✅" else "SELL")
    stop_loss_5m = round(entry_5m - atr_val_5m, 2) if decision_5m == "BUY ✅" else round(entry_5m + atr_val_5m, 2)

    ob_1m = detect_order_block(candles_1m)
    fvg_1m = detect_fvg(candles_1m)
    chart_pattern_1m = analyze_chart_pattern(candles_1m)

    ob_5m = detect_order_block(candles_5m)
    fvg_5m = detect_fvg(candles_5m)
    chart_pattern_5m = analyze_chart_pattern(candles_5m)

    def format_signals(signals, filter_keys):
        return "\n".join(f"• {name}: {signal}" for name, signal in signals.items() if signal != "⚪️neutral" and name in filter_keys)

    trend_keys = [
        "EMA Crossover", "MACD", "ADX", "Parabolic SAR", "Keltner Channels",
        "TRIX", "Awesome Oscillator", "Supertrend", "Ichimoku"
    ]

    reversal_keys = ["RSI", "Bollinger Bands", "Stochastic", "CCI", "MFI", "VWAP"]

    candle_pattern_keys = [
        "Wick Rejection", "Engulfing", "Hammer/Hanging Man", "Harami",
        "Doji", "Morning Star", "Evening Star", "Pin Bar", "Fakey",
        "3 White Soldiers", "3 Black Crows"
    ]

    smc_keys = [
        "FVG", "Order Block", "Liquidity Grab", "Mitigation Block", "ICT", "SMC"
    ]

    msg = f"📊 *BTCUSDT Multi-Timeframe Strategy Summary*\n\n"

    msg += f"🕐 *1-Minute Timeframe*\n"
    msg += f"🧠 Majority Decision: {decision_1m} (🟢{votes_1m['🟢up']} | 🔴{votes_1m['🔴down']} | ⚪️{votes_1m['⚪️neutral']})\n"
    msg += f"📈 15-Candle Forecast: {prediction_1m}\n"
    msg += f"🎯 Entry: `{entry_1m:.2f}` | 🛑 SL: `{stop_loss_1m}` | 🎯 TP: `{fib_tp_1m}`\n"
    msg += f"📉 Chart Pattern: {chart_pattern_1m['pattern']}\n\n"

    msg += f"🕔 *5-Minute Timeframe*\n"
    msg += f"🧠 Majority Decision: {decision_5m} (🟢{votes_5m['🟢up']} | 🔴{votes_5m['🔴down']} | ⚪️{votes_5m['⚪️neutral']})\n"
    msg += f"📈 15-Candle Forecast: {prediction_5m}\n"
    msg += f"🎯 Entry: `{entry_5m:.2f}` | 🛑 SL: `{stop_loss_5m}` | 🎯 TP: `{fib_tp_5m}`\n"
    msg += f"📉 Chart Pattern: {chart_pattern_5m['pattern']}\n\n"

    msg += f"🔗 *Combined Multi-Timeframe Decision*: {combined_decision} (🟢{combined_votes['🟢up']} | 🔴{combined_votes['🔴down']} | ⚪️{combined_votes['⚪️neutral']})\n\n"

    # Simple buy/sell guide:
    msg += "📘 *Multi-Timeframe Trading Guide:*\n"
    if decision_1m == "BUY ✅" and decision_5m == "BUY ✅":
        msg += "✅ Strong BUY signal across both timeframes. Consider entering a long position.\n"
    elif decision_1m == "SELL ❌" and decision_5m == "SELL ❌":
        msg += "❌ Strong SELL signal across both timeframes. Consider entering a short position.\n"
    elif decision_1m == "BUY ✅" and decision_5m != "BUY ✅":
        msg += "⚠️ 1m suggests BUY but 5m is neutral or SELL. Consider caution or wait for confirmation.\n"
    elif decision_1m == "SELL ❌" and decision_5m != "SELL ❌":
        msg += "⚠️ 1m suggests SELL but 5m is neutral or BUY. Consider caution or wait for confirmation.\n"
    else:
        msg += "⏳ Mixed signals. Best to wait or observe further.\n"

    # Trend signals 1m
    msg += "\n🟢 *1m Trend Signals*\n" + format_signals(signals_1m, trend_keys)
    # Reversal signals 1m
    msg += "\n\n🔄 *1m Reversal Signals*\n" + format_signals(signals_1m, reversal_keys)
    # Candle patterns 1m
    msg += "\n\n🕯️ *1m Candle Patterns*\n" + format_signals(signals_1m, candle_pattern_keys)
    # Smart Money 1m
    msg += "\n\n💡 *1m Smart Money Signals*\n" + format_signals(signals_1m, smc_keys)

    # Trend signals 5m
    msg += "\n\n🟢 *5m Trend Signals*\n" + format_signals(signals_5m, trend_keys)
    # Reversal signals 5m
    msg += "\n\n🔄 *5m Reversal Signals*\n" + format_signals(signals_5m, reversal_keys)
    # Candle patterns 5m
    msg += "\n\n🕯️ *5m Candle Patterns*\n" + format_signals(signals_5m, candle_pattern_keys)
    # Smart Money 5m
    msg += "\n\n💡 *5m Smart Money Signals*\n" + format_signals(signals_5m, smc_keys)

    if ob_1m:
        msg += f"\n\n🧱 1m OB: {ob_1m['type']} @ {ob_1m['price']:.2f}"
    if fvg_1m:
        zone = fvg_1m["zone"]
        msg += f"\n📐 1m FVG: {fvg_1m['type']} Zone {zone[0]:.2f} → {zone[1]:.2f}"

    if ob_5m:
        msg += f"\n\n🧱 5m OB: {ob_5m['type']} @ {ob_5m['price']:.2f}"
    if fvg_5m:
        zone = fvg_5m["zone"]
        msg += f"\n📐 5m FVG: {fvg_5m['type']} Zone {zone[0]:.2f} → {zone[1]:.2f}"

    msg += "\n\n📣 Powered by Smart Multi-Strategy AI 🔍"
    return msg

# ========== MASTER ANALYSIS FUNCTION ==========

def analyze():
    candles_1m = get_candles(interval=INTERVAL_1M, limit=CANDLE_LIMIT_1M)
    candles_5m = get_candles(interval=INTERVAL_5M, limit=CANDLE_LIMIT_5M)

    if len(candles_1m) < 30 or len(candles_5m) < 30:
        logging.warning("Not enough candles for 1m or 5m.")
        return

    signals_1m = run_all_strategies(candles_1m)
    decision_1m, votes_1m = aggregate_signals(signals_1m)
    prediction_1m = predict_next_15(candles_1m)

    signals_5m = run_all_strategies(candles_5m)
    decision_5m, votes_5m = aggregate_signals(signals_5m)
    prediction_5m = predict_next_15(candles_5m)

    combined_decision, combined_votes = combine_multi_tf_votes(votes_1m, votes_5m)

    message = generate_telegram_message(
        candles_1m, signals_1m, votes_1m, decision_1m,
        candles_5m, signals_5m, votes_5m, decision_5m,
        combined_decision, combined_votes,
        prediction_1m, prediction_5m
    )

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
    return "🔁 BTCUSDT Multi-Timeframe Strategy Bot is Running."

if __name__ == "__main__":
    threading.Thread(target=run_scheduler).start()
    app.run(host="0.0.0.0", port=7000)
