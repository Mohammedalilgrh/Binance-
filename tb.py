import os
import io
from flask import Flask, request
from binance.client import Client
import pandas as pd
import pytz
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import numpy as np
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

# ------------------- Settings -------------------
BOT_TOKEN = '7970489926:AAGjDmazd_EXkdT1cv8Lh8aNGZ1hPlkbcJg'
CHANNEL_ID = '@tradegrh'
API_KEY = 'cVRnAxc6nrVHQ6sbaAQNcrznHhOO7PcVZYlsES8Y75r34VJbYjQDfUTNcC8T2Fct'
API_SECRET = 'GEYh2ck82RcaDTaHjbLafYWBLqkAMw90plNSkfmhrvVbAFcowBxcst4L3u0hBLfC'
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
TIMEFRAME = '30m'
LOOKBACK = 40
RISK_PCT = 0.02
RR = 2.5

# ------------------- Flask App -------------------
app = Flask(__name__)
bot = Bot(token=BOT_TOKEN)

# Binance client
client = Client(API_KEY, API_SECRET)

# Dispatcher for handlers
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# ------------------- Market Structure Utils -------------------
def fetch_klines(symbol, interval, limit):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'time','open','high','low','close','volume','close_time','qav','trades',
        'taker_base_vol','taker_quote_vol','ignore'
    ])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Dubai')
    df.set_index('time', inplace=True)
    return df

def detect_bos(df, lookback=15):
    highs = df['high'][-lookback:]
    lows = df['low'][-lookback:]
    last_close = df['close'].iloc[-1]
    swing_high = highs[:-1].max()
    swing_low = lows[:-1].min()
    if last_close > swing_high:
        return {'type':'bull', 'level': swing_high}
    elif last_close < swing_low:
        return {'type':'bear', 'level': swing_low}
    return None

def detect_choch(df):
    if len(df) < 6:
        return None
    highs = df['high'][-6:]
    lows = df['low'][-6:]
    prev_high = highs[:-2].max()
    prev_low = lows[:-2].min()
    if (df['close'].iloc[-2] > prev_high and df['close'].iloc[-1] < df['low'].iloc[-2]):
        return {'type':'bear', 'level': df['low'].iloc[-2]}
    elif (df['close'].iloc[-2] < prev_low and df['close'].iloc[-1] > df['high'].iloc[-2]):
        return {'type':'bull', 'level': df['high'].iloc[-2]}
    return None

def detect_order_blocks(df, window=8):
    obs = []
    for i in range(window, len(df)):
        if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and
            df['close'].iloc[i] > df['high'].iloc[i-1] and
            df['volume'].iloc[i] > df['volume'].iloc[i-window:i].mean()):
            obs.append({'type':'bull', 'price':df['low'].iloc[i-1], 'index':i-1})
        if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and
            df['close'].iloc[i] < df['low'].iloc[i-1] and
            df['volume'].iloc[i] > df['volume'].iloc[i-window:i].mean()):
            obs.append({'type':'bear', 'price':df['high'].iloc[i-1], 'index':i-1})
    return obs[-2:] if obs else []

def detect_fvg(df):
    fvgs = []
    for i in range(2, len(df)):
        prev = df.iloc[i-2]
        curr = df.iloc[i]
        if curr['low'] > prev['high']:
            fvgs.append({'type':'bull', 'zone':(prev['high'], curr['low']), 'index':i})
        if curr['high'] < prev['low']:
            fvgs.append({'type':'bear', 'zone':(curr['high'], prev['low']), 'index':i})
    return fvgs[-2:] if fvgs else []

def calculate_fibonacci(high, low):
    diff = high - low
    return {
        '0.236': high - diff*0.236,
        '0.382': high - diff*0.382,
        '0.5': high - diff*0.5,
        '0.618': high - diff*0.618,
        '0.786': high - diff*0.786
    }

# ------------------- Chart Drawing -------------------
def draw_chart(symbol, df, fib, bos, choch, obs, fvgs):
    addplots = []
    # Fibonacci Levels
    for lvl in fib.values():
        addplots.append(mpf.make_addplot([lvl]*len(df),color='cyan',width=1,linestyle='dotted'))
    # BOS/CHoCH horizontal lines
    if bos:
        color = 'g' if bos['type']=='bull' else 'r'
        addplots.append(mpf.make_addplot([bos['level']]*len(df),color=color,linestyle='dashdot'))
    if choch:
        color = 'b'
        addplots.append(mpf.make_addplot([choch['level']]*len(df),color=color,linestyle='dotted'))
    # OB rectangles
    for ob in obs:
        idx = ob['index']
        price = ob['price']
        color = 'lime' if ob['type']=='bull' else 'red'
        addplots.append(mpf.make_addplot([np.nan]*idx+[price]+[np.nan]*(len(df)-idx-1),markersize=80,marker='v' if ob['type']=='bear' else '^',color=color))
    # FVGs as shaded regions
    fvg_col = {'bull':'#00bcff33', 'bear':'#ff003366'}
    fvg_boxes = []
    for fvg in fvgs:
        idx = fvg['index']
        zl, zh = fvg['zone']
        box = dict(ymin=zl, ymax=zh, xmin=idx-0.5, xmax=idx+0.5, facecolor=fvg_col[fvg['type']], alpha=0.3, linewidth=0)
        fvg_boxes.append(box)
    # Plotting
    fig, ax = mpf.plot(df, type='candle', mav=(5,13), addplot=addplots,
            style='yahoo', title=f"{symbol} - 30min",
            volume=True, returnfig=True, figsize=(8,5))
    # Draw FVG boxes
    for box in fvg_boxes:
        ax[0].add_patch(matplotlib.patches.Rectangle(
            (box['xmin'], box['ymin']),
            box['xmax']-box['xmin'],
            box['ymax']-box['ymin'],
            color=box['facecolor'], alpha=box['alpha'], linewidth=0
        ))
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ------------------- Signal Generation -------------------
def generate_signal(symbol, balance):
    df = fetch_klines(symbol, TIMEFRAME, LOOKBACK)
    bos = detect_bos(df)
    choch = detect_choch(df)
    obs = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    recent_high = df['high'][-15:].max()
    recent_low = df['low'][-15:].min()
    fib = calculate_fibonacci(recent_high, recent_low)
    # Direction logic
    direction = None
    score = 0
    explanations = []
    if bos:
        direction = 'BUY' if bos['type']=='bull' else 'SELL'
        score += 30
        explanations.append(f"{'🟢' if bos['type']=='bull' else '🔴'} BOS at {bos['level']:.2f}")
    if choch:
        if choch['type']=='bull':
            direction = direction or 'BUY'
            score += 25
        else:
            direction = direction or 'SELL'
            score += 25
        explanations.append(f"🔵 CHoCH {choch['type']} at {choch['level']:.2f}")
    if obs:
        ob = obs[-1]
        if ob['type']=='bull' and df['close'].iloc[-1] > ob['price']:
            direction = direction or 'BUY'
            score += 20
            explanations.append(f"🟩 Bull OB at {ob['price']:.2f}")
        if ob['type']=='bear' and df['close'].iloc[-1] < ob['price']:
            direction = direction or 'SELL'
            score += 20
            explanations.append(f"🟥 Bear OB at {ob['price']:.2f}")
    if fvgs:
        fvg = fvgs[-1]
        zone = fvg['zone']
        if fvg['type']=='bull' and zone[0]<=df['close'].iloc[-1]<=zone[1]:
            direction = direction or 'BUY'
            score += 10
            explanations.append(f"🟦 Bull FVG {zone[0]:.2f}-{zone[1]:.2f}")
        if fvg['type']=='bear' and zone[1]<=df['close'].iloc[-1]<=zone[0]:
            direction = direction or 'SELL'
            score += 10
            explanations.append(f"🟧 Bear FVG {zone[1]:.2f}-{zone[0]:.2f}")
    # Default direction
    if direction is None:
        direction = 'BUY' if df['close'].iloc[-1] < fib['0.618'] else 'SELL'
        explanations.append(f"⚖️ Defaulted on Fib 0.618")
    entry = df['close'].iloc[-1]
    lows = df['low'][-8:]
    highs = df['high'][-8:]
    if direction=='BUY':
        sl = lows.min()
        tp = entry + (entry-sl)*RR
    else:
        sl = highs.max()
        tp = entry - (sl-entry)*RR
    risk_amt = balance * RISK_PCT
    pos_size = risk_amt / max(abs(entry-sl),1e-8)
    # Generate chart buffer
    chart_buf = draw_chart(symbol, df, fib, bos, choch, obs, fvgs)
    return {
        'symbol': symbol,
        'direction': direction,
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'score': score,
        'position_size': pos_size,
        'risk_amount': risk_amt,
        'explanations': explanations,
        'chart_buf': chart_buf
    }

# ------------------- Telegram Bot Handlers -------------------
user_balances = {}

def start(update, context):
    chat_id = update.effective_chat.id
    user_balances[chat_id] = 1000.0
    buttons = [
        [InlineKeyboardButton(f"{s}", callback_data=f"analyze_{s}")] for s in SYMBOLS
    ]
    context.bot.send_message(
        chat_id,
        f"🤖 *Institutional Sniper Bot*\n\n"
        f"Your balance: ${user_balances[chat_id]:.2f}\n"
        f"Choose a symbol:",
        reply_markup=InlineKeyboardMarkup(buttons),
        parse_mode="Markdown"
    )

def set_balance(update, context):
    chat_id = update.effective_chat.id
    context.bot.send_message(chat_id, "Send your new balance (number):")
    return

def handle_message(update, context):
    chat_id = update.effective_chat.id
    try:
        val = float(update.message.text)
        user_balances[chat_id] = val
        context.bot.send_message(chat_id, f"Balance set to ${val:.2f}")
        start(update, context)
    except:
        context.bot.send_message(chat_id, "❌ Please send a valid number.")

def analyze_symbol(update, context):
    query = update.callback_query
    chat_id = query.message.chat.id
    symbol = query.data.split('_', 1)[1]
    balance = user_balances.get(chat_id, 1000.0)
    context.bot.send_message(chat_id, f"🔄 Analyzing {symbol} market...")
    try:
        signal = generate_signal(symbol, balance)
        buf = signal['chart_buf']
        caption = (
            f"*{symbol}*\n"
            f"Direction: *{signal['direction']}*\n"
            f"Entry: `{signal['entry']:.3f}`\n"
            f"SL: `{signal['sl']:.3f}`\n"
            f"TP: `{signal['tp']:.3f}`\n"
            f"PosSize: `{signal['position_size']:.4f}`\n"
            f"Risk: `${signal['risk_amount']:.2f}`\n"
            f"Score: `{signal['score']}`\n"
            f"\n" + "\n".join(signal['explanations'])
        )
        context.bot.send_photo(chat_id, buf, caption=caption, parse_mode="Markdown")
        # Also send to channel
        buf.seek(0)
        context.bot.send_photo(CHANNEL_ID, buf, caption=f"🚨 New {symbol} Signal\n{caption}", parse_mode="Markdown")
    except Exception as e:
        context.bot.send_message(chat_id, f"⚠️ Error: {e}")

# Add handlers
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('balance', set_balance))
dispatcher.add_handler(CallbackQueryHandler(analyze_symbol, pattern=r"^analyze_"))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

# ------------------- Webhook Endpoint -------------------
@app.route(f"/{BOT_TOKEN}", methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return 'ok'

@app.route('/')
def index():
    return 'Bot running!'

# ------------------- Main Entrypoint -------------------
if __name__ == "__main__":
    # Set webhook URL (do this once, or via Render Dashboard)
    # bot.set_webhook("https://binance-bql7.onrender.com" + BOT_TOKEN)
    # Run Flask app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
