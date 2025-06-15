import logging
import re
from telegram import Update, ParseMode
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

BOT_TOKEN = '7663809573:AAGkEti5LjlBOuXA_NhdR9h64J63FDmFzMA'  # Replace with your bot token

SOURCE_CHANNEL = 'tradegrh'  # Source channel username without '@'
TARGET_CHANNEL = 'agrhpy'    # Target channel username without '@'

last_three_messages = []

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Regex patterns
MAJORITY_DECISION_RE = re.compile(r'Majority Decision:\s*(BUY|SELL)', re.IGNORECASE)
VOTES_RE = re.compile(r'🟢(\d+)\s*\|\s*🔴(\d+)')
ENTRY_PRICE_RE = re.compile(r'Entry:\s*([\d\.]+)')
TP_RE = re.compile(r'TP.*?:\s*([\d\.]+)')
SL_RE = re.compile(r'SL.*?:\s*([\d\.]+)')
FVG_RE = re.compile(r'FVG.*?(\d{1,6}\.\d+)\s*[→-]\s*(\d{1,6}\.\d+)')
OB_RE = re.compile(r'OB.*?@ (\d{1,6}\.\d+)')

def extract_direction(text):
    match = MAJORITY_DECISION_RE.search(text)
    if match:
        return match.group(1).capitalize()
    return None

def extract_votes(text):
    match = VOTES_RE.search(text)
    if match:
        green_votes = int(match.group(1))
        red_votes = int(match.group(2))
        return green_votes, red_votes
    return None, None

def extract_entry(text):
    match = ENTRY_PRICE_RE.search(text)
    if match:
        return float(match.group(1))
    return None

def extract_tp_sl(text):
    tp_match = TP_RE.search(text)
    sl_match = SL_RE.search(text)
    tp = float(tp_match.group(1)) if tp_match else None
    sl = float(sl_match.group(1)) if sl_match else None
    return tp, sl

def extract_fvg(text):
    matches = FVG_RE.findall(text)
    zones = []
    for low, high in matches:
        zones.append((float(low), float(high)))
    return zones

def extract_ob(text):
    matches = OB_RE.findall(text)
    return [float(m) for m in matches]

def best_zone(zones, direction):
    if not zones:
        return None
    if direction == 'Buy':
        best = min(zones, key=lambda x: x[0])
    else:
        best = max(zones, key=lambda x: x[1])
    if isinstance(best, tuple):
        if best[0] == best[1]:
            return f"{best[0]:.5f}"
        else:
            return f"{best[0]:.5f} - {best[1]:.5f}"
    else:
        return f"{best:.5f}"

def analyze_messages(messages):
    directions = []
    green_votes_total = 0
    red_votes_total = 0
    entries = []
    tps = []
    sls = []
    fvg_all = []
    ob_all = []

    for msg in messages:
        text = msg.text or ""

        dir_ = extract_direction(text)
        if dir_:
            directions.append(dir_)

        green, red = extract_votes(text)
        if green is not None and red is not None:
            green_votes_total += green
            red_votes_total += red

        entry = extract_entry(text)
        if entry:
            entries.append(entry)

        tp, sl = extract_tp_sl(text)
        if tp:
            tps.append(tp)
        if sl:
            sls.append(sl)

        fvg_all.extend(extract_fvg(text))
        ob_all.extend(extract_ob(text))

    if not directions:
        logger.info("No Majority Decision found in last 3 messages.")
        return None

    buy_count = directions.count('Buy')
    sell_count = directions.count('Sell')

    # Resolve dominant direction by message majority or votes if tie
    if buy_count == sell_count:
        if green_votes_total > red_votes_total:
            dominant_direction = 'Buy'
        elif red_votes_total > green_votes_total:
            dominant_direction = 'Sell'
        else:
            logger.info("No clear consensus from votes.")
            return None
    else:
        dominant_direction = 'Buy' if buy_count > sell_count else 'Sell'

    total_votes = green_votes_total + red_votes_total
    if total_votes > 0:
        if dominant_direction == 'Buy':
            confidence = int((green_votes_total / total_votes) * 100)
        else:
            confidence = int((red_votes_total / total_votes) * 100)
    else:
        confidence = int((max(buy_count, sell_count) / 3) * 100)

    # Best Entry
    best_entry = "N/A"
    if entries:
        best_entry = f"{min(entries):.5f}" if dominant_direction == 'Buy' else f"{max(entries):.5f}"

    # Best TP and SL
    best_tp = "N/A"
    best_sl = "N/A"
    if tps:
        best_tp = f"{max(tps):.5f}" if dominant_direction == 'Buy' else f"{min(tps):.5f}"
    if sls:
        best_sl = f"{min(sls):.5f}" if dominant_direction == 'Buy' else f"{max(sls):.5f}"

    # Best FVG and OB zones
    best_fvg_zone = best_zone(fvg_all, dominant_direction) or "N/A"
    best_ob_zone = "N/A"
    if ob_all:
        best_ob_zone = f"{min(ob_all):.5f}" if dominant_direction == 'Buy' else f"{max(ob_all):.5f}"

    # Example simple guidance based on direction & zones
    guidance = ""
    if best_fvg_zone != "N/A":
        if dominant_direction == 'Buy':
            guidance += f"Best FVG to buy from: {best_fvg_zone}. Consider buying if price bounces off this zone.\n"
        else:
            guidance += f"Best FVG to sell from: {best_fvg_zone}. Consider selling if price rejects this zone.\n"

    if best_ob_zone != "N/A":
        if dominant_direction == 'Buy':
            guidance += f"Best OB to buy from: {best_ob_zone}. Look for bullish confirmations near this zone.\n"
        else:
            guidance += f"Best OB to sell from: {best_ob_zone}. Look for bearish confirmations near this zone.\n"

    report = (
        f"*🔥 Professional Trading Alert 🔥*\n\n"
        f"*Direction:* `{dominant_direction}` ({confidence}%)\n"
        f"*Best Entry Price:* `{best_entry}`\n"
        f"*Best TP:* `{best_tp}`\n"
        f"*Best SL:* `{best_sl}`\n"
        f"*Best FVG Zones:* `{best_fvg_zone}`\n"
        f"*Best OB Zones:* `{best_ob_zone}`\n\n"
        f"*Guidance:*\n{guidance}\n"
        f"_Signals are based on consensus from last 3 notifications._"
    )
    return report

def message_handler(update: Update, context: CallbackContext):
    global last_three_messages

    message = update.effective_message
    chat = update.effective_chat

    if chat and chat.username and chat.username.lower() == SOURCE_CHANNEL.lower():
        last_three_messages.append(message)
        logger.info(f"Collected {len(last_three_messages)} messages from @{SOURCE_CHANNEL}")

        if len(last_three_messages) == 3:
            report = analyze_messages(last_three_messages)
            if report:
                try:
                    context.bot.send_message(
                        chat_id=f"@{TARGET_CHANNEL}",
                        text=report,
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=True,
                    )
                    logger.info(f"Sent filtered trading alert to @{TARGET_CHANNEL}")
                except Exception as e:
                    logger.error(f"Failed to send message to @{TARGET_CHANNEL}: {e}")
            else:
                logger.info("No clear consensus detected; no alert sent.")
            last_three_messages.clear()

def main():
    updater = Updater(BOT_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.chat(username=f"@{SOURCE_CHANNEL}"), message_handler))

    logger.info("Bot started. Listening to source channel...")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
