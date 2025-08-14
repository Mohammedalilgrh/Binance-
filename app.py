
from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
import json
import time
import threading
import requests
from datetime import datetime, timedelta
import random
import schedule

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('smm_agent.db')
    c = conn.cursor()
    
    # Bot accounts table
    c.execute('''CREATE TABLE IF NOT EXISTS bot_accounts
                 (id INTEGER PRIMARY KEY, platform TEXT, username TEXT, 
                  password TEXT, status TEXT, created_at TIMESTAMP)''')
    
    # Tasks table
    c.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id INTEGER PRIMARY KEY, task_type TEXT, target_url TEXT,
                  amount INTEGER, platform TEXT, status TEXT, created_at TIMESTAMP)''')
    
    # Orders table
    c.execute('''CREATE TABLE IF NOT EXISTS orders
                 (id INTEGER PRIMARY KEY, service TEXT, target TEXT, quantity INTEGER,
                  status TEXT, progress INTEGER, created_at TIMESTAMP)''')
    
    conn.commit()
    conn.close()

# Bot account management
class BotManager:
    def __init__(self):
        self.active_bots = {}
    
    def add_bot_account(self, platform, username, password):
        conn = sqlite3.connect('smm_agent.db')
        c = conn.cursor()
        c.execute("INSERT INTO bot_accounts (platform, username, password, status, created_at) VALUES (?, ?, ?, ?, ?)",
                 (platform, username, password, 'active', datetime.now()))
        conn.commit()
        conn.close()
        return True
    
    def get_bot_accounts(self, platform=None):
        conn = sqlite3.connect('smm_agent.db')
        c = conn.cursor()
        if platform:
            c.execute("SELECT * FROM bot_accounts WHERE platform = ? AND status = 'active'", (platform,))
        else:
            c.execute("SELECT * FROM bot_accounts WHERE status = 'active'")
        accounts = c.fetchall()
        conn.close()
        return accounts
    
    def simulate_action(self, platform, action_type, target, amount):
        # Simulate social media actions with random delays
        bots = self.get_bot_accounts(platform)
        if not bots:
            return False
        
        success_count = 0
        for i in range(min(amount, len(bots) * 10)):  # Each bot can do multiple actions
            bot = random.choice(bots)
            
            # Simulate action with random delay
            time.sleep(random.uniform(0.5, 3.0))
            
            # Simulate success rate (90% success)
            if random.random() < 0.9:
                success_count += 1
                print(f"Bot {bot[2]} performed {action_type} on {target}")
        
        return success_count

bot_manager = BotManager()

# Service handlers
class SMMService:
    def __init__(self):
        self.services = {
            'instagram_followers': {'price': 0, 'min': 10, 'max': 10000},
            'instagram_likes': {'price': 0, 'min': 10, 'max': 5000},
            'tiktok_followers': {'price': 0, 'min': 10, 'max': 10000},
            'tiktok_likes': {'price': 0, 'min': 10, 'max': 5000},
            'youtube_views': {'price': 0, 'min': 100, 'max': 100000},
            'telegram_members': {'price': 0, 'min': 10, 'max': 5000}
        }
    
    def create_order(self, service, target, quantity):
        conn = sqlite3.connect('smm_agent.db')
        c = conn.cursor()
        order_id = int(time.time() * 1000)  # Unique order ID
        c.execute("INSERT INTO orders (id, service, target, quantity, status, progress, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (order_id, service, target, quantity, 'processing', 0, datetime.now()))
        conn.commit()
        conn.close()
        
        # Start processing order in background
        threading.Thread(target=self.process_order, args=(order_id, service, target, quantity)).start()
        
        return order_id
    
    def process_order(self, order_id, service, target, quantity):
        platform = service.split('_')[0]
        action = service.split('_')[1]
        
        # Update order status
        conn = sqlite3.connect('smm_agent.db')
        c = conn.cursor()
        
        # Simulate processing
        for i in range(quantity):
            if bot_manager.simulate_action(platform, action, target, 1):
                progress = int((i + 1) / quantity * 100)
                c.execute("UPDATE orders SET progress = ? WHERE id = ?", (progress, order_id))
                conn.commit()
                time.sleep(random.uniform(1, 5))  # Random delay between actions
        
        # Mark as completed
        c.execute("UPDATE orders SET status = 'completed' WHERE id = ?", (order_id,))
        conn.commit()
        conn.close()

smm_service = SMMService()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/services')
def get_services():
    return jsonify(smm_service.services)

@app.route('/api/order', methods=['POST'])
def create_order():
    data = request.json
    service = data.get('service')
    target = data.get('target')
    quantity = int(data.get('quantity'))
    
    if service not in smm_service.services:
        return jsonify({'error': 'Invalid service'}), 400
    
    if quantity < smm_service.services[service]['min'] or quantity > smm_service.services[service]['max']:
        return jsonify({'error': 'Invalid quantity'}), 400
    
    order_id = smm_service.create_order(service, target, quantity)
    return jsonify({'order_id': order_id, 'status': 'processing'})

@app.route('/api/order/<int:order_id>')
def get_order(order_id):
    conn = sqlite3.connect('smm_agent.db')
    c = conn.cursor()
    c.execute("SELECT * FROM orders WHERE id = ?", (order_id,))
    order = c.fetchone()
    conn.close()
    
    if not order:
        return jsonify({'error': 'Order not found'}), 404
    
    return jsonify({
        'order_id': order[0],
        'service': order[1],
        'target': order[2],
        'quantity': order[3],
        'status': order[4],
        'progress': order[5],
        'created_at': order[6]
    })

@app.route('/api/bots/add', methods=['POST'])
def add_bot():
    data = request.json
    platform = data.get('platform')
    username = data.get('username')
    password = data.get('password')
    
    if bot_manager.add_bot_account(platform, username, password):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to add bot'}), 400

@app.route('/api/bots')
def get_bots():
    platform = request.args.get('platform')
    bots = bot_manager.get_bot_accounts(platform)
    return jsonify([{'id': b[0], 'platform': b[1], 'username': b[2], 'status': b[4]} for b in bots])

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Auto-add demo bot accounts for testing
def setup_demo_bots():
    platforms = ['instagram', 'tiktok', 'youtube', 'telegram']
    for platform in platforms:
        for i in range(50):  # Add 50 bots per platform
            bot_manager.add_bot_account(
                platform,
                f"{platform}_bot_{i+1}",
                f"password_{i+1}"
            )

if __name__ == '__main__':
    init_db()
    setup_demo_bots()  # Add demo bots
    print("SMM Agent starting...")
    print("Available services: Instagram, TikTok, YouTube, Telegram")
    print("Bot accounts loaded: 200+ accounts ready")
    app.run(host='0.0.0.0', port=5000, debug=True)
