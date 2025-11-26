import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

DB_POSITIONS = "positions.db"
DB_MARKET = "market_data.db"

def init_dbs():
    """Initialize SQLite tables if they don't exist."""
    # 1. Positions DB
    conn = sqlite3.connect(DB_POSITIONS)
    c = conn.cursor()
    
    # Trade Blotter
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        open_ts TEXT,
        status TEXT,
        ticker_pay TEXT,
        ticker_rec TEXT,
        tenor_pay REAL,
        tenor_rec REAL,
        dv01 REAL,
        entry_rate_pay REAL,
        entry_rate_rec REAL,
        entry_z_spread REAL,
        model_z_score REAL,
        close_ts TEXT,
        close_reason TEXT,
        realized_pnl_cash REAL DEFAULT 0.0
    )''')
    
    # System State (Singleton)
    c.execute('''CREATE TABLE IF NOT EXISTS system_state (
        id INTEGER PRIMARY KEY,
        last_update_ts TEXT,
        shock_block_remaining INTEGER DEFAULT 0,
        regime_status TEXT DEFAULT 'SAFE',
        cumulative_realized_pnl REAL DEFAULT 0.0
    )''')
    
    # Init state row if not exists
    c.execute("INSERT OR IGNORE INTO system_state (id, regime_status) VALUES (1, 'SAFE')")
    conn.commit()
    conn.close()

    # 2. Market Data DB (Rolling Buffer)
    conn = sqlite3.connect(DB_MARKET)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ticks (
        timestamp TEXT,
        ticker TEXT,
        rate REAL
    )''')
    conn.commit()
    conn.close()

def log_ticks(tick_list):
    """Batch insert ticks for EOD processing."""
    if not tick_list: return
    conn = sqlite3.connect(DB_MARKET)
    c = conn.cursor()
    data = [(t['ts'], t['ticker'], t['rate']) for t in tick_list]
    c.executemany("INSERT INTO ticks (timestamp, ticker, rate) VALUES (?, ?, ?)", data)
    conn.commit()
    conn.close()

def add_position(trade_dict):
    conn = sqlite3.connect(DB_POSITIONS)
    c = conn.cursor()
    cols = ', '.join(trade_dict.keys())
    placeholders = ', '.join(['?'] * len(trade_dict))
    sql = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"
    c.execute(sql, list(trade_dict.values()))
    conn.commit()
    conn.close()

def update_position_status(trade_id, status, close_reason=None, close_ts=None, pnl=0.0):
    conn = sqlite3.connect(DB_POSITIONS)
    c = conn.cursor()
    c.execute("""
        UPDATE trades 
        SET status=?, close_reason=?, close_ts=?, realized_pnl_cash=?
        WHERE trade_id=?
    """, (status, close_reason, close_ts, pnl, trade_id))
    conn.commit()
    conn.close()

def delete_position(trade_id):
    """Hard delete for mistakes."""
    conn = sqlite3.connect(DB_POSITIONS)
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE trade_id=?", (trade_id,))
    conn.commit()
    conn.close()

def get_open_positions():
    conn = sqlite3.connect(DB_POSITIONS)
    df = pd.read_sql("SELECT * FROM trades WHERE status='OPEN'", conn)
    conn.close()
    return df

def get_all_positions():
    conn = sqlite3.connect(DB_POSITIONS)
    df = pd.read_sql("SELECT * FROM trades ORDER BY open_ts DESC", conn)
    conn.close()
    return df

def get_system_state():
    conn = sqlite3.connect(DB_POSITIONS)
    row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    conn.close()
    return row
