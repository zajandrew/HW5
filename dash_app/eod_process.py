import pandas as pd
import numpy as np
import sqlite3
import sys
import datetime
from pathlib import Path

# --- Hook into Research Libraries ---
sys.path.append(str(Path(__file__).parent.parent))
import cr_config as cr
import feature_creation as fc
import hybrid_filter as hf
import portfolio_test as pt # For rule constants like MAX_HOLD, Z_STOP

# --- Constants ---
DB_MARKET = "market_data.db"
DB_POSITIONS = "positions.db"

def get_db_connection(db_name):
    return sqlite3.connect(db_name)

def step_1_export_ticks_to_parquet(yymm):
    """
    Query market_data.db for the SPECIFIC MONTH (YYMM), 
    pivot to Wide Format, save to cr.PATH_DATA.
    """
    print(f"[EOD] Step 1: Exporting ticks for {yymm}...")
    conn = get_db_connection(DB_MARKET)
    
    # 1. Calculate Start/End dates for SQL filtering
    # YYMM format (e.g., '2311') needs to become '2023-11' for SQL string matching
    # assuming we are in 21st century
    year = "20" + yymm[:2]
    month = yymm[2:]
    date_filter = f"{year}-{month}" # e.g. "2023-11"
    
    # 2. Query with Filter (SQLite strftime)
    # This ensures we don't pull old months if the DB grows
    sql = f"""
        SELECT timestamp, ticker, rate 
        FROM ticks 
        WHERE strftime('%Y-%m', timestamp) = '{date_filter}'
    """
    
    df = pd.read_sql(sql, conn)
    conn.close()
    
    if df.empty:
        print(f"[WARN] No ticks found for {date_filter}. Skipping export.")
        return False

    # ... (Rest of the function: pivoting and saving remains exactly the same) ...
    df['ts'] = pd.to_datetime(df['timestamp'])
    df_wide = df.pivot_table(index='ts', columns='ticker', values='rate', aggfunc='last')
    df_wide = df_wide.sort_index()
    
    out_path = Path(cr.PATH_DATA) / f"{yymm}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_parquet(out_path)
    print(f"[EOD] Saved raw features to {out_path}")
    return True

def step_2_build_model(yymm):
    """
    Run the heavy math: Splines and PCA.
    Generates data/enh/{YYMM}_enh.parquet
    """
    print(f"[EOD] Step 2: Running Feature Creation for {yymm}...")
    try:
        # This calls your research library directly
        fc.build_month(yymm)
        print("[EOD] Model Build Complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Feature Creation failed: {e}")
        return False

def step_3_update_signals_and_state():
    """
    Run Hybrid Filter to check if tomorrow is SAFE or SHOCK.
    Updates 'system_state' table in positions.db.
    """
    print(f"[EOD] Step 3: Updating System State (Regime/Shock)...")
    
    # 1. Generate Signals (Uses the _enh.parquet we just built)
    # Force rebuild to ensure it sees the new EOD data
    signals = hf.get_or_build_hybrid_signals(force_rebuild=True)
    
    # 2. Get latest status
    last_signal = signals.iloc[-1]
    
    # Check Regime (Logic from hybrid_filter)
    # Simple check: Is signal_health_z good?
    is_regime_safe = True
    if last_signal['signal_health_z'] < cr.MIN_SIGNAL_HEALTH_Z:
        is_regime_safe = False
        
    # Check Shock (This usually requires pnl history, simplistic check here)
    # We read the 'system_state' to decrement any existing block
    conn = get_db_connection(DB_POSITIONS)
    state_row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    # Schema: id, last_update, shock_remaining, regime_status, cum_pnl
    
    shock_remaining = state_row[2] if state_row else 0
    
    # Decrement shock block if it exists (one day passed)
    new_shock_remaining = max(0, shock_remaining - 1)
    
    # Determine Status String
    regime_str = "SAFE" if is_regime_safe else "UNSTABLE"
    
    # Update DB
    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        UPDATE system_state 
        SET last_update_ts=?, shock_block_remaining=?, regime_status=?
        WHERE id=1
    """, (ts_now, new_shock_remaining, regime_str))
    
    conn.commit()
    conn.close()
    print(f"[EOD] System State Updated: Regime={regime_str}, ShockRem={new_shock_remaining}")

def step_4_mark_positions(yymm):
    """
    Load the OFFICIAL EOD Model.
    Mark all OPEN positions.
    If they violate rules (Stop/MaxHold), update status to 'PENDING_CLOSE'.
    """
    print(f"[EOD] Step 4: Marking Positions against Official Close...")
    
    # 1. Load Official EOD Data
    enh_path = Path(cr.PATH_ENH) / f"{yymm}_enh.parquet"
    if not enh_path.exists():
        print("[ERROR] EOD Model file missing.")
        return

    df_enh = pd.read_parquet(enh_path)
    last_ts = df_enh['ts'].max()
    
    # Get Snapshot of the last timestamp
    snapshot = df_enh[df_enh['ts'] == last_ts].set_index('tenor_yrs')
    # Helper to get Z
    def get_z(tenor):
        # Find nearest tenor in snapshot
        idx = snapshot.index.get_indexer([tenor], method='nearest')[0]
        return snapshot.iloc[idx]['z_comb']

    # 2. Get Open Positions
    conn = get_db_connection(DB_POSITIONS)
    trades = pd.read_sql("SELECT * FROM trades WHERE status='OPEN'", conn)
    
    if trades.empty:
        print("[EOD] No open positions to mark.")
        conn.close()
        return

    # 3. Iterate and Check Rules
    for _, t in trades.iterrows():
        trade_id = t['trade_id']
        t_pay = t['tenor_pay']
        t_rec = t['tenor_rec']
        entry_z = t['entry_z_spread']
        
        # Calculate Official EOD Z-Spread
        z_pay = get_z(t_pay)
        z_rec = get_z(t_rec)
        curr_z = z_pay - z_rec
        
        # Calculate Days Held (Simple approx)
        open_dt = pd.to_datetime(t['open_ts'])
        days_held = (pd.Timestamp.now() - open_dt).days
        
        # --- RULE CHECKING (Matches portfolio_test.py) ---
        new_status = 'OPEN'
        reason = None
        
        # 1. Reversion (Take Profit)
        if abs(curr_z) < cr.Z_EXIT:
            new_status = 'FLAGGED_CLOSE'
            reason = 'reversion'
            
        # 2. Stop Loss
        elif abs(curr_z - entry_z) > cr.Z_STOP:
            new_status = 'FLAGGED_CLOSE'
            reason = 'stop_loss'
            
        # 3. Max Hold
        elif days_held >= cr.MAX_HOLD_DAYS:
            new_status = 'FLAGGED_CLOSE'
            reason = 'max_hold'
            
        # Update DB if flag changed
        if new_status == 'FLAGGED_CLOSE':
            print(f" -> Flagging Trade {trade_id} for {reason} (Z: {curr_z:.2f})")
            # We store the reason so the trader sees it tomorrow
            conn.execute("""
                UPDATE trades 
                SET close_reason = ? 
                WHERE trade_id = ?
            """, (reason, trade_id))
            # Note: We do NOT set status to CLOSED. We leave it OPEN (or FLAGGED)
            # The UI Blotter will highlight it based on the close_reason.
            
    conn.commit()
    conn.close()
    print("[EOD] Position Marking Complete.")

def run_eod_main():
    # Determine current YYMM based on today
    now = datetime.datetime.now()
    yymm = now.strftime("%y%m")
    
    print("="*40)
    print(f"STARTING EOD BATCH FOR {yymm}")
    print("="*40)
    
    if step_1_export_ticks_to_parquet(yymm):
        if step_2_build_model(yymm):
            step_3_update_signals_and_state()
            step_4_mark_positions(yymm)
            
    print("="*40)
    print("EOD BATCH COMPLETED")
    print("="*40)

if __name__ == "__main__":
    run_eod_main()
