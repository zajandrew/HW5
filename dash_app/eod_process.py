import pandas as pd
import numpy as np
import sqlite3
import sys
import time
import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta

# --- Hook into Parent Directory to import Research Libraries ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import cr_config as cr
import feature_creation as fc
import hybrid_filter as hf

# --- Constants ---
DB_MARKET = "market_data.db"
DB_POSITIONS = "positions.db"

def get_db_connection(db_name, retries=5):
    """
    Robust connection with retry logic to handle Recorder locks.
    """
    for i in range(retries):
        try:
            # timeout=10 tells SQLite driver to busy-wait for 10s
            conn = sqlite3.connect(db_name, timeout=10)
            return conn
        except sqlite3.OperationalError:
            if i < retries - 1:
                time.sleep(1) # Explicit python-side wait
            else:
                raise
    raise Exception(f"Could not connect to {db_name} after {retries} retries.")

def step_1_export_and_stitch(yymm):
    """
    1. Query SQLite for CURRENT month ticks.
    2. Save as {YYMM}_features.parquet.
    
    CRITICAL CHANGE: No longer stitches historical RAW files. 
    Relies on feature_creation.py to load historical SUMMARIES.
    """
    print(f"[EOD] Step 1: Exporting Live Ticks for {yymm}...")
    
    # --- A. Get Live Ticks from SQLite ---
    conn = get_db_connection(DB_MARKET)
    
    # Calculate date filter for SQLite
    # Assuming yymm is "2512" -> "2025-12"
    year = "20" + yymm[:2]
    month = yymm[2:]
    date_filter = f"{year}-{month}"
    
    sql = f"""
        SELECT timestamp, ticker, rate 
        FROM ticks 
        WHERE strftime('%Y-%m', timestamp) = '{date_filter}'
    """
    df_live = pd.read_sql(sql, conn)
    conn.close()
    
    if df_live.empty:
        print(f"[WARN] No live ticks found for {date_filter} in SQLite.")
        return False

    # Pivot to Wide Format
    df_live['ts'] = pd.to_datetime(df_live['timestamp'])
    df_live_wide = df_live.pivot_table(index='ts', columns='ticker', values='rate', aggfunc='last')
    
    # --- B. Save DIRECTLY (No Stitching) ---
    # We simply save the current month's data. 
    # feature_creation.py will load this as the "Target" and load "History" from summaries.
    
    out_path = Path(cr.PATH_DATA) / f"{yymm}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_live_wide.to_parquet(out_path)
    print(f"[EOD] Exported {len(df_live_wide)} rows to {out_path}")
    return True

def step_2_build_model(yymm):
    """
    Run Feature Creation (Spline/PCA).
    Generates {YYMM}_enh.parquet based on the stitched features.
    """
    print(f"[EOD] Step 2: Running Feature Creation for {yymm}...")
    try:
        # Calls research library directly
        fc.build_month(yymm)
        print("[EOD] Model Build Complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Feature Creation failed: {e}")
        return False

def step_3_update_signals_and_state():
    """
    Run Filters (Regime/Shock) based on new data and PnL history.
    Updates 'system_state' table in positions.db.
    """
    print(f"[EOD] Step 3: Updating System State (Regime & Shock)...")
    
    # --- 1. REGIME CALCULATION ---
    # Force rebuild signals using the NEW _enh.parquet file
    signals = hf.get_or_build_hybrid_signals(force_rebuild=True)
    
    if signals.empty:
        print("[WARN] Signals generation failed. Defaulting to UNSTABLE.")
        regime_str = "UNSTABLE"
    else:
        last_signal = signals.iloc[-1]
        health_z = last_signal.get('signal_health_z', -99)
        
        # Logic: If signal health is too low, curve is noisy/untradeable
        if health_z >= cr.MIN_SIGNAL_HEALTH_Z:
            regime_str = "SAFE"
        else:
            regime_str = "UNSTABLE"
            print(f" -> Regime Deteriorated: Health Z {health_z:.2f}")

    # --- 2. SHOCK CALCULATION (PNL FEEDBACK) ---
    conn = get_db_connection(DB_POSITIONS)
    
    # Get current block status
    state_row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    current_block = state_row[2] if state_row else 0
    
    # Check Metric Type from Config (Default Cash)
    metric_type = str(getattr(cr, "SHOCK_METRIC_TYPE", "REALIZED_CASH"))
    
    # Select column based on metric preference
    # We now have explicit columns for BP and CASH in the DB
    if metric_type == "REALIZED_BPS":
        # Sum realized_pnl_bp
        target_col = "realized_pnl_bp"
    else:
        # Sum realized_pnl_cash
        target_col = "realized_pnl_cash"

    sql_query = f"""
        SELECT date(close_ts) as dts, sum({target_col}) as daily_pnl 
        FROM trades 
        WHERE status='CLOSED' 
        GROUP BY date(close_ts) 
        ORDER BY dts
    """

    pnl_df = pd.read_sql(sql_query, conn)
    
    new_shock_detected = False
    
    # Rolling Z-Score Check on PnL
    if len(pnl_df) >= 5: 
        recent_pnl = pnl_df['daily_pnl'].values
        win = int(getattr(cr, "SHOCK_PNL_WINDOW", 10))
        
        if len(recent_pnl) > win:
            slice_pnl = recent_pnl[-win:]
            mu = np.mean(slice_pnl)
            sd = np.std(slice_pnl, ddof=1)
            
            if sd > 1e-6:
                last_pnl = slice_pnl[-1]
                z_score = (last_pnl - mu) / sd
                
                thresh = float(getattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -1.5))
                
                if z_score < thresh:
                    new_shock_detected = True
                    print(f" -> SHOCK DETECTED: Daily PnL ({metric_type}) Z-Score {z_score:.2f} < {thresh}")

    # C. Update Block Counter
    if new_shock_detected:
        # Reset counter to full block length (e.g. 10 days)
        new_block_remaining = int(getattr(cr, "SHOCK_BLOCK_LENGTH", 10))
    else:
        # Decrement existing block (if any)
        new_block_remaining = max(0, current_block - 1)

    # --- 3. SAVE TO DB ---
    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        UPDATE system_state 
        SET last_update_ts=?, shock_block_remaining=?, regime_status=?
        WHERE id=1
    """, (ts_now, new_block_remaining, regime_str))
    
    conn.commit()
    conn.close()
    print(f"[EOD] State Updated: Regime={regime_str}, ShockBlocks={new_block_remaining}")

def step_4_mark_positions(yymm):
    """
    Mark OPEN positions against the new Official EOD Model.
    Sets 'close_reason' flag if Max Hold or Stops are hit.
    """
    print(f"[EOD] Step 4: Marking Positions against Official Close...")
    suffix = cr.ENH_SUFFIX
    enh_path = Path(cr.PATH_ENH) / f"{yymm}_enh{suffix}.parquet"
    if not enh_path.exists():
        print("[ERROR] EOD Model file missing.")
        return

    df_enh = pd.read_parquet(enh_path)
    
    # --- 1. ROBUST SNAPSHOT LOADING (Fixes Fragmentation Bug) ---
    # Round timestamps to nearest minute to handle write delays
    df_enh['ts_group'] = df_enh['ts'].dt.floor('min')
    
    # Find the latest complete batch
    last_batch_time = df_enh['ts_group'].max()
    print(f"[EOD] Loading Snapshot Batch: {last_batch_time}")
    
    # Filter for that batch and deduplicate
    snapshot_subset = df_enh[df_enh['ts_group'] == last_batch_time].copy()
    snapshot_subset = snapshot_subset.sort_values('ts').drop_duplicates(subset=['tenor_yrs'], keep='last')
    
    # Create Dictionary Map for safe lookups (Round keys to avoid float mismatch)
    snapshot_map = {round(row['tenor_yrs'], 4): row['z_comb'] for _, row in snapshot_subset.iterrows()}
    # -------------------------------------------------------------
    
    def get_z_robust(target_tenor):
        t_round = round(target_tenor, 4)
        
        # A. Exact Match
        if t_round in snapshot_map:
            return snapshot_map[t_round]
            
        # B. Nearest Neighbor with Strict Tolerance (e.g. 0.05 years)
        # Prevents 5Y being used for 7Y if 7Y is missing.
        if not snapshot_map: return None
        
        closest_t = min(snapshot_map.keys(), key=lambda k: abs(k - t_round))
        dist = abs(closest_t - t_round)
        
        if dist > 0.05: 
            print(f"   [WARN] No matching tenor for {target_tenor}Y. Nearest is {closest_t}Y (Dist: {dist:.4f}). Skipping.")
            return None
            
        return snapshot_map[closest_t]

    conn = get_db_connection(DB_POSITIONS)
    trades = pd.read_sql("SELECT * FROM trades WHERE status='OPEN'", conn)
    
    if trades.empty:
        print("[EOD] No open positions.")
        conn.close()
        return

    updates = []

    for _, t in trades.iterrows():
        trade_id = t['trade_id']
        
        # 1. Calculate Official Z-Spread using Robust Lookup
        z_pay = get_z_robust(t['tenor_pay'])
        z_rec = get_z_robust(t['tenor_rec'])
        
        if z_pay is None or z_rec is None:
            print(f" -> Trade {trade_id}: Skipped (Missing EOD Data).")
            continue
            
        curr_z = z_pay - z_rec
        
        # 2. Robust Business Day Calculation (String Method)
        raw_ts = str(t['open_ts'])
        date_str_open = raw_ts[:10] if len(raw_ts) >= 10 else datetime.datetime.now().strftime('%Y-%m-%d')
        date_str_now = datetime.datetime.now().strftime('%Y-%m-%d')
        
        try:
            days_held = int(np.busday_count(date_str_open, date_str_now))
        except ValueError:
            days_held = 0
        
        entry_z = t['entry_z_spread']
        reason = None
        
        # 3. Rule Logic
        if abs(curr_z) < cr.Z_EXIT: 
            reason = 'reversion' # Take Profit
        elif abs(curr_z - entry_z) > cr.Z_STOP: 
            reason = 'stop_loss'
        elif days_held >= cr.MAX_HOLD_DAYS: 
            reason = 'max_hold'
            
        if reason:
            print(f" -> Flagging Trade {trade_id} for {reason} (Z: {curr_z:.2f} | Days: {days_held})")
            updates.append((reason, trade_id))
    
    # 4. Batch Execute Updates
    for r, tid in updates:
        conn.execute("UPDATE trades SET close_reason = ? WHERE trade_id = ?", (r, tid))
            
    conn.commit()
    conn.close()
    print("[EOD] Marking Complete.")

def run_eod_main():
    now = datetime.datetime.now()
    yymm = now.strftime("%y%m")
    
    print("="*40)
    print(f"STARTING EOD BATCH FOR {yymm}")
    print("="*40)
    
    if step_1_export_and_stitch(yymm): 
        if step_2_build_model(yymm):
            step_3_update_signals_and_state()
            step_4_mark_positions(yymm)
            
    print("="*40)
    print("EOD BATCH COMPLETED")
    print("="*40)

if __name__ == "__main__":
    run_eod_main()
