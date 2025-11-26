import pandas as pd
import numpy as np
import sqlite3
import sys
import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta

# --- Hook into Research Libraries ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
import cr_config as cr
import feature_creation as fc
import hybrid_filter as hf

# --- Constants ---
DB_MARKET = "market_data.db"
DB_POSITIONS = "positions.db"

def get_db_connection(db_name):
    return sqlite3.connect(db_name)

def get_lookback_files(current_yymm, months_back=6):
    """
    Finds the previous N months of _features.parquet files.
    Used to stitch history so PCA works correctly.
    """
    # Convert YYMM to datetime
    curr_date = datetime.datetime.strptime(current_yymm, "%y%m")
    
    found_files = []
    
    # Loop backwards
    for i in range(1, months_back + 1):
        prev_date = curr_date - relativedelta(months=i)
        prev_yymm = prev_date.strftime("%y%m")
        
        path = Path(cr.PATH_DATA) / f"{prev_yymm}_features.parquet"
        if path.exists():
            found_files.append(path)
        else:
            print(f"[WARN] Missing history file: {path}. PCA might be unstable.")
            
    # Return sorted (oldest first)
    return sorted(found_files)

def step_1_export_and_stitch(yymm):
    """
    1. Query SQLite for CURRENT month ticks.
    2. Load PREVIOUS 6 months of parquet files.
    3. Stitch together.
    4. Save as {YYMM}_features.parquet.
    """
    print(f"[EOD] Step 1: Stitching History + Live Ticks for {yymm}...")
    
    # --- A. Get Live Ticks from SQLite ---
    conn = get_db_connection(DB_MARKET)
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
    
    df_live_wide = pd.DataFrame()
    if not df_live.empty:
        df_live['ts'] = pd.to_datetime(df_live['timestamp'])
        df_live_wide = df_live.pivot_table(index='ts', columns='ticker', values='rate', aggfunc='last')
    else:
        print(f"[WARN] No live ticks found for {date_filter}.")

    # --- B. Get Historical Data ---
    hist_files = get_lookback_files(yymm, months_back=6)
    
    dfs_to_concat = []
    
    # Load history
    for p in hist_files:
        try:
            d = pd.read_parquet(p)
            dfs_to_concat.append(d)
        except Exception as e:
            print(f"[ERROR] Failed to read {p}: {e}")
            
    # Add live data
    if not df_live_wide.empty:
        dfs_to_concat.append(df_live_wide)
        
    if not dfs_to_concat:
        print("[ERROR] No data available (History or Live). Aborting.")
        return False
        
    # --- C. Stitch and Save ---
    # Combine
    df_final = pd.concat(dfs_to_concat).sort_index()
    
    # Deduplicate: If we re-run EOD, don't duplicate overlapping timestamps
    df_final = df_final[~df_final.index.duplicated(keep='last')]
    
    # Save
    out_path = Path(cr.PATH_DATA) / f"{yymm}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_parquet(out_path)
    print(f"[EOD] Stitched {len(df_final)} rows. Saved to {out_path}")
    return True

def step_2_build_model(yymm):
    """
    Run Feature Creation. 
    Because we stitched history in Step 1, PCA will see the full lookback window.
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
    print(f"[EOD] Step 3: Updating System State (Regime & Shock)...")
    
    # --- 1. REGIME CALCULATION ---
    # Rebuild hybrid signals using the NEW _enh.parquet file we just made
    signals = hf.get_or_build_hybrid_signals(force_rebuild=True)
    
    if signals.empty:
        print("[WARN] Signals generation failed. Defaulting to UNSTABLE.")
        regime_str = "UNSTABLE"
    else:
        last_signal = signals.iloc[-1]
        # Check Signal Health Z-Score against Config
        health_z = last_signal.get('signal_health_z', -99)
        if health_z >= cr.MIN_SIGNAL_HEALTH_Z:
            regime_str = "SAFE"
        else:
            regime_str = "UNSTABLE"
            print(f" -> Regime Deteriorated: Health Z {health_z:.2f}")

    # --- 2. SHOCK CALCULATION (PNL FEEDBACK) ---
    conn = get_db_connection(DB_POSITIONS)
    
    # A. Get current state to see if we are already blocked
    state_row = conn.execute("SELECT * FROM system_state WHERE id=1").fetchone()
    current_block = state_row[2] if state_row else 0
    
    # B. Calculate Recent Daily PnL Stats
    # We sum realized_pnl_cash by close_ts (bucketed by day)
    # Note: In a real prod system, you might also include MTM PnL of open positions here
    pnl_df = pd.read_sql("""
        SELECT date(close_ts) as dts, sum(realized_pnl_cash) as daily_pnl 
        FROM trades 
        WHERE status='CLOSED' 
        GROUP BY date(close_ts) 
        ORDER BY dts
    """, conn)
    
    new_shock_detected = False
    
    if len(pnl_df) >= 5: # Need minimum history to calc stats
        recent_pnl = pnl_df['daily_pnl'].values
        # Lookback window from config
        win = int(getattr(cr, "SHOCK_PNL_WINDOW", 10))
        
        if len(recent_pnl) > win:
            slice_pnl = recent_pnl[-win:]
            mu = np.mean(slice_pnl)
            sd = np.std(slice_pnl, ddof=1)
            
            if sd > 1e-6: # Avoid div by zero
                # Check the MOST RECENT day's Z-score
                last_pnl = slice_pnl[-1]
                z_score = (last_pnl - mu) / sd
                
                thresh = float(getattr(cr, "SHOCK_RAW_PNL_Z_THRESH", -1.5))
                if z_score < thresh:
                    new_shock_detected = True
                    print(f" -> SHOCK DETECTED: Daily PnL Z-Score {z_score:.2f} < {thresh}")

    # C. Update Block Counter
    if new_shock_detected:
        # Reset counter to full block length
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
    print(f"[EOD] Step 4: Marking Positions against Official Close...")
    
    enh_path = Path(cr.PATH_ENH) / f"{yymm}_enh.parquet"
    if not enh_path.exists():
        print("[ERROR] EOD Model file missing.")
        return

    df_enh = pd.read_parquet(enh_path)
    last_ts = df_enh['ts'].max()
    
    snapshot = df_enh[df_enh['ts'] == last_ts].set_index('tenor_yrs')
    
    def get_z(tenor):
        # Nearest neighbor lookup
        if tenor not in snapshot.index:
             # try finding nearest
             idx = snapshot.index.get_indexer([tenor], method='nearest')[0]
             return snapshot.iloc[idx]['z_comb']
        return snapshot.loc[tenor]['z_comb']

    conn = get_db_connection(DB_POSITIONS)
    trades = pd.read_sql("SELECT * FROM trades WHERE status='OPEN'", conn)
    
    if trades.empty:
        print("[EOD] No open positions.")
        conn.close()
        return

    for _, t in trades.iterrows():
        trade_id = t['trade_id']
        curr_z = get_z(t['tenor_pay']) - get_z(t['tenor_rec'])
        
        open_dt = pd.to_datetime(t['open_ts'])
        days_held = (pd.Timestamp.now() - open_dt).days
        
        entry_z = t['entry_z_spread']
        reason = None
        
        # Rule Logic
        if abs(curr_z) < cr.Z_EXIT: reason = 'reversion'
        elif abs(curr_z - entry_z) > cr.Z_STOP: reason = 'stop_loss'
        elif days_held >= cr.MAX_HOLD_DAYS: reason = 'max_hold'
            
        if reason:
            print(f" -> Flagging Trade {trade_id} for {reason} (Z: {curr_z:.2f})")
            conn.execute("UPDATE trades SET close_reason = ? WHERE trade_id = ?", (reason, trade_id))
            
    conn.commit()
    conn.close()
    print("[EOD] Marking Complete.")

def run_eod_main():
    now = datetime.datetime.now()
    yymm = now.strftime("%y%m")
    
    print(f"STARTING EOD BATCH FOR {yymm}")
    # Step 1 now stitches history automatically
    if step_1_export_and_stitch(yymm): 
        if step_2_build_model(yymm):
            step_3_update_signals_and_state()
            step_4_mark_positions(yymm)
    print("EOD BATCH COMPLETED")

if __name__ == "__main__":
    run_eod_main()
