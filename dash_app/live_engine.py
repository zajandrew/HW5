import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# --- Hook into parent directory ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import cr_config as cr

# ==============================================================================
# GLOBAL STATE
# ==============================================================================
MODEL_SNAPSHOT = {} 
GLOBAL_FUNDING_ANCHOR = 0.0 
DEFAULT_VOL_PROXY = 0.05 

# In live_engine.py

def load_midnight_model(yymm_str):
    """Loads the EOD parquet file with robust timestamp handling."""
    global MODEL_SNAPSHOT, GLOBAL_FUNDING_ANCHOR
    
    try:
        suffix = cr.ENH_SUFFIX
        path = Path(cr.PATH_ENH) / f"{yymm_str}_enh{suffix}.parquet"
        
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            return False

        df = pd.read_parquet(path)
        if df.empty:
            print(f"[WARN] Model file empty: {path}")
            return False

        # --- FIX: ROBUST TIMESTAMP SELECTION ---
        # 1. Round timestamps to the nearest minute to group fragmented writes
        #    (This handles 'loop-writing' where every row has a slightly diff microsecond)
        df['ts_group'] = df['ts'].dt.floor('min')
        
        # 2. Find the latest "Minute Batch"
        last_batch_time = df['ts_group'].max()
        
        # 3. Filter for that entire batch
        snapshot = df[df['ts_group'] == last_batch_time].copy()
        
        # 4. Deduplicate (just in case) - keep last entry per tenor
        snapshot = snapshot.sort_values('ts').drop_duplicates(subset=['tenor_yrs'], keep='last')
        # ---------------------------------------
        
        if not snapshot.empty:
            min_tenor_idx = snapshot['tenor_yrs'].idxmin()
            # Safety check if index is integer or label
            if isinstance(min_tenor_idx, int) and min_tenor_idx in snapshot.index:
                 GLOBAL_FUNDING_ANCHOR = float(snapshot.loc[min_tenor_idx, 'rate'])
            else:
                 # Fallback if index reset
                 GLOBAL_FUNDING_ANCHOR = float(snapshot.loc[snapshot['tenor_yrs'].idxmin()]['rate'])
        
        has_scale = 'scale' in snapshot.columns
        has_vol = 'vol' in snapshot.columns
        
        MODEL_SNAPSHOT = snapshot.set_index('tenor_yrs').to_dict('index')
        
        for t, row in MODEL_SNAPSHOT.items():
            if has_scale:
                row['vol_proxy'] = float(row['scale'])
            elif has_vol:
                row['vol_proxy'] = float(row['vol'])
            else:
                row['vol_proxy'] = DEFAULT_VOL_PROXY
        
        print(f"[SYSTEM] Loaded Midnight Model: {last_batch_time} ({len(MODEL_SNAPSHOT)} tenors). Funding Anchor: {GLOBAL_FUNDING_ANCHOR:.4f}%")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==============================================================================
# HELPER FUNCTIONS (MATCHING PORTFOLIO_TEST)
# ==============================================================================

def assign_bucket(tenor):
    """Matches bucket logic from cr_config."""
    buckets = getattr(cr, "BUCKETS", {})
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

# In live_engine.py

def _get_funding_rate_live(live_map):
    """Finds proxy for funding rate (shortest available tenor)."""
    # Default to Model Anchor
    funding_rate = GLOBAL_FUNDING_ANCHOR if GLOBAL_FUNDING_ANCHOR > 0 else 4.0
    
    if live_map:
        valid_keys = [k for k in live_map.keys() if k in cr.TENOR_YEARS]
        if valid_keys:
            shortest_ticker = min(valid_keys, key=lambda k: cr.TENOR_YEARS[k])
            shortest_tenor = cr.TENOR_YEARS[shortest_ticker]
            
            if shortest_tenor <= 1.0:
                live_val = live_map[shortest_ticker]
                # --- FIX: Only use live if valid, otherwise keep anchor ---
                if live_val > 0.0001:
                    funding_rate = live_val
                    
    return funding_rate

def _get_z_at_tenor_dict(snapshot_dict, tenor, tol=None):
    """
    Finds Z-Score in the dictionary snapshot with tolerance.
    Adapts portfolio_test._get_z_at_tenor to work with Dict instead of DataFrame.
    """
    if tol is None: tol = float(getattr(cr, "FLY_TENOR_TOL_YEARS", 0.02))
    
    best_t, best_dist = None, float("inf")
    
    for t_key in snapshot_dict.keys():
        dist = abs(t_key - tenor)
        if dist < best_dist:
            best_dist = dist
            best_t = t_key
            
    if best_t is not None and best_dist <= tol:
        return float(snapshot_dict[best_t].get('z_comb', 0.0))
    return None

def compute_fly_z_dict(snapshot_dict, a, b, c):
    """Adapts compute_fly_z for Dict snapshot."""
    try:
        z_a = _get_z_at_tenor_dict(snapshot_dict, a)
        z_b = _get_z_at_tenor_dict(snapshot_dict, b)
        z_c = _get_z_at_tenor_dict(snapshot_dict, c)
        
        if any(v is None for v in (z_a, z_b, z_c)): return None
        
        # Calculate std dev of all Zs in snapshot for scaling (naive approx matching p_test)
        xs = [float(r.get('z_comb', 0.0)) for r in snapshot_dict.values()]
        sd = np.nanstd(xs, ddof=1) if len(xs) > 1 else 1.0
        
        return (0.5*(z_a + z_c) - z_b) / (sd if sd > 0 else 1.0)
    except Exception:
        return None

def fly_alignment_ok_live(leg_tenor, leg_sign_z, snapshot_dict, zdisp_for_pair=None):
    """
    Checks if buying/selling this tenor fights a broken butterfly.
    """
    if not getattr(cr, "FLY_ENABLE", True) or getattr(cr, "FLY_MODE", "tolerant") == "off": return True
    
    # Allow big Z-Dislocations to override fly checks
    if getattr(cr, "FLY_ALLOW_BIG_ZDISP", True) and zdisp_for_pair:
        z_entry = float(getattr(cr, "Z_ENTRY", 0.75))
        big_margin = float(getattr(cr, "FLY_BIG_ZDISP_MARGIN", 0.20))
        if zdisp_for_pair >= (z_entry + big_margin):
            return True
    
    skip = getattr(cr, "FLY_SKIP_SHORT_UNDER", None)
    if skip and leg_tenor < float(skip): return True

    triplets = getattr(cr, "FLY_DEFS", [])
    if getattr(cr, "FLY_NEIGHBOR_ONLY", True):
        W = float(getattr(cr, "FLY_WINDOW_YEARS", 3.0))
        triplets = [t for t in triplets if abs(t[1] - leg_tenor) <= W]
        if not triplets: return True

    contras = 0
    min_z = float(getattr(cr, "FLY_Z_MIN", 0.8))
    
    for (a,b,c) in triplets:
        fz = compute_fly_z_dict(snapshot_dict, a, b, c)
        # If fly is broken (high |Z|) and we are betting against it:
        if fz and abs(fz) >= min_z and np.sign(fz) * np.sign(leg_sign_z) < 0:
            contras += 1

    mode = getattr(cr, "FLY_MODE", "tolerant").lower()
    req = int(getattr(cr, "FLY_REQUIRE_COUNT", 2))
    
    if mode == "strict": return contras == 0
    if mode == "loose": return contras <= 1
    return contras <= req

# ==============================================================================
# CORE CALCULATIONS
# ==============================================================================

def get_live_z_scores(live_rates_map):
    """Projects Live Rates onto the Frozen Model."""
    if not MODEL_SNAPSHOT:
        return {}

    z_map = {}
    
    for ticker, live_rate in live_rates_map.items():
        if live_rate <= 0.00001: continue
        
        tenor = cr.TENOR_YEARS.get(ticker)
        
        if tenor is None or tenor not in MODEL_SNAPSHOT:
            continue
            
        model_row = MODEL_SNAPSHOT[tenor]
        model_rate = float(model_row.get('rate', live_rate))
        model_z = float(model_row.get('z_comb', 0.0))
        vol_proxy = float(model_row.get('vol_proxy', DEFAULT_VOL_PROXY))

        if vol_proxy == 0: vol_proxy = DEFAULT_VOL_PROXY

        rate_diff = live_rate - model_rate
        z_live = model_z + (rate_diff / vol_proxy)
        
        z_map[ticker] = {
            'z_live': z_live, 
            'tenor': tenor, 
            'model_z': model_z,
            'model_rate': model_rate,
            'vol_proxy': vol_proxy
        }
    return z_map

def calc_live_drift(tenor, direction, live_map):
    """
    Calculates Daily Drift (Carry + Roll) in BPS.
    Matches portfolio_test.py logic.
    direction: -1.0 (Pay/Short), +1.0 (Rec/Long)
    """
    # 1. Get Candidate Fixed Rate
    rate_fixed = 0.0
    
    # Reverse lookup ticker for live map check
    ticker = None
    for k, v in cr.TENOR_YEARS.items():
        if abs(v - tenor) < 0.001:
            ticker = k
            break
            
    if ticker and ticker in live_map:
        rate_fixed = live_map[ticker]
    elif tenor in MODEL_SNAPSHOT:
        rate_fixed = float(MODEL_SNAPSHOT[tenor].get('rate', 0.0))
    else:
        return -999.0

    # 2. Get Funding Rate
    funding_rate = _get_funding_rate_live(live_map)

    # 3. Calculate Rolldown
    # Build sorted composite curve for interpolation
    # CRITICAL FIX: Ensure 'xp' is sorted for np.interp
    curve_data = {t: r.get('rate', 0.0) for t, r in MODEL_SNAPSHOT.items()}
    for k, v in live_map.items():
        t_live = cr.TENOR_YEARS.get(k)
        if t_live: curve_data[t_live] = v
    
    xp = sorted(curve_data.keys())
    fp = [curve_data[t] for t in xp]

    if not xp: return 0.0

    # Roll 1 day (Act/360)
    day_fraction = 1.0 / 360.0
    t_roll = max(0.0, tenor - day_fraction)
    
    # Linear Interpolation
    rate_rolled = np.interp(t_roll, xp, fp)

    # 4. Total Drift (BPS)
    # Carry = (Fixed - Float) * 100 * (1/360)
    raw_carry = (rate_fixed - funding_rate) * 100.0 * day_fraction
    
    # Roll = (Curr - Rolled) * 100
    raw_roll = (rate_fixed - rate_rolled) * 100.0
    
    # Total
    total_drift = (raw_carry + raw_roll) * direction
    return total_drift

def get_valid_partners(active_ticker, direction, live_z_map, live_map):
    """
    Finds candidates based on RELATIVE NORMALIZED SCORE (Option B).
    Matches portfolio_test.py 'Overlay' logic exactly.
    """
    active_tenor = cr.TENOR_YEARS.get(active_ticker)
    if active_ticker not in live_z_map:
        return []

    active_z = live_z_map[active_ticker]['z_live']
    candidates = []
    
    # Configs
    min_alt_tenor = getattr(cr, "ALT_LEG_TENOR_YEARS", 0.0)
    min_sep = getattr(cr, "MIN_SEP_YEARS", 0.5)
    
    # Use "Option B" Thresholds
    DRIFT_GATE = float(getattr(cr, "DRIFT_GATE_BPS", -0.5))
    DRIFT_WEIGHT = float(getattr(cr, "DRIFT_WEIGHT", 2.0))
    Z_ENTRY = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_EXTRA = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))
    
    # Direction Scalar:
    # If input is 'PAY' (Short), side_s = -1.0
    # If input is 'REC' (Long), side_s = +1.0
    dir_scalar = -1.0 if direction == 'PAY' else 1.0

    # 1. Calculate Baseline Drift (The trade we are hedging/replacing)
    # We use the SAME direction scalar, because we want to know 
    # "Does candidate pay me MORE than the hedge leg?"
    drift_exec = calc_live_drift(active_tenor, dir_scalar, live_map)

    for ticker, info in live_z_map.items():
        tenor = info['tenor']
        z_live = info['z_live']
        
        # --- Constraints (Matching portfolio_test) ---
        if tenor == active_tenor: continue
        if tenor < min_alt_tenor: continue
        
        dist = abs(tenor - active_tenor)
        if dist < min_sep: continue
        
        # Bucket Check: Prevent crossing Short/Long buckets
        b_exec = assign_bucket(active_tenor)
        b_cand = assign_bucket(tenor)
        
        if b_cand == "short" and b_exec == "long": continue
        if b_exec == "short" and b_cand == "long": continue
        
        # --- 1. Z-Spread ---
        # "disp" in portfolio_test. 
        # If buying (REC), we want High Z (Cheap). disp = Z_Cand - Z_Exec
        # If selling (PAY), we want Low Z (Rich). disp = Z_Exec - Z_Cand
        z_spread = (z_live - active_z) if dir_scalar > 0 else (active_z - z_live)
        
        # Effective Entry Threshold (Assuming Standard Size since no DV01 input here)
        z_threshold = Z_ENTRY
        
        # Short End Padding
        if (b_cand == "short" or b_exec == "short"):
            z_threshold += SHORT_EXTRA
            
        if z_spread < z_threshold: continue

        # --- 2. Fly Gate ---
        # Check Candidate (Leg 1 - New)
        # Note: If we Rec Candidate, we are buying. Sign = +1.
        if not fly_alignment_ok_live(tenor, 1.0, MODEL_SNAPSHOT, zdisp_for_pair=z_spread): continue
        
        # Check Execution (Leg 2 - Replaced)
        # Note: We are replacing the hedge. If Hedge was Rec, we Rec Candidate.
        # But 'fly_alignment_ok' asks: is this specific point compatible with our direction?
        # If we are effectively "Selling" the execution leg (by not doing it), sign is -1.
        if not fly_alignment_ok_live(active_tenor, -1.0, MODEL_SNAPSHOT, zdisp_for_pair=z_spread): continue

        # --- 3. Net Drift & Normalization (Option B) ---
        drift_alt = calc_live_drift(tenor, dir_scalar, live_map)
        
        # Net Advantage: How much better is Candidate than Exec?
        net_drift_bps = drift_alt - drift_exec
        
        # Scaling (Per Year of Extension)
        scaling_factor = dist # Matches portfolio_test logic strictly (dist_years)
        if scaling_factor == 0: scaling_factor = 1.0 # Safety
        
        norm_drift_bps = net_drift_bps / scaling_factor
        
        # --- 4. Gate & Score ---
        if net_drift_bps < DRIFT_GATE: continue
        
        score = z_spread + (norm_drift_bps * DRIFT_WEIGHT)

        candidates.append({
            'ticker': ticker,
            'tenor': tenor,
            'z_live': z_live,
            'spread_imp': z_spread,
            'drift_bps': drift_alt,
            'net_drift_bps': net_drift_bps,
            'norm_drift_bps': norm_drift_bps,
            'composite': score,
            'is_gated': False # If we are here, it passed the gate
        })
        
    # Sort by COMPOSITE Score (High to Low)
    candidates.sort(key=lambda x: x['composite'], reverse=True)
    return candidates

def calculate_live_pnl(row, live_map, now_dt):
    """
    Calculates Price, Carry, and Rolldown PnL using Fractional Time.
    This enables 'Live' intraday accretion rather than waiting for EOD.
    """
    # 1. Calculate Fractional Time (Act/360 basis)
    entry_dt = pd.to_datetime(row['open_ts'])
    time_diff = now_dt - entry_dt
    
    # Use total seconds to get fractional days (e.g., 1.5 days instead of 1)
    # This allows carry to accrue smoothly every second the app refreshes.
    dt_days = max(0.0, time_diff.total_seconds() / 86400.0)
    
    # 2. Get Rates
    entry_pay = float(row['entry_rate_pay'])
    entry_rec = float(row['entry_rate_rec'])
    
    # Get current rates (fallback to entry rate if ticker missing from feed)
    curr_pay = float(live_map.get(row['ticker_pay'], entry_pay))
    curr_rec = float(live_map.get(row['ticker_rec'], entry_rec))
    
    # Get funding (Float Leg)
    funding_rate = _get_funding_rate_live(live_map)

    dv01 = float(row['dv01'])
    
    # -------------------------------------------------------
    # A. Price PnL (Mark-to-Market)
    # -------------------------------------------------------
    # Price moves instantly based on rate change. No time decay here.
    # Pay Leg (Short): Rate Up -> Price Down (Gain for Short).
    # Rec Leg (Long):  Rate Up -> Price Down (Loss for Long).
    
    prc_pay_bp = (curr_pay - entry_pay) * 100.0  # (Current - Entry) is positive if rate rose. 
    prc_rec_bp = (entry_rec - curr_rec) * 100.0  # (Entry - Current) is negative if rate rose.
    
    # -------------------------------------------------------
    # B. Carry PnL (Accrual)
    # -------------------------------------------------------
    # Formula: (FixedRate - FloatRate) * Direction * Time
    # Pay Leg (-1): Pays Fixed, Receives Float.
    # Rec Leg (+1): Receives Fixed, Pays Float.
    
    cry_pay_bp = (entry_pay - funding_rate) * 100.0 * (-1.0) * (dt_days / 360.0)
    cry_rec_bp = (entry_rec - funding_rate) * 100.0 * (+1.0) * (dt_days / 360.0)
    
    # -------------------------------------------------------
    # C. Rolldown PnL (Curve Slide)
    # -------------------------------------------------------
    # We interpolate where the yield *would be* if we slid down the curve by dt_days.
    rol_pay_bp, rol_rec_bp = 0.0, 0.0
    
    # Build curve for interpolation (Model + Live updates)
    curve_data = {t: r.get('rate', 0.0) for t, r in MODEL_SNAPSHOT.items()}
    for k, v in live_map.items():
        t = cr.TENOR_YEARS.get(k)
        if t: curve_data[t] = v
            
    # CRITICAL: Sort keys for np.interp to work correctly
    xp = sorted(curve_data.keys())
    fp = [curve_data[t] for t in xp]
    
    if xp:
        # Pay Leg (Short): If rolled yield < current yield, Price is higher. 
        # Since we are Short, Higher Price = Loss.
        t_roll_pay = max(0.0, float(row['tenor_pay']) - (dt_days/360.0))
        y_roll_pay = np.interp(t_roll_pay, xp, fp)
        rol_pay_bp = (curr_pay - y_roll_pay) * 100.0 * (-1.0)
        
        # Rec Leg (Long): If rolled yield < current yield, Price is higher.
        # Since we are Long, Higher Price = Gain.
        t_roll_rec = max(0.0, float(row['tenor_rec']) - (dt_days/360.0))
        y_roll_rec = np.interp(t_roll_rec, xp, fp)
        rol_rec_bp = (curr_rec - y_roll_rec) * 100.0 * (+1.0)

    # -------------------------------------------------------
    # D. Aggregation
    # -------------------------------------------------------
    total_price_bp = prc_pay_bp + prc_rec_bp
    total_carry_bp = cry_pay_bp + cry_rec_bp
    total_roll_bp  = rol_pay_bp + rol_rec_bp
    total_bp       = total_price_bp + total_carry_bp + total_roll_bp
    
    cash_tuple = (
        total_bp * dv01,
        total_price_bp * dv01,
        total_carry_bp * dv01,
        total_roll_bp * dv01
    )
    
    bp_tuple = (
        total_bp,
        total_price_bp,
        total_carry_bp,
        total_roll_bp
    )
    
    return cash_tuple, bp_tuple

