# portfolio_test.py
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# All config access via module namespace
import cr_config as cr


# ------------------------
# Utilities / conventions
# ------------------------
Path(getattr(cr, "PATH_OUT", ".")).mkdir(parents=True, exist_ok=True)

def _to_float(x, default=np.nan):
    """Safe scalar float extraction (works for scalar, 1-elem Series, or np types)."""
    try:
        if isinstance(x, (pd.Series, pd.Index)):
            if len(x) == 0:
                return default
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return default

def pv01_proxy(tenor_yrs, rate_pct):
    """Simple PV01 proxy so pair is roughly DV01-neutral."""
    return tenor_yrs / max(1e-6, 1.0 + 0.01 * rate_pct)

def assign_bucket(tenor):
    buckets = getattr(cr, "BUCKETS", {})
    for name, (lo, hi) in buckets.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

# ------------------------
# Fly alignment (optional)
# ------------------------
def _row_for_tenor(snap_last: pd.DataFrame, tenor: float) -> pd.Series | None:
    r = snap_last.loc[snap_last["tenor_yrs"] == tenor]
    if r.empty:
        return None
    return r.iloc[0]

def _get_z_at_tenor(snap_last: pd.DataFrame, tenor: float, tol: float | None = None) -> float | None:
    if tol is None:
        tol = float(getattr(cr, "FLY_TENOR_TOL_YEARS", 0.02))
    t = float(tenor)
    s = snap_last[["tenor_yrs", "z_comb"]].dropna()
    if s.empty:
        return None
    # choose nearest tenor within tolerance
    s = s.assign(_dist=(s["tenor_yrs"] - t).abs())
    row = s.loc[s["_dist"].idxmin()]
    if row["_dist"] <= tol:
        return float(row["z_comb"])
    return None

def compute_fly_z(snap_last: pd.DataFrame, a: float, b: float, c: float) -> float | None:
    try:
        z_a = _get_z_at_tenor(snap_last, float(a))
        z_b = _get_z_at_tenor(snap_last, float(b))
        z_c = _get_z_at_tenor(snap_last, float(c))
        if any(v is None for v in (z_a, z_b, z_c)):
            return None
        xs = snap_last["z_comb"].astype(float).to_numpy()
        sd = np.nanstd(xs, ddof=1) if xs.size > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        fly_raw = 0.5*(z_a + z_c) - z_b
        return fly_raw / sd
    except Exception:
        return None

def fly_alignment_ok(
    leg_tenor: float,
    leg_sign_z: float,                 # +1 for cheap (expect z↓), -1 for rich (expect z↑)
    snap_last: pd.DataFrame,
    *,
    zdisp_for_pair: float | None = None
) -> bool:
    FLY_ENABLE          = bool(getattr(cr, "FLY_ENABLE", True))
    FLY_MODE            = str(getattr(cr, "FLY_MODE", "tolerant")).lower()
    FLY_DEFS            = list(getattr(cr, "FLY_DEFS", []))
    FLY_Z_MIN           = float(getattr(cr, "FLY_Z_MIN", 0.8))
    FLY_REQUIRE_COUNT   = int(getattr(cr, "FLY_REQUIRE_COUNT", 2))
    FLY_NEIGHBOR_ONLY   = bool(getattr(cr, "FLY_NEIGHBOR_ONLY", True))
    FLY_WINDOW_YEARS    = float(getattr(cr, "FLY_WINDOW_YEARS", 3.0))
    FLY_SKIP_SHORT_UNDER= getattr(cr, "FLY_SKIP_SHORT_UNDER", None)
    FLY_ALLOW_BIG_ZDISP = bool(getattr(cr, "FLY_ALLOW_BIG_ZDISP", True))
    FLY_BIG_ZDISP_MARGIN= float(getattr(cr, "FLY_BIG_ZDISP_MARGIN", 0.20))
    Z_ENTRY             = float(getattr(cr, "Z_ENTRY", 0.75))

    if not FLY_ENABLE or FLY_MODE == "off":
        return True

    # Big dispersion waiver
    if (FLY_ALLOW_BIG_ZDISP and (zdisp_for_pair is not None) and
        (float(zdisp_for_pair) >= float(Z_ENTRY) + float(FLY_BIG_ZDISP_MARGIN))):
        return True

    # Skip short end
    if (FLY_SKIP_SHORT_UNDER is not None) and (leg_tenor < float(FLY_SKIP_SHORT_UNDER)):
        return True

    triplets = FLY_DEFS
    if FLY_NEIGHBOR_ONLY:
        W = float(FLY_WINDOW_YEARS)
        triplets = [(a,b,c) for (a,b,c) in FLY_DEFS if abs(float(b) - float(leg_tenor)) <= W]
        if not triplets:
            return True

    contradictions = 0
    for (a, b, c) in triplets:
        fz = compute_fly_z(snap_last, a, b, c)
        if fz is None or not np.isfinite(fz) or abs(fz) < float(FLY_Z_MIN):
            continue
        # CONTRADICTION when sign(fly)*sign(leg) < 0
        if np.sign(fz) * np.sign(leg_sign_z) < 0:
            contradictions += 1

    if FLY_MODE == "strict":
        return contradictions == 0
    if FLY_MODE == "loose":
        return contradictions <= 1
    if FLY_MODE == "tolerant":
        return contradictions <= int(FLY_REQUIRE_COUNT)
    return True

# ------------------------
# Pair object
# ------------------------
class PairPos:
    def __init__(self, open_ts, cheap_row, rich_row, w_i, w_j, decisions_per_day: int):
        self.open_ts = open_ts

        self.tenor_i = _to_float(cheap_row["tenor_yrs"])
        self.rate_i  = _to_float(cheap_row["rate"])
        self.tenor_j = _to_float(rich_row["tenor_yrs"])
        self.rate_j  = _to_float(rich_row["rate"])

        self.w_i = float(w_i); self.w_j = float(w_j)

        zi = _to_float(cheap_row["z_comb"])
        zj = _to_float(rich_row["z_comb"])
        self.entry_zspread = zi - zj

        self.closed = False
        self.close_ts = None
        self.exit_reason = None
        self.pnl = 0.0

        # bookkeeping / aging
        self.decisions_per_day = decisions_per_day
        self.age_decisions = 0  # increments by 1 each mark

        self.bucket_i = assign_bucket(self.tenor_i)
        self.bucket_j = assign_bucket(self.tenor_j)

        # attribution proxy
        self.last_zspread = self.entry_zspread
        self.conv_pnl_proxy = 0.0

    def mark(self, snap_last: pd.DataFrame):
        """Mark-to-market at decision time using last rate per tenor; update convergence proxy."""
        ri = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "rate"])
        rj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "rate"])

        # Pair PnL: (old - new) * weight * 100  (rates in % → *100 to bps)
        d_i = (self.rate_i - ri) * self.w_i * 100.0
        d_j = (self.rate_j - rj) * self.w_j * 100.0
        self.pnl = d_i + d_j

        zi = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_i, "z_comb"])
        zj = _to_float(snap_last.loc[snap_last["tenor_yrs"] == self.tenor_j, "z_comb"])
        zsp = zi - zj

        if np.isfinite(zsp) and np.isfinite(self.last_zspread):
            self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp

        # age by one decision step
        self.age_decisions += 1
        return zsp

# ------------------------
# Greedy selector with caps
# ------------------------
def choose_pairs_under_caps(
    snap_last: pd.DataFrame,
    max_pairs: int,
    per_bucket_cap: float,
    total_cap: float,
    front_end_cap: float,
    extra_z_entry: float
):
    """
    Returns list of (cheap_row, rich_row, w_i, w_j).
    Greedy: rank by dispersion, enforce tenor uniqueness & DV01 caps by bucket and total.
    Uses tolerant fly gate; big zdisp can waive fly blocks.
    """
    # minimal columns required
    cols_need = {"tenor_yrs", "rate", "z_comb"}
    if not cols_need.issubset(snap_last.columns):
        return []

    sig = snap_last[list(cols_need)].dropna().copy()
    if sig.empty:
        return []

    # Sort cross-section by z to find extremes
    sig = sig.sort_values("z_comb", kind="mergesort")  # stable sort

    candidates = []
    used_tenors = set()

    # Build pool from a few lowest and highest z points
    low_take  = min(5, len(sig))
    high_take = min(8, len(sig))

    MIN_SEP_YEARS = float(getattr(cr, "MIN_SEP_YEARS", 0.5))
    Z_ENTRY       = float(getattr(cr, "Z_ENTRY", 0.75))
    SHORT_END_EXTRA_Z = float(getattr(cr, "SHORT_END_EXTRA_Z", 0.30))

    for k_low in range(low_take):
        rich = sig.iloc[k_low]
        for k_hi in range(1, high_take + 1):
            cheap = sig.iloc[-k_hi]

            Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
            if Ti in used_tenors or Tj in used_tenors:
                continue

            # minimum tenor separation
            if abs(Ti - Tj) < MIN_SEP_YEARS:
                continue

            zdisp = float(cheap["z_comb"] - rich["z_comb"])
            if zdisp < (Z_ENTRY + float(extra_z_entry)):
                continue

            # tolerant fly gate (can waive if signal already large per config)
            if not fly_alignment_ok(Ti, +1.0, snap_last, zdisp_for_pair=zdisp):
                continue
            if not fly_alignment_ok(Tj, -1.0, snap_last, zdisp_for_pair=zdisp):
                continue

            candidates.append((zdisp, cheap, rich))

    # Greedy pack by decreasing dispersion under DV01 caps
    BUCKETS              = getattr(cr, "BUCKETS", {})
    PER_BUCKET_DV01_CAP  = float(getattr(cr, "PER_BUCKET_DV01_CAP", 1.0))
    TOTAL_DV01_CAP       = float(getattr(cr, "TOTAL_DV01_CAP", 3.0))
    FRONT_END_DV01_CAP   = float(getattr(cr, "FRONT_END_DV01_CAP", 1.0))

    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv01 = 0.0
    selected = []

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs:
            break

        Ti, Tj = float(cheap["tenor_yrs"]), float(rich["tenor_yrs"])
        if Ti in used_tenors or Tj in used_tenors:
            continue

        # PV01-neutral weights within the pair
        pv_i = pv01_proxy(Ti, float(cheap["rate"]))
        pv_j = pv01_proxy(Tj, float(rich["rate"]))
        w_i =  1.0
        w_j = - w_i * pv_i / (pv_j if pv_j != 0 else 1e-12)

        # Bucket caps
        b_i = assign_bucket(Ti)
        b_j = assign_bucket(Tj)
        dv_i = abs(w_i) * pv_i
        dv_j = abs(w_j) * pv_j

        # per-bucket caps
        if b_i in bucket_dv01 and (bucket_dv01[b_i] + dv_i) > PER_BUCKET_DV01_CAP:
            continue
        if b_j in bucket_dv01 and (bucket_dv01[b_j] + dv_j) > PER_BUCKET_DV01_CAP:
            continue

        # front-end aggregate cap
        short_add = (dv_i if b_i == "short" else 0.0) + (dv_j if b_j == "short" else 0.0)
        short_tot = bucket_dv01.get("short", 0.0)
        if (short_tot + short_add) > FRONT_END_DV01_CAP:
            continue

        # total cap
        if (total_dv01 + dv_i + dv_j) > TOTAL_DV01_CAP:
            continue

        # extra threshold if any leg sits in the short bucket
        if (b_i == "short" or b_j == "short") and (zdisp < (Z_ENTRY + SHORT_END_EXTRA_Z)):
            continue

        # accept
        used_tenors.add(Ti); used_tenors.add(Tj)
        bucket_dv01[b_i] = bucket_dv01.get(b_i, 0.0) + dv_i
        bucket_dv01[b_j] = bucket_dv01.get(b_j, 0.0) + dv_j
        total_dv01 += (dv_i + dv_j)

        selected.append((cheap, rich, w_i, w_j))

    return selected

# ------------------------
# Filename helpers (suffixed IO)
# ------------------------
def _enhanced_in_path(yymm: str) -> Path:
    """Use cr.enh_fname if present; else {yymm}_enh{ENH_SUFFIX}.parquet; fallback {yymm}_enh.parquet."""
    if hasattr(cr, "enh_fname") and callable(cr.enh_fname):
        name = cr.enh_fname(yymm)
    else:
        suffix = getattr(cr, "ENH_SUFFIX", "")
        name = f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name

def _positions_out_path() -> Path:
    if hasattr(cr, "positions_fname") and callable(cr.positions_fname):
        name = cr.positions_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"positions_ledger{suffix}.parquet" if suffix else "positions_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _marks_out_path() -> Path:
    if hasattr(cr, "marks_fname") and callable(cr.marks_fname):
        name = cr.marks_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"marks_ledger{suffix}.parquet" if suffix else "marks_ledger.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _pnl_out_path() -> Path:
    if hasattr(cr, "pnl_fname") and callable(cr.pnl_fname):
        name = cr.pnl_fname()
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"pnl_by_bucket{suffix}.parquet" if suffix else "pnl_by_bucket.parquet"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

def _pnl_curve_png(yymm: str) -> Path:
    if hasattr(cr, "pnl_curve_png") and callable(cr.pnl_curve_png):
        name = cr.pnl_curve_png(yymm)
    else:
        suffix = getattr(cr, "OUT_SUFFIX", "")
        name = f"pnl_curve_{yymm}{suffix}.png" if suffix else f"pnl_curve_{yymm}.png"
    return Path(getattr(cr, "PATH_OUT", ".")) / name

# ------------------------
# Month runner
# ------------------------
def run_month(
    yymm: str,
    *,
    decision_freq: str | None = None,
    open_positions: list[PairPos] | None = None,   # carry-in
    carry_in: bool = True                           # if False, ignore any carry-in
):
    """
    Run a single month. If 'open_positions' is provided and carry_in=True,
    they are continued (aged/marked/exited) through this month.
    Returns: (closed_positions_df, ledger_df, pnl_by_df, open_positions_out)
    where open_positions_out are the still-open positions to carry into next month.
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()

    # File naming with suffix (D/H) — assumes you already use cr._suffix()
    suff = cr._suffix(decision_freq)
    enh_path = Path(cr.PATH_ENH) / f"{yymm}_enh{suff}.parquet"
    if not enh_path.exists():
        raise FileNotFoundError(f"Missing enhanced file {enh_path}. Run feature_creation.py first.")

    df = pd.read_parquet(enh_path)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), (open_positions or [])

    need = {"ts", "tenor_yrs", "rate", "z_spline", "z_pca", "z_comb"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{enh_path} missing columns: {missing}")

    # Decision buckets & decisions-per-day
    df["ts"] = pd.to_datetime(df["ts"], utc=False, errors="coerce")
    if decision_freq == "D":
        df["decision_ts"] = df["ts"].dt.floor("D")
        decisions_per_day = 1
    elif decision_freq == "H":
        df["decision_ts"] = df["ts"].dt.floor("H")
        # compute actual # decisions per trading day in this month
        per_day_counts = df.groupby(df["decision_ts"].dt.floor("D"))["decision_ts"].nunique()
        decisions_per_day = int(per_day_counts.mean()) if len(per_day_counts) else 24
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")

    max_hold_decisions = cr.MAX_HOLD_DAYS * decisions_per_day

    # Carry-in
    open_positions = (open_positions or []) if carry_in else []

    ledger_rows: list[dict] = []
    closed_rows: list[dict] = []

    # Iterate decisions
    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = (
            snap.sort_values("ts")
                .groupby("tenor_yrs", as_index=False)
                .tail(1)
                .reset_index(drop=True)
        )
        if snap_last.empty:
            continue

        # 1) Mark & evaluate exits on any open (carried + this month’s)
        still_open: list[PairPos] = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)
            # Exit rules
            if np.isfinite(zsp) and abs(zsp) <= cr.Z_EXIT:
                pos.closed, pos.close_ts, pos.exit_reason = True, dts, "reversion"
            elif np.isfinite(zsp) and np.isfinite(pos.entry_zspread) and abs(zsp - pos.entry_zspread) >= cr.Z_STOP:
                pos.closed, pos.close_ts, pos.exit_reason = True, dts, "stop"
            elif pos.age_decisions >= max_hold_decisions:
                pos.closed, pos.close_ts, pos.exit_reason = True, dts, "max_hold"

            # Ledger mark
            ledger_rows.append({
                "decision_ts": dts, "event": "mark",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "pnl": pos.pnl, "z_spread": zsp,
                "conv_proxy": pos.conv_pnl_proxy,
                "open_ts": pos.open_ts, "closed": pos.closed,
                "exit_reason": pos.exit_reason
            })

            if pos.closed:
                closed_rows.append({
                    "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": pos.exit_reason,
                    "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                    "w_i": pos.w_i, "w_j": pos.w_j, "entry_zspread": pos.entry_zspread,
                    "pnl": pos.pnl, "days_held_equiv": pos.age_decisions / decisions_per_day,
                    "conv_proxy": pos.conv_pnl_proxy
                })
            else:
                still_open.append(pos)

        open_positions = still_open

        # 2) New entries under caps
        selected = choose_pairs_under_caps(
            snap_last=snap_last,
            max_pairs=max(0, cr.MAX_CONCURRENT_PAIRS - len(open_positions)),
            per_bucket_cap=cr.PER_BUCKET_DV01_CAP,
            total_cap=cr.TOTAL_DV01_CAP,
            front_end_cap=cr.FRONT_END_DV01_CAP,
            extra_z_entry=0.0  # short-end extra handled inside chooser using SHORT_END_EXTRA_Z
        )

        for (cheap, rich, w_i, w_j) in selected:
            t_i = _to_float(cheap["tenor_yrs"]); t_j = _to_float(rich["tenor_yrs"])
            if (assign_bucket(t_i) == "short") or (assign_bucket(t_j) == "short"):
                zdisp = _to_float(cheap["z_comb"]) - _to_float(rich["z_comb"])
                if not np.isfinite(zdisp) or (zdisp < (cr.Z_ENTRY + cr.SHORT_END_EXTRA_Z)):
                    continue

            pos = PairPos(open_ts=dts, cheap_row=cheap, rich_row=rich,
                          w_i=w_i, w_j=w_j, decisions_per_day=decisions_per_day)
            open_positions.append(pos)
            ledger_rows.append({
                "decision_ts": dts, "event": "open",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread
            })

    # Outputs for this month (closed only)
    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)

    # PnL by bucket from marks
    if not ledger.empty:
        marks = ledger[ledger["event"] == "mark"].copy()
        idx = marks["decision_ts"].dt.floor("D" if decision_freq == "D" else "H")
        pnl_by = marks.groupby(idx)["pnl"].sum().rename("pnl").to_frame().reset_index()
        pnl_by = pnl_by.rename(columns={"decision_ts": "bucket"})
    else:
        pnl_by = pd.DataFrame(columns=["bucket", "pnl"])

    return pos_df, ledger, pnl_by, open_positions  # carry-out
            

# ------------------------
# Multi-month runner
# ------------------------
# --- REPLACE your existing run_all with this ---
def run_all(
    yymms: list[str],
    *,
    decision_freq: str | None = None,
    carry: bool = True,
    force_close_end: bool = False
):
    """
    Run multiple months, carrying open positions across months when carry=True.
    If force_close_end=True, anything still open after the last month is
    closed at the final bucket time (exit_reason='eoc' end-of-cycle).
    Returns concatenated (positions_closed, ledger, pnl_by).
    """
    decision_freq = (decision_freq or cr.DECISION_FREQ).upper()
    all_pos, all_ledger, all_by = [], [], []
    open_positions: list[PairPos] = []

    print(f"[INFO] months: {len(yymms)} -> {yymms}")
    for yymm in yymms:
        print(f"[RUN] month {yymm}")
        p, l, b, open_positions = run_month(
            yymm,
            decision_freq=decision_freq,
            open_positions=open_positions,
            carry_in=carry
        )
        if not p.empty: all_pos.append(p.assign(yymm=yymm))
        if not l.empty: all_ledger.append(l.assign(yymm=yymm))
        if not b.empty: all_by.append(b.assign(yymm=yymm))
        print(f"[DONE] {yymm} | closed={len(p)} | open_carry={len(open_positions)}")

    # Optionally force-close what's left after the last month for tidy reporting
    if force_close_end and open_positions:
        # final timestamp = last bucket of last processed month (from ledger/by if present)
        if all_ledger:
            final_ts = max(x["decision_ts"].max() for x in all_ledger if not x.empty)
        else:
            # Fallback: synthesize month end
            final_ts = pd.Timestamp.now()
        closed_rows = []
        for pos in open_positions:
            pos.closed, pos.close_ts, pos.exit_reason = True, final_ts, "eoc"
            closed_rows.append({
                "open_ts": pos.open_ts, "close_ts": pos.close_ts, "exit_reason": pos.exit_reason,
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j, "entry_zspread": pos.entry_zspread,
                "pnl": pos.pnl, "days_held_equiv": pos.age_decisions / max(1, pos.decisions_per_day),
                "conv_proxy": pos.conv_pnl_proxy
            })
        if closed_rows:
            all_pos.append(pd.DataFrame(closed_rows).assign(yymm=yymms[-1]))

    pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    led = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
    by  = pd.concat(all_by,  ignore_index=True) if all_by  else pd.DataFrame()
    return pos, led, by

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python portfolio_test.py 2304 [2305 2306 ...]")
        sys.exit(1)
    months = sys.argv[1:]
    pos, led, by = run_all(months, carry=True, force_close_end=False)  # keep positions alive
    out_dir = Path(cr.PATH_OUT); out_dir.mkdir(parents=True, exist_ok=True)
    if not pos.empty: pos.to_parquet(out_dir / f"positions_ledger{cr._suffix()}.parquet")
    if not led.empty: led.to_parquet(out_dir / f"marks_ledger{cr._suffix()}.parquet")
    if not by.empty:  by.to_parquet(out_dir / f"pnl_by_bucket{cr._suffix()}.parquet")
    print("[DONE]")
