from zoneinfo import ZoneInfo

PATH_DATA = x
PATH_ENH  = x
PATH_MOD  = x
PATH_OUT  = x

TRADING_HOURS = ("10:00", "22:10")
DECISION_FREQ = 'D'   # 'D' or 'H'

# === Add sub-2Y short end if you have these mids in your parquet ===
# If your files only have 1Y/2Y+, keep those. Add/remove as your columns allow.
TENOR_YEARS = {
    "USOSFR0.5Y BGN Curncy": 0.5,   # e.g., 6M OIS mid column name example
    "USOSFR0.75Y BGN Curncy": 0.75, # 9M (if present)
    "USOSFR1 BGN Curncy": 1.0,
    "USOSFR1.5 BGN Curncy": 1.5,    # 18M (if present)
    "USOSFR2 BGN Curncy": 2.0,
    "USOSFR3 BGN Curncy": 3.0,
    "USOSFR4 BGN Curncy": 4.0,
    "USOSFR5 BGN Curncy": 5.0,
    "USOSFR6 BGN Curncy": 6.0,
    "USOSFR7 BGN Curncy": 7.0,
    "USOSFR8 BGN Curncy": 8.0,
    "USOSFR9 BGN Curncy": 9.0,
    "USOSFR10 BGN Curncy": 10.0,
    "USOSFR12 BGN Curncy": 12.0,
    "USOSFR15 BGN Curncy": 15.0,
    "USOSFR20 BGN Curncy": 20.0,
    "USOSFR25 BGN Curncy": 25.0,
    "USOSFR30 BGN Curncy": 30.0,
    "USOSFR40 BGN Curncy": 40.0,
}
MID_COLS = [f"{bbg}_mid" for bbg in TENOR_YEARS.keys()]

# Buckets (now include sub-2Y)
BUCKETS = {
    "short":  (0.4, 1.9),  # 6M–<2Y (adjust to actual)
    "front":  (2.0, 3.0),
    "belly":  (3.1, 9.0),
    "long":   (10.0, 40.0),
}
MIN_SEP_YEARS = 1.0  # allow closer pairs if they’re in different buckets

# PCA & z-thresholds
PCA_LOOKBACK_DAYS = 126
PCA_COMPONENTS = 3
Z_ENTRY = 1.25
Z_EXIT  = 0.35
Z_STOP  = 2.0
MAX_HOLD_DAYS = 10

# ===== New: multi-pair + caps + gating =====
MAX_CONCURRENT_PAIRS = 3                 # allow several pairs at once
PER_BUCKET_DV01_CAP  = 1.0               # proxy units
TOTAL_DV01_CAP       = 3.0               # proxy units
FRONT_END_DV01_CAP   = 1.0               # cap agg DV01 in 'short' bucket

FLY_GATE_ENABLE      = True
FLY_DEFS = [          # choose flies we check for alignment (use what exists in data)
    (1.0, 3.0, 5.0),  # 1s3s5s
    (2.0, 5.0, 10.0)  # 2s5s10s (classic)
]
FLY_Z_MIN            = 0.3    # require fly z magnitude to not contradict leg’s z
FLY_ALIGN_MODE       = "loose"  # "loose" (only reject if opposite sign) or "strict" (must agree)

# De-risk short end around events (optional soft gate)
SHORT_END_EXTRA_Z    = 0.3     # add to Z_ENTRY if either leg is in 'short' bucket

CHI = ZoneInfo("America/Chicago")




import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

from cr_config import (
    PATH_DATA, PATH_ENH, PATH_MOD, MID_COLS, TENOR_YEARS, TRADING_HOURS,
    PCA_LOOKBACK_DAYS, PCA_COMPONENTS
)

os.makedirs(PATH_ENH, exist_ok=True)
os.makedirs(PATH_MOD, exist_ok=True)

DAILY_LEVELS_PATH = Path(PATH_MOD) / "daily_levels.parquet"

def _ensure_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    # enforce UTC-naive to match your current infra
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _load_month_parquet(yymm: str) -> pd.DataFrame:
    p = Path(PATH_DATA) / f"{yymm}_features.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_parquet(p, columns=[c for c in MID_COLS if c])
    df = _ensure_datetime_index(df)
    if TRADING_HOURS:
        df = df.between_time(*TRADING_HOURS)
    return df

def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    # columns: "<BBG>_mid"
    rows = []
    for col in df_wide.columns:
        if not col.endswith("_mid"):
            continue
        bbg = col[:-4]
        if bbg not in TENOR_YEARS:
            continue
        t = TENOR_YEARS[bbg]
        tmp = df_wide[[col]].rename(columns={col: "rate"})
        tmp["tenor_yrs"] = t
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0)
    out = out.reset_index().rename(columns={"index": "ts"})
    return out[["ts","tenor_yrs","rate"]]

def _fit_spline_snapshot(df_t: pd.DataFrame) -> CubicSpline:
    x = df_t["tenor_yrs"].values
    y = df_t["rate"].values
    idx = np.argsort(x)
    return CubicSpline(x[idx], y[idx], bc_type="natural")

def _apply_spline(df_snap):
    cs = _fit_spline_snapshot(df_snap)
    m = cs(df_snap["tenor_yrs"].values)
    eps = df_snap["rate"].values - m
    return m, eps

def _ewma_std(x: np.ndarray, lam=0.97):
    v = 0.0; out=[]
    for xi in x:
        v = lam*v + (1-lam)*xi*xi
        out.append(np.sqrt(max(v,1e-12)))
    return np.array(out)

def _update_daily_panel(daily_panel: pd.DataFrame, new_daily: pd.DataFrame) -> pd.DataFrame:
    # daily_panel: index=date, columns=str(tenor_yrs), values=rate (close)
    if daily_panel is None or daily_panel.empty:
        return new_daily
    out = pd.concat([daily_panel, new_daily], axis=0)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out

def _save_daily_panel(panel: pd.DataFrame):
    panel.to_parquet(DAILY_LEVELS_PATH)

def _load_daily_panel() -> pd.DataFrame:
    if DAILY_LEVELS_PATH.exists():
        return pd.read_parquet(DAILY_LEVELS_PATH)
    return pd.DataFrame()

def _train_pca(dfx_levels: pd.DataFrame, asof_date: pd.Timestamp, lookback_days=126, K=3):
    # dfx_levels: index=date, columns=str(tenor_yrs), % levels
    hist = dfx_levels[dfx_levels.index < asof_date].iloc[-lookback_days:]
    if hist.shape[0] < lookback_days//2:
        return None  # not enough history
    dX = hist.diff().dropna()
    C = np.cov(dX.values.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    V = eigvecs[:, order[:K]]
    mu = hist.mean().values
    cols = list(hist.columns)
    model = {"mu": mu, "V": V, "cols": cols}
    return model

def _write_pca_model(date_key: str, model: dict):
    path = Path(PATH_MOD) / f"pca_{date_key}.npz"
    np.savez_compressed(path, mu=model["mu"], V=model["V"], cols=np.array(model["cols"], dtype="U"))

def _read_pca_model(date_key: str):
    path = Path(PATH_MOD) / f"pca_{date_key}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    return {"mu": z["mu"], "V": z["V"], "cols": list(z["cols"])}

def _pca_fair_value(levels_row: pd.Series, model: dict) -> pd.Series:
    x = levels_row.reindex(model["cols"]).values
    mu = model["mu"]; V = model["V"]
    coeffs = (x - mu) @ V
    x_hat = mu + coeffs @ V.T
    return pd.Series(x_hat, index=model["cols"])

def build_month(yymm: str, combine=True, lam=0.97, w=0.5):
    """
    combine=True -> write enhanced monthly parquet with z_spline, z_pca, z_comb per tick.
    We freeze a PCA model per decision day (daily close), apply across that day's ticks.
    """
    df_wide = _load_month_parquet(yymm)
    if df_wide.empty:
        print(f"[{yymm}] empty month slice")
        return

    # Long panel per tick
    df_long = _wide_to_long(df_wide)
    if df_long.empty:
        print(f"[{yymm}] no OIS mid cols available")
        return

    # Build daily close levels (for PCA)
    # pivot per day close
    df_long["date"] = pd.to_datetime(df_long["ts"]).dt.floor("D")
    daily_close = (
        df_long.sort_values("ts")
               .groupby(["date","tenor_yrs"])
               .tail(1)
               .pivot(index="date", columns="tenor_yrs", values="rate")
               .rename_axis(None, axis=1)
    )
    daily_close.columns = [f"{c:.1f}" for c in daily_close.columns]

    # Load and update persistent daily panel
    panel = _load_daily_panel()
    panel = _update_daily_panel(panel, daily_close)
    _save_daily_panel(panel)

    # Precompute rolling EWMA std of spline and (later) pca residuals within the month
    # We'll build outputs day by day to bind the correct frozen PCA.
    out_rows = []

    # Prebuild list of decision dates in this month
    dates = sorted(daily_close.index.unique())

    for d in dates:
        date_key = pd.Timestamp(d).strftime("%Y-%m-%d")
        # try loading PCA; if missing, (re)train
        pca_model = _read_pca_model(date_key)
        if pca_model is None:
            pca_model = _train_pca(panel, asof_date=pd.Timestamp(d))
            if pca_model is not None:
                _write_pca_model(date_key, pca_model)

        # subset all ticks of this date
        day_slice = df_long[df_long["date"] == d].copy()
        if day_slice.empty:
            continue

        # ----- Spline residuals per tick (cross-sectional) -----
        # do it snapshot-by-snapshot
        def _one(ts, sub):
            m, e = _apply_spline(sub)
            sub = sub.copy()
            sub["model_spline"] = m
            sub["eps_spline"] = e
            return sub

        day_splits = [g for _, g in day_slice.groupby("ts")]
        day_splined = pd.concat([_one(None, g) for g in day_splits], axis=0)

        # EWMA std for spline residuals per tenor (within month to keep memory bounded)
        day_splined = day_splined.sort_values("ts")
        day_splined["eps_spline_ewstd"] = (
            day_splined.groupby("tenor_yrs")["eps_spline"]
            .apply(lambda s: pd.Series(_ewma_std(s.values, lam=lam), index=s.index))
            .values
        )
        day_splined["z_spline"] = day_splined["eps_spline"] / day_splined["eps_spline_ewstd"].replace(0.0, np.nan)

        # ----- PCA fair value (frozen per day) -----
        if pca_model is not None:
            # build level vector per tick snapshot, project, residual
            def _pca_apply_block(df_block):
                # df_block: one timestamp across tenors
                lv = df_block.pivot(index=None, columns="tenor_yrs", values="rate")
                lv.columns = [f"{c:.1f}" for c in lv.columns]
                lv_row = lv.iloc[0]
                xhat = _pca_fair_value(lv_row, pca_model)  # index=model['cols']
                # map back to rows
                df_block = df_block.copy()
                df_block["model_pca"] = df_block["tenor_yrs"].apply(lambda t: xhat.get(f"{t:.1f}", np.nan))
                df_block["eps_pca"] = df_block["rate"] - df_block["model_pca"]
                return df_block

            day_pca = pd.concat([_pca_apply_block(g) for _, g in day_splined.groupby("ts")], axis=0)
            # EWMA std for pca residuals
            day_pca = day_pca.sort_values("ts")
            day_pca["eps_pca_ewstd"] = (
                day_pca.groupby("tenor_yrs")["eps_pca"]
                .apply(lambda s: pd.Series(_ewma_std(s.values, lam=lam), index=s.index))
                .values
            )
            day_pca["z_pca"] = day_pca["eps_pca"] / day_pca["eps_pca_ewstd"].replace(0.0, np.nan)
        else:
            day_pca = day_splined.copy()
            day_pca["model_pca"] = np.nan
            day_pca["eps_pca"] = np.nan
            day_pca["eps_pca_ewstd"] = np.nan
            day_pca["z_pca"] = np.nan

        # combine
        day_pca["z_comb"] = 0.5 * day_pca["z_spline"].astype(float) + 0.5 * day_pca["z_pca"].astype(float)
        out_rows.append(day_pca)

    enh = pd.concat(out_rows, axis=0) if out_rows else pd.DataFrame()
    if enh.empty:
        print(f"[{yymm}] no enhanced rows produced (insufficient PCA history?)")
        return

    enh = enh[["ts","tenor_yrs","rate",
               "model_spline","eps_spline","z_spline",
               "model_pca","eps_pca","z_pca","z_comb"]].sort_values(["ts","tenor_yrs"])

    out_path = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    enh.to_parquet(out_path)
    print(f"[{yymm}] wrote enhanced -> {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_features_ois.py 2501 [2502 2503 ...]")
        sys.exit(1)
    for yymm in sys.argv[1:]:
        build_month(yymm)
        
        
import os, sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from cr_config import (
    PATH_ENH, PATH_OUT, DECISION_FREQ, MIN_SEP_YEARS,
    Z_ENTRY, Z_EXIT, Z_STOP, MAX_HOLD_DAYS,
    BUCKETS, TENOR_YEARS,
    MAX_CONCURRENT_PAIRS, PER_BUCKET_DV01_CAP, TOTAL_DV01_CAP, FRONT_END_DV01_CAP,
    FLY_GATE_ENABLE, FLY_DEFS, FLY_Z_MIN, FLY_ALIGN_MODE,
    SHORT_END_EXTRA_Z
)

os.makedirs(PATH_OUT, exist_ok=True)

def pv01_proxy(tenor_yrs, rate_pct):
    return tenor_yrs / max(1e-6, 1.0 + 0.01*rate_pct)

def assign_bucket(tenor):
    for name, (lo, hi) in BUCKETS.items():
        if (tenor >= lo) and (tenor <= hi):
            return name
    return "other"

# ---- fly z computation (from z_comb cross-section) ----
def compute_fly_z(snap_last, a, b, c):
    # 0.5*(a+c) - b  on z_comb (shape, not on levels)
    vals = {}
    for T in [a,b,c]:
        row = snap_last[snap_last["tenor_yrs"]==T]
        if row.empty: return None
        vals[T] = float(row["z_comb"])
    fly = 0.5*(vals[a] + vals[c]) - vals[b]
    # crude standardization: use cross-sec std at that ts
    xs = snap_last["z_comb"].astype(float)
    sd = xs.std(ddof=1) if xs.size>1 else 1.0
    return fly / (sd if sd>0 else 1.0)

def fly_alignment_ok(leg_tenor, leg_sign_z, snap_last):
    """Return True if flies do not contradict the leg direction."""
    if not FLY_GATE_ENABLE: return True
    for (a,b,c) in FLY_DEFS:
        zz = compute_fly_z(snap_last, a,b,c)
        if zz is None: 
            continue
        if abs(zz) < FLY_Z_MIN:
            continue
        # If strict: require same sign; if loose: just reject opposite sign
        if FLY_ALIGN_MODE == "strict" and np.sign(zz) != np.sign(leg_sign_z):
            return False
        if FLY_ALIGN_MODE == "loose" and (np.sign(zz) * np.sign(leg_sign_z) < 0):
            return False
    return True

# ---- position object (unchanged-ish) ----
class PairPos:
    def __init__(self, open_ts, cheap, rich, w_i, w_j):
        self.open_ts = open_ts
        self.tenor_i = cheap["tenor_yrs"]; self.rate_i = cheap["rate"]
        self.tenor_j = rich["tenor_yrs"];  self.rate_j = rich["rate"]
        self.w_i = w_i; self.w_j = w_j
        self.entry_zspread = cheap["z_comb"] - rich["z_comb"]
        self.closed = False; self.close_ts = None; self.exit_reason = None
        self.pnl = 0.0; self.days_held = 0
        self.bucket_i = assign_bucket(self.tenor_i); self.bucket_j = assign_bucket(self.tenor_j)
        # attribution proxies
        self.last_zspread = self.entry_zspread
        self.conv_pnl_proxy = 0.0  # cum (Δz_spread) scaled

    def mark(self, snap_last):
        ri = float(snap_last[snap_last["tenor_yrs"]==self.tenor_i]["rate"])
        rj = float(snap_last[snap_last["tenor_yrs"]==self.tenor_j]["rate"])
        d_i = (self.rate_i - ri) * self.w_i * 100.0
        d_j = (self.rate_j - rj) * self.w_j * 100.0
        self.pnl = d_i + d_j

        # convergence proxy: move in z_spread (scale by 10 for readability)
        zi = float(snap_last[snap_last["tenor_yrs"]==self.tenor_i]["z_comb"])
        zj = float(snap_last[snap_last["tenor_yrs"]==self.tenor_j]["z_comb"])
        zsp = zi - zj
        self.conv_pnl_proxy += (self.last_zspread - zsp) * 10.0
        self.last_zspread = zsp
        return zsp

# ---- greedy selector for multiple pairs under caps ----
def choose_pairs_under_caps(snap_last, max_pairs,
                            per_bucket_cap, total_cap, front_end_cap, extra_z_entry):
    """
    Returns list of (cheap_row, rich_row, w_i, w_j)
    Greedy: rank by dispersion, enforce tenor uniqueness & DV01 caps by bucket and total.
    """
    sig = snap_last[["tenor_yrs","rate","z_comb"]].dropna().copy()
    if sig.empty: return []

    # Sort by z for extremes
    sig = sig.sort_values("z_comb")
    candidates = []
    used_tenors = set()

    # Build extreme pairs pool
    for k_low in range(min(5, len(sig))):
        rich = sig.iloc[k_low]
        for k_hi in range(1, min(8, len(sig))+1):
            cheap = sig.iloc[-k_hi]
            if cheap["tenor_yrs"] in used_tenors or rich["tenor_yrs"] in used_tenors:
                continue
            # separation and bucket diversification preferred
            if abs(cheap["tenor_yrs"] - rich["tenor_yrs"]) < MIN_SEP_YEARS:
                continue
            # fly alignment gates
            if not fly_alignment_ok(cheap["tenor_yrs"], +1, snap_last):  # cheap leg should revert down
                continue
            if not fly_alignment_ok(rich["tenor_yrs"], -1, snap_last):   # rich leg should revert up
                continue
            zdisp = float(cheap["z_comb"] - rich["z_comb"])
            if zdisp < (Z_ENTRY + extra_z_entry):
                continue
            candidates.append((zdisp, cheap, rich))

    # Greedy pack by zdisp descending under caps
    bucket_dv01 = {b: 0.0 for b in BUCKETS.keys()}
    total_dv01 = 0.0
    selected = []

    for zdisp, cheap, rich in sorted(candidates, key=lambda x: x[0], reverse=True):
        if len(selected) >= max_pairs:
            break
        if cheap["tenor_yrs"] in used_tenors or rich["tenor_yrs"] in used_tenors:
            continue

        # PV01 proxy weights (neutrality within pair)
        pv_i = pv01_proxy(cheap["tenor_yrs"], cheap["rate"])
        pv_j = pv01_proxy(rich["tenor_yrs"],  rich["rate"])
        w_i =  1.0
        w_j = - w_i * pv_i / pv_j

        # DV01 usage by buckets (proxy, abs weights)
        b_i = assign_bucket(cheap["tenor_yrs"]); b_j = assign_bucket(rich["tenor_yrs"])
        dv_i = abs(w_i) * pv_i; dv_j = abs(w_j) * pv_j
        # Check caps
        def _would_violate(bmap, b, add, cap):
            return (bmap[b] + add) > cap

        # per-bucket
        if b_i in bucket_dv01 and _would_violate(bucket_dv01, b_i, dv_i, PER_BUCKET_DV01_CAP):
            continue
        if b_j in bucket_dv01 and _would_violate(bucket_dv01, b_j, dv_j, PER_BUCKET_DV01_CAP):
            continue
        # front-end aggregate cap
        short_add = (dv_i if b_i=="short" else 0.0) + (dv_j if b_j=="short" else 0.0)
        short_tot = sum(v for k,v in bucket_dv01.items() if k=="short")
        if short_tot + short_add > front_end_cap:
            continue
        # total cap
        if total_dv01 + dv_i + dv_j > total_cap:
            continue

        # accept
        used_tenors.add(cheap["tenor_yrs"]); used_tenors.add(rich["tenor_yrs"])
        bucket_dv01[b_i] += dv_i; bucket_dv01[b_j] += dv_j
        total_dv01 += dv_i + dv_j
        selected.append((cheap, rich, w_i, w_j))

    return selected

def run_month(yymm: str, decision_freq='D'):
    p = Path(PATH_ENH) / f"{yymm}_enh.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run build_features_ois.py first.")
    df = pd.read_parquet(p)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts"])
    if decision_freq == 'D':
        df["decision_ts"] = df["ts"].dt.floor("D")
    elif decision_freq == 'H':
        df["decision_ts"] = df["ts"].dt.floor("H")
    else:
        raise ValueError("decision_freq must be 'D' or 'H'.")

    # z-heatmap (per month)
    _heat = (df.pivot_table(index="ts", columns="tenor_yrs", values="z_comb", aggfunc="last")
               .sort_index())
    heat_path = Path(PATH_OUT) / f"z_heatmap_{yymm}.png"
    if not _heat.empty:
        plt.figure()
        plt.imshow(_heat.T, aspect="auto", origin="lower")
        plt.title(f"z_comb heatmap {yymm}")
        plt.xlabel("time idx"); plt.ylabel("tenor (yrs)")
        plt.colorbar()
        plt.savefig(heat_path, dpi=120, bbox_inches="tight")
        plt.close()

    open_positions: list[PairPos] = []
    ledger_rows = []
    closed_rows = []

    for dts, snap in df.groupby("decision_ts", sort=True):
        snap_last = snap.sort_values("ts").groupby("tenor_yrs").tail(1)
        if snap_last.empty:
            continue

        # 1) Mark all open positions, decide exits
        still_open = []
        for pos in open_positions:
            zsp = pos.mark(snap_last)
            # exit rules
            if abs(zsp) <= Z_EXIT:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "reversion"
            elif abs(zsp - pos.entry_zspread) >= Z_STOP:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "stop"
            elif pos.days_held >= MAX_HOLD_DAYS:
                pos.closed = True; pos.close_ts = dts; pos.exit_reason = "max_hold"
            else:
                pos.days_held += (1 if DECISION_FREQ=='D' else 0)  # light aging
            # ledger mark
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
                    "pnl": pos.pnl, "days_held": pos.days_held,
                    "conv_proxy": pos.conv_pnl_proxy
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2) Entries (greedy under caps), extra z threshold if front-end involved
        # Compute extra entry cost if any leg lands in short bucket
        extra_z = 0.0
        # try preview: if best dispersion uses any 'short' we’ll re-check in selector with added threshold.
        selected = choose_pairs_under_caps(
            snap_last=snap_last,
            max_pairs=max(0, MAX_CONCURRENT_PAIRS - len(open_positions)),
            per_bucket_cap=PER_BUCKET_DV01_CAP,
            total_cap=TOTAL_DV01_CAP,
            front_end_cap=FRONT_END_DV01_CAP,
            extra_z_entry=extra_z
        )

        for (cheap, rich, w_i, w_j) in selected:
            # “short” penalty: require larger dispersion if short bucket used
            if assign_bucket(cheap["tenor_yrs"])=="short" or assign_bucket(rich["tenor_yrs"])=="short":
                if (float(cheap["z_comb"] - rich["z_comb"]) < (Z_ENTRY + SHORT_END_EXTRA_Z)):
                    continue
            pos = PairPos(open_ts=dts, cheap=cheap, rich=rich, w_i=w_i, w_j=w_j)
            open_positions.append(pos)
            ledger_rows.append({
                "decision_ts": dts, "event": "open",
                "tenor_i": pos.tenor_i, "tenor_j": pos.tenor_j,
                "w_i": pos.w_i, "w_j": pos.w_j,
                "entry_zspread": pos.entry_zspread
            })

    # Outputs
    pos_df = pd.DataFrame(closed_rows)
    ledger = pd.DataFrame(ledger_rows)
    # PnL by day
    pnl_by_day = (ledger[ledger["event"]=="mark"]
                  .groupby(ledger["decision_ts"].dt.floor("D"))["pnl"].sum()
                  .rename("daily_pnl").to_frame().reset_index()
                  .rename(columns={"decision_ts":"date"}))
    # Save month-level plots (cumulative PnL)
    if not pnl_by_day.empty:
        pnl_by_day = pnl_by_day.sort_values("date")
        pnl_by_day["cum"] = pnl_by_day["daily_pnl"].cumsum()
        plt.figure()
        plt.plot(pnl_by_day["date"], pnl_by_day["cum"])
        plt.title(f"Cumulative PnL proxy {yymm}")
        plt.xlabel("Date"); plt.ylabel("Cum PnL")
        out_png = Path(PATH_OUT) / f"pnl_curve_{yymm}.png"
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close()

    return pos_df, ledger, pnl_by_day

def run_all(yymms):
    all_pos = []; all_ledger = []; all_byday = []
    for yymm in yymms:
        print(f"[RUN] month {yymm}")
        p, l, b = run_month(yymm, decision_freq=DECISION_FREQ)
        if not p.empty: all_pos.append(p)
        if not l.empty: all_ledger.append(l)
        if not b.empty: all_byday.append(b.assign(yymm=yymm))
    pos = pd.concat(all_pos, ignore_index=True) if all_pos else pd.DataFrame()
    led = pd.concat(all_ledger, ignore_index=True) if all_ledger else pd.DataFrame()
    byday = pd.concat(all_byday, ignore_index=True) if all_byday else pd.DataFrame()
    return pos, led, byday

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cheap_rich_backtest.py 2501 [2502 2503 ...]")
        sys.exit(1)
    pos, led, byday = run_all(sys.argv[1:])
    if not pos.empty: pos.to_parquet(Path(PATH_OUT) / "positions_ledger.parquet")
    if not led.empty: led.to_parquet(Path(PATH_OUT) / "marks_ledger.parquet")
    if not byday.empty: byday.to_parquet(Path(PATH_OUT) / "pnl_by_day.parquet")
    print("[DONE]")