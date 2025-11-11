# feature_creation.py
import os, sys, time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Use module-level config access everywhere
import cr_config as cr


# -----------------------
# Small utilities
# -----------------------
def _now():
    return time.strftime("%H:%M:%S")

def _to_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC-naive DatetimeIndex column 'ts' exists and is sorted."""
    if "ts" not in df.columns:
        # older files may have index named "sec" or "ts"
        if df.index.name in ("ts", "sec"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "ts"})
        else:
            raise KeyError("No 'ts' column or index found.")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    # convert to tz-naive UTC (consistent downstream)
    df["ts"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.sort_values("ts")


# -----------------------
# Calendar & hours
# -----------------------
def _get_ql_calendar():
    if not getattr(cr, "USE_QL_CALENDAR", False):
        return None
    try:
        import QuantLib as ql
    except Exception:
        print("[CAL] QuantLib not available; falling back to weekday filter.")
        return None

    try:
        # Try direct enum (most builds): ql.UnitedStates.FederalReserve, etc.
        market = str(getattr(cr, "QL_US_MARKET", "FederalReserve"))
        direct = getattr(ql.UnitedStates, market, None)
        if direct is not None:
            return ql.UnitedStates(direct)

        # Fallback: nested Market enum
        market_enum_cls = getattr(ql.UnitedStates, "Market", None)
        if market_enum_cls is not None:
            mkt = getattr(market_enum_cls, market, None)
            if mkt is not None:
                return ql.UnitedStates(mkt)

        # Last resort defaults
        if hasattr(ql.UnitedStates, "FederalReserve"):
            return ql.UnitedStates(ql.UnitedStates.FederalReserve)
        if hasattr(ql.UnitedStates, "NYSE"):
            return ql.UnitedStates(ql.UnitedStates.NYSE)
        return ql.UnitedStates()
    except Exception as e:
        print(f"[CAL] Failed to init QuantLib calendar: {e}; using weekday filter.")
        return None


def _apply_calendar_and_hours(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Filter to business days (QuantLib if available) + local trading hours."""
    if df_wide.empty:
        return df_wide

    # 1) Business-day filter (QuantLib if available, else Mon–Fri)
    cal = _get_ql_calendar()
    ts = pd.to_datetime(df_wide["ts"])
    if cal is not None:
        try:
            import QuantLib as ql
            ymd = np.array([(t.year, t.month, t.day) for t in ts], dtype=int)
            days = pd.to_datetime(
                {"year": ymd[:, 0], "month": ymd[:, 1], "day": ymd[:, 2]}
            ).drop_duplicates().sort_values()
            bd_mask = []
            for d in days:
                qd = ql.Date(int(d.day), int(d.month), int(d.year))
                bd_mask.append(cal.isBusinessDay(qd))
            day_ok = pd.Series(bd_mask, index=days)
            ok = day_ok.reindex(ts.dt.floor("D"), fill_value=False).values
            df_wide = df_wide.loc[ok]
            print(f"[CAL] QuantLib calendar active ({getattr(cr,'QL_US_MARKET','?')}); days={day_ok.sum()}")
        except Exception as e:
            print(f"[CAL] QuantLib daily check failed: {e}; using weekday filter.")
            df_wide = df_wide[ts.dt.weekday < 5]
            print(f"[CAL] Simple Mon–Fri filter active; "
                  f"days={df_wide['ts'].dt.floor('D').nunique()}")
    else:
        df_wide = df_wide[ts.dt.weekday < 5]
        print(f"[CAL] Simple Mon–Fri filter active; "
              f"days={df_wide['ts'].dt.floor('D').nunique()}")

    # 2) Session hours in local tz (e.g., America/New_York)
    tz_local = getattr(cr, "CAL_TZ", "America/New_York")
    start_str, end_str = getattr(cr, "TRADING_HOURS", ("07:00", "17:30"))
    pre = len(df_wide)

    # Make a tz-aware local-time index specifically for slicing,
    # then convert back to UTC-naive.
    df_wide = df_wide.copy()
    df_wide["ts_local"] = df_wide["ts"].dt.tz_localize("UTC").dt.tz_convert(tz_local)

    tmp = df_wide.set_index("ts_local").sort_index()
    tmp = tmp.between_time(start_str, end_str)

    # Recreate UTC-naive 'ts' from the local index; then drop the index safely.
    tmp = tmp.copy()
    tmp["ts"] = tmp.index.tz_convert("UTC").tz_localize(None)
    tmp = tmp.reset_index(drop=True)  # removes 'ts_local' index

    kept = len(tmp)
    if pre > 0:
        print(f"[CAL] kept {kept:,}/{pre:,} rows ({(kept/pre*100):.2f}%) after calendar+hours")

    # Ensure only expected columns remain
    if "ts_local" in tmp.columns:
        tmp = tmp.drop(columns=["ts_local"], errors="ignore")

    return tmp.reset_index(drop=True)


# -----------------------
# Cleaning & reshaping
# -----------------------
def _zeros_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric, turn literal 0.0 to NaN (but not timestamps)."""
    num = df.drop(columns=["ts"])
    num = num.apply(pd.to_numeric, errors="coerce")
    num = num.mask(num == 0)
    out = df[["ts"]].join(num)
    return out


def _melt_long(df_wide: pd.DataFrame, tenormap: Dict[str, float]) -> pd.DataFrame:
    """Wide → long using tolerant column matching to TENOR_YEARS."""
    def norm(s: str) -> str:
        s = str(s).strip()
        if s.endswith("_mid"):
            s = s[:-4]
        return " ".join(s.split())

    tenormap_norm = {norm(k): v for k, v in tenormap.items()}

    cand = []
    for c in df_wide.columns:
        if c == "ts":
            continue
        nc = norm(c)
        if nc in tenormap_norm:
            cand.append((c, tenormap_norm[nc]))

    if not cand:
        sample_cols = [c for c in df_wide.columns if c != "ts"][:20]
        raise ValueError(
            "No overlapping tickers between input file and TENOR_YEARS.\n"
            f"Example file columns: {sample_cols}\n"
            f"Example tenor keys: {list(tenormap.keys())[:20]}"
        )

    use_cols = ["ts"] + [c for c, _ in cand]
    df_sel = df_wide[use_cols].copy()

    col_to_tenor = {c: t for c, t in cand}
    long = df_sel.melt(id_vars="ts", var_name="instrument", value_name="rate")
    long["tenor_yrs"] = long["instrument"].map(col_to_tenor).astype(float)

    long["rate"] = pd.to_numeric(long["rate"], errors="coerce")
    long = long.dropna(subset=["ts", "tenor_yrs", "rate"])
    return long[["ts", "tenor_yrs", "rate"]]


def _decision_key(ts: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return ts.dt.floor("D")
    elif freq == "H":
        return ts.dt.floor("H")
    else:
        raise ValueError("DECISION_FREQ must be 'D' or 'H'.")


# -----------------------
# Spline “shape” z (robust)
# -----------------------
def _spline_fit_safe(snap_long: pd.DataFrame) -> pd.Series:
    """
    Per-bucket cross-section: fit low-order polynomial (cubic) to rate(tenor),
    z-score residuals using robust scale.
    """
    s = snap_long[["tenor_yrs", "rate"]].dropna()
    out = pd.Series(index=snap_long.index, dtype=float)

    if s.shape[0] < 5:
        return out  # leave NaN

    x = s["tenor_yrs"].values.astype(float)
    y = s["rate"].values.astype(float)

    # degree 3 if possible, else lower
    deg = 3 if len(x) >= 4 else min(2, len(x)-1)
    try:
        coef = np.polyfit(x, y, deg=deg)
        fit = np.polyval(coef, x)
        resid = y - fit

        # robust scale: MAD or fallback to std
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        scale = (1.4826 * mad) if mad > 0 else resid.std(ddof=1)
        if not np.isfinite(scale) or scale == 0:
            return out

        z = (resid - resid.mean()) / scale
        m = {ten: val for ten, val in zip(x, z)}
        out.loc[s.index] = s["tenor_yrs"].map(m).values
        return out
    except Exception:
        return out


# -----------------------
# PCA helpers (fit on lookback restricted to current columns)
# -----------------------
def _pca_fit_panel(panel_long: pd.DataFrame, cols_ordered: List[float], n_comps: int):
    """Return dict{'cols','mean','components','evr'} or None."""
    if panel_long.empty:
        return None
    W = (panel_long.pivot(index="ts", columns="tenor_yrs", values="rate")
                  .sort_index())
    W = W.reindex(columns=cols_ordered).ffill().dropna(how="any")
    if W.shape[0] < (n_comps + 5) or W.shape[1] < n_comps:
        return None

    X = W.values.astype(float)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n_comps, :]           # (k, n_features)
    evr = (S**2) / (S**2).sum()
    return {"cols": list(W.columns), "mean": mu.ravel(), "components": comps, "evr": evr[:n_comps]}


def _pca_apply_block(df_block: pd.DataFrame, pca_model: dict) -> pd.Series:
    """
    Robust application of a previously-fitted PCA model to one decision bucket.
    - Chooses the LAST tick per tenor inside the bucket
    - Aligns strictly to model['cols'] (both set & order)
    - Returns a Series of standardized reconstructed scores mapped to df_block rows
    """
    out = pd.Series(index=df_block.index, dtype=float)
    if not pca_model or df_block.empty:
        return out

    cols  = list(pca_model["cols"])
    mu    = np.asarray(pca_model["mean"], dtype=float)
    comps = np.asarray(pca_model["components"], dtype=float)

    # Last observation per tenor in this bucket
    last = (df_block.sort_values("ts")
                     .groupby("tenor_yrs", as_index=False)
                     .tail(1)
                     .set_index("tenor_yrs")["rate"])
    # Align to model cols (exact set + order)
    if any(c not in last.index for c in cols):
        # Missing a required tenor → skip PCA for this bucket
        return out

    x = last.reindex(cols).values.astype(float)   # shape (n_features,)
    xc = x - mu                                   # center
    score = comps @ xc                            # (k,)
    recon = comps.T @ score                       # (n_features,)

    # Standardize reconstruction across features for a z-like shape
    sd = recon.std()
    if not np.isfinite(sd) or sd == 0:
        return out
    z_std = (recon - recon.mean()) / sd

    # Map back to tenor_yrs, then to the original df_block rows
    z_map = dict(zip(cols, z_std))
    return df_block["tenor_yrs"].map(z_map)


# -----------------------
# Per-bucket processor
# -----------------------
def _process_bucket(dts, df_bucket, df_all, lookback_days, pca_enable, pca_n_comps, yymm):
    out = df_bucket[["ts","tenor_yrs","rate"]].copy().reset_index(drop=True)

    # 1) spline z
    out["z_spline"] = _spline_fit_safe(out)

    # 2) PCA z (fit on lookback restricted to current columns)
    out["z_pca"] = np.nan
    if pca_enable:
        cols_now = sorted(df_bucket["tenor_yrs"].unique().tolist())
        if len(cols_now) >= pca_n_comps:
            t_end   = df_bucket["ts"].min()
            t_start = t_end - pd.Timedelta(days=float(lookback_days))
            panel = df_all[(df_all["ts"]>=t_start) & (df_all["ts"]<t_end) &
                           (df_all["tenor_yrs"].isin(cols_now))]
            model = _pca_fit_panel(panel, cols_now, pca_n_comps)
            if model:
                evr = list(np.round(model["evr"], 3))
                print(f"[PCA] {yymm} {str(dts)[:16]} obs={len(panel):,} cols={len(cols_now)} EVR={evr}")
                out["z_pca"] = _pca_apply_block(out, model)
            else:
                print(f"[WARN] PCA skipped {yymm} {dts}: insufficient history/features "
                      f"(cols={len(cols_now)}, lookback_days={lookback_days})")

    # 3) Combine
    if out["z_pca"].notna().any():
        out["z_comb"] = 0.5*out["z_spline"] + 0.5*out["z_pca"]
    else:
        out["z_comb"] = out["z_spline"]

    return out


# -----------------------
# Filename helpers (non-destructive outputs)
# -----------------------
def _enhanced_out_path(yymm: str) -> Path:
    """
    Use cr.enh_fname if present; else fall back to {yymm}_enh{ENH_SUFFIX}.parquet;
    else final fallback {yymm}_enh.parquet.
    """
    if hasattr(cr, "enh_fname") and callable(cr.enh_fname):
        name = cr.enh_fname(yymm)
    else:
        suffix = getattr(cr, "ENH_SUFFIX", "")
        name = f"{yymm}_enh{suffix}.parquet" if suffix else f"{yymm}_enh.parquet"
    return Path(getattr(cr, "PATH_ENH", ".")) / name


# -----------------------
# Month builder (public)
# -----------------------
def build_month(yymm: str) -> None:
    path_data = Path(getattr(cr, "PATH_DATA", "."))
    path_enh  = Path(getattr(cr, "PATH_ENH", "."))
    path_enh.mkdir(parents=True, exist_ok=True)

    in_path  = path_data / f"{yymm}_features.parquet"
    out_path = _enhanced_out_path(yymm)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing raw month file: {in_path}")

    # Load & normalize
    df_wide = pd.read_parquet(in_path)
    df_wide = _to_ts_index(df_wide)
    df_wide = df_wide[~df_wide["ts"].duplicated(keep="last")]
    # Calendar + hours
    df_wide = _apply_calendar_and_hours(df_wide)
    # Clean
    df_wide = _zeros_to_nan(df_wide)

    # Long
    tenormap = getattr(cr, "TENOR_YEARS", {})
    df_long = _melt_long(df_wide, tenormap)
    decision_freq = str(getattr(cr, "DECISION_FREQ", "D")).upper()
    df_long["decision_ts"] = _decision_key(df_long["ts"], decision_freq)

    buckets = (df_long["decision_ts"].dropna().unique().tolist())
    buckets.sort()

    # jobs
    N_JOBS = int(getattr(cr, "N_JOBS", 1))
    if isinstance(N_JOBS, int):
        if N_JOBS == 0:
            import multiprocessing as mp
            jobs = max(1, min((mp.cpu_count() // 2), 8))
        else:
            jobs = int(N_JOBS)
    else:
        jobs = 1

    pca_enable = bool(getattr(cr, "PCA_ENABLE", True))
    lookback_days = float(getattr(cr, "PCA_LOOKBACK_DAYS", 126))
    pca_components = int(getattr(cr, "PCA_COMPONENTS", 3))

    # Log PCA cap info for hourly if present
    cap_hours = getattr(cr, "PCA_LOOKBACK_CAP_HOURS", None)
    cap_str = f" | cap_hours={cap_hours}" if cap_hours is not None else ""

    print(f"[{_now()}] [MONTH] {yymm} buckets={len(buckets)} freq={decision_freq} | jobs={jobs} "
          f"| PCA={'on' if pca_enable else 'off'} | lookback_days={lookback_days} | comps={pca_components}{cap_str}")

    if not buckets:
        pd.DataFrame(columns=['ts','tenor_yrs','rate','z_spline','z_pca','z_comb']).to_parquet(out_path, index=False)
        print(f"[SAVE] {out_path}")
        return

    def _one(dts):
        snap = df_long[df_long["decision_ts"] == dts]
        t0 = time.time()
        out = _process_bucket(
            dts=dts,
            df_bucket=snap,
            df_all=df_long[['ts','tenor_yrs','rate','decision_ts']],
            lookback_days=lookback_days,
            pca_enable=pca_enable,
            pca_n_comps=pca_components,
            yymm=yymm
        )
        dt = time.time() - t0
        print(f"[BUCKET] {yymm} {dts} rows:{len(snap):,} tenors:{snap['tenor_yrs'].nunique()} "
              f"PCA:{'yes' if out['z_pca'].notna().any() else 'no '} t={dt:.2f}s")
        return out

    parts = Parallel(n_jobs=jobs, backend="loky")(delayed(_one)(d) for d in buckets)
    out = pd.concat(parts, ignore_index=True).sort_values(['ts','tenor_yrs']).reset_index(drop=True)

    zr = pd.to_numeric(out['z_comb'], errors='coerce')
    z_valid_pct = float(np.isfinite(zr).mean() * 100.0) if not out.empty else 0.0
    med_ten = out.groupby('ts')['tenor_yrs'].nunique().median() if not out.empty else 0
    print(f"[DONE] {yymm} rows:{len(out):,} tenors_med:{med_ten:.1f} z_valid%:{z_valid_pct:.2f}")

    out.to_parquet(out_path, index=False)
    print(f"[SAVE] {out_path}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python feature_creation.py 2304 [2305 2306 ...]")
        sys.exit(1)
    for m in sys.argv[1:]:
        build_month(m)
