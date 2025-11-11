# sweeper.py
from __future__ import annotations
from pathlib import Path
import itertools, random, importlib, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np, pandas as pd, matplotlib.pyplot as plt

# All config goes through the module namespace
import cr_config as cr
import portfolio_test as pt


# ------------------ USER INPUTS ------------------
YYMMS          = ["2304","2305"]       # pick several months for robustness
RANDOMIZE      = True                  # shuffle variant order
MAX_VARIANTS   = 120                   # cap to keep runtime reasonable
SAVE_PREFIX    = "sweep_results"       # results saved under PATH_OUT/<prefix>[OUT_SUFFIX].{parquet,csv}
MAX_WORKERS    = max(1, min(8, mp.cpu_count() // 2))  # parallel variants
# -------------------------------------------------


# ---- Parameter grids (edit freely) ----
GRID_FLY_MODE           = ["off", "loose", "tolerant", "strict"]
GRID_FLY_ALLOW_BIG      = [False, True]
GRID_FLY_BIG_MARGIN     = [0.4, 0.6]                # used only when allow_big=True
GRID_FLY_Z_MIN          = [0.3, 0.4, 0.6]
GRID_FLY_WINDOW         = [1.0, 1.5, 2.5]
GRID_FLY_NEIGH_ONLY     = [True]                    # add False if you want global checks
GRID_FLY_SKIP_SHORT     = [1.0, 2.0]                # yrs; 2.0 = skip <2y
GRID_FLY_REQUIRE_COUNT  = [1, 2]                    # only matters for tolerant

GRID_Z_ENTRY            = [0.75, 0.9, 1.05]         # higher = fewer, higher-conviction
GRID_Z_EXIT             = [0.35, 0.40, 0.45]        # lower = take profit sooner
GRID_MAX_HOLD_DAYS      = [7, 10]
GRID_SHORT_END_EXTRA_Z  = [0.20, 0.30, 0.40]
# --------------------------------------


# ---------- helpers: config round-trip & metrics ----------
ORIG = {k: getattr(cr, k) for k in dir(cr) if k.isupper()}

def _reload_pt_only():
    importlib.reload(pt)

def _apply_variant_inproc(v: dict):
    """Single-process mode: mutate cr in-memory, then reload pt so it picks up new values."""
    # restore originals
    for k, val in ORIG.items():
        try: setattr(cr, k, val)
        except: pass
    # apply overrides
    for k, val in v.items():
        if hasattr(cr, k):
            setattr(cr, k, val)
    _reload_pt_only()

def _metrics(pos, led, by) -> dict:
    m = {}
    n = int(len(pos)) if pos is not None else 0
    m["trades"] = n
    m["avg_pnl_per_trade"] = float(pos["pnl"].mean()) if n else np.nan
    if n:
        c = pos["exit_reason"].value_counts()
        for r in ["reversion","max_hold","stop"]:
            m[f"exit_{r}"] = int(c.get(r, 0))
    else:
        m["exit_reversion"] = m["exit_max_hold"] = m["exit_stop"] = 0
    if by is not None and len(by):
        s = by.sort_values("bucket")["pnl"].astype(float).cumsum()
        m["cum_pnl"] = float(s.iloc[-1])
        dd = (s.cummax() - s)
        m["max_drawdown"] = float(dd.max())
        pnl = by["pnl"].astype(float)
        m["pnl_mean"] = float(pnl.mean())
        m["pnl_std"]  = float(pnl.std(ddof=1)) if len(pnl) > 1 else np.nan
        m["mean_over_std"] = float(m["pnl_mean"]/m["pnl_std"]) if (m["pnl_std"] and not np.isnan(m["pnl_std"])) else np.nan
    else:
        m.update({"cum_pnl":0.0,"max_drawdown":0.0,"pnl_mean":np.nan,"pnl_std":np.nan,"mean_over_std":np.nan})
    return m


# ---------- Build candidate set (with pruning) ----------
def _variant_iter():
    for (mode, allow_big, big_margin,
         zmin, window, neigh_only, skip_short, req_cnt,
         z_entry, z_exit, max_hold, short_extra) in itertools.product(
        GRID_FLY_MODE, GRID_FLY_ALLOW_BIG, GRID_FLY_BIG_MARGIN,
        GRID_FLY_Z_MIN, GRID_FLY_WINDOW, GRID_FLY_NEIGH_ONLY, GRID_FLY_SKIP_SHORT, GRID_FLY_REQUIRE_COUNT,
        GRID_Z_ENTRY, GRID_Z_EXIT, GRID_MAX_HOLD_DAYS, GRID_SHORT_END_EXTRA_Z
    ):
        v = dict(
            FLY_MODE=mode,
            FLY_ALLOW_BIG_ZDISP=allow_big,
            FLY_BIG_ZDISP_MARGIN=big_margin,
            FLY_Z_MIN=zmin,
            FLY_WINDOW_YEARS=window,
            FLY_NEIGHBOR_ONLY=neigh_only,
            FLY_SKIP_SHORT_UNDER=skip_short,
            FLY_REQUIRE_COUNT=req_cnt,
            Z_ENTRY=z_entry,
            Z_EXIT=z_exit,
            MAX_HOLD_DAYS=max_hold,
            SHORT_END_EXTRA_Z=short_extra,
        )
        # Prune: fly params irrelevant when FLY_MODE='off'
        if mode == "off":
            # keep just one representative combo for irrelevant fly params
            if not (zmin == GRID_FLY_Z_MIN[0] and window == GRID_FLY_WINDOW[0] and skip_short == GRID_FLY_SKIP_SHORT[0]):
                continue
            v["FLY_ALLOW_BIG_ZDISP"] = False
        # Prune: ignore extra margins when not allowing big waiver
        if not allow_big and big_margin != GRID_FLY_BIG_MARGIN[0]:
            continue
        # Prune: req_cnt only matters for tolerant
        if mode != "tolerant" and req_cnt != GRID_FLY_REQUIRE_COUNT[0]:
            continue
        yield v

candidates = list(_variant_iter())
if RANDOMIZE: random.shuffle(candidates)
if MAX_VARIANTS and len(candidates) > MAX_VARIANTS:
    candidates = candidates[:MAX_VARIANTS]
print(f"[INFO] candidate variants: {len(candidates)}")


# ---------- Worker: run one variant (subprocess-safe) ----------
def _worker_run(variant: dict, yymms: list[str]) -> dict:
    """
    Runs in a separate process (no shared state).
    We import cr/pt fresh here so each process has a clean config.
    """
    import importlib
    import cr_config as _cr
    import portfolio_test as _pt

    # restore to originals in that process, then apply overrides
    for k, v in ORIG.items():
        try: setattr(_cr, k, v)
        except: pass
    for k, v in variant.items():
        if hasattr(_cr, k):
            setattr(_cr, k, v)

    # Ensure portfolio_test picks up mutated config
    importlib.reload(_pt)

    pos, led, by = _pt.run_all(yymms)

    # compute metrics here to reduce parent memory pressure
    n = int(len(pos)) if pos is not None else 0
    res = dict(variant)
    res["trades"] = n
    res["avg_pnl_per_trade"] = float(pos["pnl"].mean()) if n else np.nan
    if n:
        c = pos["exit_reason"].value_counts()
        res["exit_reversion"] = int(c.get("reversion", 0))
        res["exit_max_hold"]  = int(c.get("max_hold", 0))
        res["exit_stop"]      = int(c.get("stop", 0))
    else:
        res["exit_reversion"] = res["exit_max_hold"] = res["exit_stop"] = 0

    if by is not None and len(by):
        s = by.sort_values("bucket")["pnl"].astype(float).cumsum()
        res["cum_pnl"] = float(s.iloc[-1])
        dd = (s.cummax() - s)
        res["max_drawdown"] = float(dd.max())
        pnl = by["pnl"].astype(float)
        res["pnl_mean"] = float(pnl.mean())
        res["pnl_std"]  = float(pnl.std(ddof=1)) if len(pnl) > 1 else np.nan
        res["mean_over_std"] = float(res["pnl_mean"]/res["pnl_std"]) if (res["pnl_std"] and not np.isnan(res["pnl_std"])) else np.nan
    else:
        res.update({"cum_pnl":0.0,"max_drawdown":0.0,"pnl_mean":np.nan,"pnl_std":np.nan,"mean_over_std":np.nan})

    return res


# ---------- Run sweep (parallel if desired) ----------
def run_sweep(yymms: list[str], variants: list[dict], max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    rows: list[dict] = []

    if max_workers <= 1:
        # single-process (mutate config in-memory and reload pt)
        for i, v in enumerate(variants, 1):
            print(f"\n[{i}/{len(variants)}] {json.dumps(v, sort_keys=True)}")
            _apply_variant_inproc(v)
            pos, led, by = pt.run_all(yymms)
            rows.append({**v, **_metrics(pos, led, by)})
        # restore config
        _apply_variant_inproc({})
    else:
        print(f"[PARALLEL] running {len(variants)} variants with {max_workers} workers…")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_worker_run, v, yymms): v for v in variants}
            for j, fut in enumerate(as_completed(futs), 1):
                v = futs[fut]
                try:
                    res = fut.result()
                    rows.append(res)
                    print(f"[{j}/{len(variants)}] done: {json.dumps(v, sort_keys=True)} -> cum_pnl={res.get('cum_pnl'):.2f}")
                except Exception as e:
                    print(f"[ERROR] variant failed: {json.dumps(v, sort_keys=True)} | {e}")

    df = pd.DataFrame(rows)

    # Safe helpers to avoid divide-by-zero
    def _safe_div(a, b):
        try:
            return float(a) / float(b) if (b not in (0, None) and pd.notna(b)) else np.nan
        except Exception:
            return np.nan

    # Core risk/quality metrics
    if not df.empty:
        df["reversion_ratio"] = df.apply(lambda r: _safe_div(r.get("exit_reversion", np.nan), r.get("trades", np.nan)), axis=1)
        df["maxhold_ratio"]   = df.apply(lambda r: _safe_div(r.get("exit_max_hold",  np.nan), r.get("trades", np.nan)), axis=1)
        df["stop_ratio"]      = df.apply(lambda r: _safe_div(r.get("exit_stop",      np.nan), r.get("trades", np.nan)), axis=1)

        # Drawdown diagnostics
        df["dd_ratio"]        = df.apply(lambda r: _safe_div(r["max_drawdown"], r["cum_pnl"]) if r["cum_pnl"] > 0 else np.nan, axis=1)

        # Sharpe-like (already computed as mean_over_std; keep a clean alias)
        if "mean_over_std" in df.columns:
            df["sharpe_like"] = df["mean_over_std"]

        # Blended score: profit minus a penalty on drawdown (tune the 0.25 if you like)
        df["score"]           = df["cum_pnl"] - 0.25 * df["max_drawdown"]

        # Optional: risk-adjusted cum pnl (normalized by 1 + DD)
        df["cum_over_dd"]     = df.apply(lambda r: _safe_div(r["cum_pnl"], 1.0 + max(0.0, r["max_drawdown"])), axis=1)

        # Choose a clean column order for review (keep only those present)
        _cols = [
            # objective
            "score", "cum_pnl", "max_drawdown", "dd_ratio", "sharpe_like", "cum_over_dd",
            # trade quality
            "trades", "avg_pnl_per_trade", "reversion_ratio", "maxhold_ratio", "stop_ratio",
            # raw counts
            "exit_reversion", "exit_max_hold", "exit_stop",
            # swept knobs
            "FLY_MODE", "FLY_ALLOW_BIG_ZDISP", "FLY_BIG_ZDISP_MARGIN",
            "FLY_Z_MIN", "FLY_WINDOW_YEARS", "FLY_SKIP_SHORT_UNDER", "FLY_REQUIRE_COUNT",
            "Z_ENTRY", "Z_EXIT", "MAX_HOLD_DAYS", "SHORT_END_EXTRA_Z"
        ]
        _cols = [c for c in _cols if c in df.columns]

        # Final sorted table: prioritize score, then higher reversion_ratio, then lower dd_ratio
        df = (df
              .sort_values(["score", "reversion_ratio", "dd_ratio"], ascending=[False, False, True])
              .reset_index(drop=True))

        # Reorder for display/save (don’t drop other columns; just put the important ones first)
        front = [c for c in _cols if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]

    # Save
    out_dir = Path(getattr(cr, "PATH_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = getattr(cr, "OUT_SUFFIX", "")
    parq = out_dir / f"{SAVE_PREFIX}{suffix}.parquet"
    csv  = out_dir / f"{SAVE_PREFIX}{suffix}.csv"
    df.to_parquet(parq, index=False)
    df.to_csv(csv, index=False)
    print(f"[SAVE] {parq}\n[SAVE] {csv}")

    # Best variant JSON (based on score; if equal, highest reversion_ratio then lowest dd_ratio)
    if not df.empty:
        best_row = df.iloc[0].to_dict()
        best_variant = {k: best_row[k] for k in list(ORIG.keys()) if k in best_row}  # keep only known config keys
        with open(out_dir / f"{SAVE_PREFIX}_best{suffix}.json", "w") as f:
            json.dump(best_variant, f, indent=2)
        print("Best variant:\n", json.dumps(best_variant, indent=2))
    else:
        print("[WARN] No results to rank.")

    return df


# ---------- main ----------
if __name__ == "__main__":
    df = run_sweep(YYMMS, candidates, MAX_WORKERS)
    print(df.head(25))
