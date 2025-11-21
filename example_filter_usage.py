#example_filter_usage.py
#can be run after all other files have been run correctly

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import cr_config as cr
import hybrid_filter as hf
from hybrid_filter import RegimeThresholds, ShockConfig

# =========================================================
# 1) Load closed overlay positions (pnl_net_bp, close_ts)
# =========================================================
out_dir = Path(cr.PATH_OUT)

# This should be the *closed positions* output, not the mark-to-market ledger.
pos_path = out_dir / f"positions_ledger{cr.OUT_SUFFIX}.parquet"
pos_overlay = pd.read_parquet(pos_path)

# If the file also contains strategy-mode trades, keep only overlay:
if "mode" in pos_overlay.columns:
    pos_overlay = pos_overlay[pos_overlay["mode"] == "overlay"].copy()

pos_overlay["close_ts"] = pd.to_datetime(pos_overlay["close_ts"], utc=False, errors="coerce")
pos_overlay = pos_overlay.sort_values("close_ts")
print("Closed overlay positions:", len(pos_overlay))

# =========================================================
# 2) Build/load hybrid signals + regime & shock masks
# =========================================================

# Example regime thresholds (tweak later):
reg_thresholds = RegimeThresholds(
    min_signal_health_z=-0.5,   # require OK-ish health
    max_trendiness_abs=2.0,     # avoid highly trending / one-way regimes
    max_z_xs_mean_abs_z=2.0,    # avoid very extreme cross-sectional mean
)

# Example shock config (tweak later):
shock_cfg = ShockConfig(
    pnl_window=10,                       # 10-bucket window (days if DECISION_FREQ='D')
    use_raw_pnl=True,
    use_residuals=True,
    raw_pnl_z_thresh=-1.5,              # raw PnL shock threshold
    resid_z_thresh=-1.5,                # residual-based shock threshold
    regression_cols=[
        "signal_health_z",
        "trendiness_abs",
        "z_xs_mean_roll_z",
    ],                                  # will auto-drop any missing cols
    block_length=10,                    # block 10 buckets after each shock
)

hyb = hf.attach_regime_and_shock_masks(
    pos_overlay=pos_overlay,
    regime_thresholds=reg_thresholds,
    shock_cfg=shock_cfg,
    force_rebuild_signals=False,        # set True if you change RegimeConfig/base_window
)

signals      = hyb["signals"]
regime_mask  = hyb["regime_mask"]       # pd.Series indexed by decision_ts (ok_regime)
shock_res    = hyb["shock_results"]     # dict from run_shock_blocker

print("Signals shape:", signals.shape)
print("Regime mask length:", len(regime_mask))

# =========================================================
# 3) Build a diagnostic DataFrame on the common index
# =========================================================
idx          = shock_res["ts"]                      # DatetimeIndex
pnl_raw      = shock_res["pnl_raw"]                 # per-bucket PnL (bp) before blocking
pnl_blocked  = shock_res["pnl_blocked"]             # after shock-blocker (zeroed on blocks)
shock_mask   = shock_res["shock_mask"]              # True where bucket is shock
block_mask   = shock_res["block_mask"]              # True where we would BLOCK *after* shock

# Align regime mask to this index
reg = regime_mask.reindex(idx).fillna(False)

diag = pd.DataFrame({
    "pnl_raw":     pnl_raw,
    "pnl_blocked": pnl_blocked,
    "shock":       shock_mask,
    "block_shock": block_mask,
    "ok_regime":   reg,
}).sort_index()

# Hybrid "block" flag: either bad regime OR in a shock-blocked zone
diag["hybrid_block"] = (~diag["ok_regime"]) | diag["block_shock"]

print("\nDiag head:")
print(diag.head())

# =========================================================
# 4) Plot cumulative PnL: original vs shock-blocked vs hybrid-blocked
# =========================================================
pnl_raw_cum        = diag["pnl_raw"].cumsum()
pnl_shock_cum      = diag["pnl_blocked"].cumsum()
pnl_hybrid_cum     = diag["pnl_raw"].where(~diag["hybrid_block"], 0.0).cumsum()

plt.figure()
pnl_raw_cum.plot(label="Original (no filter)", linewidth=1.2)
pnl_shock_cum.plot(label="Shock-blocker only", linewidth=1.2)
pnl_hybrid_cum.plot(label="Hybrid (regime + shock)", linewidth=1.2)
plt.title("Overlay cumulative PnL (bp) – filters comparison")
plt.xlabel("Decision bucket")
plt.ylabel("Cumulative PnL (bp)")
plt.legend()
plt.grid(True)
plt.show()

# =========================================================
# 5) Quick stats: on vs off blocks (using ORIGINAL pnl_raw)
# =========================================================
def pnl_stats(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "n_days": 0,
            "mean_pnl": np.nan,
            "median_pnl": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }
    return {
        "label": label,
        "n_days": len(df),
        "mean_pnl": df["pnl_raw"].mean(),
        "median_pnl": df["pnl_raw"].median(),
        "p05": df["pnl_raw"].quantile(0.05),
        "p95": df["pnl_raw"].quantile(0.95),
    }

on_hybrid  = diag[diag["hybrid_block"]]
off_hybrid = diag[~diag["hybrid_block"]]

stats_on  = pnl_stats(on_hybrid,  "on_hybrid_block")
stats_off = pnl_stats(off_hybrid, "off_hybrid_block")

print("\n=== PnL per bucket: on vs off hybrid blocks (using ORIGINAL pnl_raw) ===")
print(pd.DataFrame([stats_on, stats_off]).set_index("label"))

# =========================================================
# 6) Optional: visualize hybrid-block days as vertical stripes
# =========================================================
plt.figure()
ax = pnl_raw_cum.plot(label="Original", linewidth=1.0)
pnl_hybrid_cum.plot(ax=ax, label="Hybrid-filtered", linewidth=1.0)

for dt, row in diag.iterrows():
    if row["hybrid_block"]:
        ax.axvspan(dt, dt, alpha=0.12)  # thin vertical highlight

ax.set_title("Cumulative PnL with hybrid-blocked periods shaded")
ax.set_xlabel("Decision bucket")
ax.set_ylabel("Cumulative PnL (bp)")
ax.legend()
ax.grid(True)
plt.show()
