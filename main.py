# Jupyter one-off: add "_D" suffix to enhanced files and sweeper outputs

from pathlib import Path
import re

import cr_config as cr  # uses your configured PATH_ENH / PATH_OUT

# -------- Settings --------
SUFFIX = "_D"
DRY_RUN = True   # set to False to actually rename

# Sweeper artifacts to rename (prefixes without suffix)
# Add/remove prefixes if you use different names
SWEEPER_PREFIXES = ["sweep_results"]   # will catch parquet/csv + *_best.json

# --------------------------

enh_dir = Path(getattr(cr, "PATH_ENH", "."))
out_dir = Path(getattr(cr, "PATH_OUT", "."))

def add_suffix_before_ext(p: Path, suffix: str) -> Path:
    """Return a new path with suffix inserted before the final extension."""
    return p.with_name(p.stem + suffix + p.suffix)

def has_any_suffix(name: str, suffixes=("_D", "_H")) -> bool:
    """Detect if a filename stem already ends with a known suffix."""
    return any(name.endswith(s) for s in suffixes)

def resolve_collision(target: Path) -> Path:
    """If target exists, append a numeric counter _D2, _D3, ..."""
    if not target.exists():
        return target
    base, ext = target.stem, target.suffix
    m = re.match(r"^(.*?)(_D|_H)(\d+)?$", base)
    if m:
        root, sfx, num = m.groups()
        n = int(num) + 1 if num else 2
        return target.with_name(f"{root}{sfx}{n}{ext}")
    # generic fallback
    n = 2
    while True:
        cand = target.with_name(f"{base}{n}{ext}")
        if not cand.exists():
            return cand
        n += 1

plan = []

# 1) Enhanced parquet files: {yymm}_enh.parquet  ->  {yymm}_enh_D.parquet
# Only pick files that look like *_enh.parquet AND do not already carry _D/_H
for p in sorted(enh_dir.glob("*_enh.parquet")):
    stem = p.stem  # e.g., "2304_enh"
    if has_any_suffix(stem):  # already suffixed like *_enh_D or *_enh_H
        continue
    target = add_suffix_before_ext(p, SUFFIX)
    target = resolve_collision(target)
    plan.append((p, target))

# 2) Sweeper outputs:
#    sweep_results.parquet / sweep_results.csv / sweep_results_best.json  ->  *_D.*
for prefix in SWEEPER_PREFIXES:
    # core tables
    for ext in (".parquet", ".csv"):
        p = out_dir / f"{prefix}{ext}"
        if p.exists():
            stem = p.stem
            if not has_any_suffix(stem):
                target = add_suffix_before_ext(p, SUFFIX)
                target = resolve_collision(target)
                plan.append((p, target))
    # best-variant JSON
    pbest = out_dir / f"{prefix}_best.json"
    if pbest.exists():
        stem = pbest.stem  # e.g., "sweep_results_best"
        if not has_any_suffix(stem):
            # insert suffix before extension (after entire stem)
            target = add_suffix_before_ext(pbest, SUFFIX)
            target = resolve_collision(target)
            plan.append((pbest, target))

# ---- Show the plan ----
if not plan:
    print("[INFO] Nothing to rename (files already suffixed or not found).")
else:
    print(f"[INFO] Planned renames ({'DRY RUN' if DRY_RUN else 'COMMIT'}):")
    for src, dst in plan:
        print(f"  {src}  ->  {dst}")

    if not DRY_RUN:
        moved = 0
        for src, dst in plan:
            try:
                src.rename(dst)
                moved += 1
            except Exception as e:
                print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")
        print(f"[DONE] Renamed {moved}/{len(plan)} files.")
    else:
        print("Set DRY_RUN=False and re-run to perform the renames.")


