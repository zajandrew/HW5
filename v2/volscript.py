import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Regex: accepts "1M" or "1MX", base like "10Y"/"3M",
# optional whitespace followed by "100"/"-100"
MNEMONIC_RE = re.compile(r'^1M(?:X)?(?P<ticker>\d+[YM])(?:\s*(?P<suffix>-?100))?$')

def collect_volsofr_csvs(root_dir: str, glob_pattern="*VolSOFR.csv", recursive: bool = False):
    """
    Collect CSV files whose names include 'VolSOFR' (e.g., 20250311VolSOFR.csv).
    Returns a list of Path objects sorted by modified time.
    """
    base = Path(root_dir)
    files = list(base.rglob(glob_pattern)) if recursive else list(base.glob(glob_pattern))
    files = sorted(set(files), key=lambda f: f.stat().st_mtime)
    return files

def _detect_vol_column(df: pd.DataFrame) -> str:
    """
    Find the implied vol column and return its name.
    Strategy:
      1) Strip header whitespace.
      2) Exact matches: 'Implied Vol', 'ImpliedVol'.
      3) Fuzzy match: header contains both 'implied' and 'vol'.
      4) Fallback: second column (if first is Mnemonic).
      5) Absolute fallback: last column.
    """
    df.columns = df.columns.str.strip()

    for c in ('Implied Vol', 'ImpliedVol'):
        if c in df.columns:
            return c

    for c in df.columns:
        lc = c.lower().replace('_', ' ').strip()
        if ('implied' in lc) and ('vol' in lc):
            return c

    if 'Mnemonic' in df.columns and len(df.columns) >= 2:
        second = [col for col in df.columns if col != 'Mnemonic'][0]
        return second

    if len(df.columns) >= 2:
        return df.columns[-1]

    raise KeyError(f"Could not detect vol column from headers: {df.columns.tolist()}")

def load_and_filter(file_path: Path, ticker_map: dict | None = None) -> pd.DataFrame:
    """
    Read a CSV robustly, keep rows with valid 1M/1MX mnemonics (ATM or +/-100 with a space),
    then flatten to one row per base ticker with columns:
      - ticker (from mnemonic; mapping applied if provided)
      - Implied Vol (ATM)
      - Skew (vol(+100) - vol(-100))
      - file_modified, SourcePath, SourceFile
    """
    # Tolerant read (do NOT set low_memory with python engine)
    df = pd.read_csv(
        file_path,
        dtype=str,
        on_bad_lines='skip',
        engine='python'
        # sep=',',       # uncomment if delimiter detection is flaky
        # encoding='utf-8-sig' # try 'latin-1' if encoding issues occur
    )

    # Ensure Mnemonic exists
    df.columns = df.columns.str.strip()
    if 'Mnemonic' not in df.columns:
        # Fallback: assume first column is Mnemonic
        df.rename(columns={df.columns[0]: 'Mnemonic'}, inplace=True)

    # Normalize whitespace in Mnemonic (collapse multiple spaces)
    df['Mnemonic'] = df['Mnemonic'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

    # Filter & parse using one consistent regex
    mask = df['Mnemonic'].str.match(MNEMONIC_RE)
    df = df[mask].copy()

    extracted = df['Mnemonic'].str.extract(MNEMONIC_RE)
    # extracted has columns ['ticker', 'suffix'] from named groups
    df[['base_ticker', 'suffix']] = extracted[['ticker', 'suffix']]

    # Classify legs: atm / +100 / -100
    df['leg'] = np.where(df['suffix'].isna(), 'atm',
                         np.where(df['suffix'].eq('100'), 'p100',
                                  np.where(df['suffix'].eq('-100'), 'n100', 'other')))
    df = df[df['leg'].isin(['atm', 'p100', 'n100'])].copy()

    # Detect vol column and normalize its name to 'Implied Vol'
    vol_col = _detect_vol_column(df)
    if vol_col != 'Implied Vol':
        df.rename(columns={vol_col: 'Implied Vol'}, inplace=True)

    # Convert vol to numeric
    df['Implied Vol'] = pd.to_numeric(df['Implied Vol'], errors='coerce')

    # If nothing remains, return empty with expected schema (prevents concat issues)
    if df.empty:
        out = pd.DataFrame(columns=['ticker', 'Implied Vol', 'Skew', 'file_modified', 'SourcePath', 'SourceFile'])
        return out

    # Pivot per file to compute ATM and Skew for each base_ticker
    pivot = df.pivot_table(
        index='base_ticker',
        columns='leg',
        values='Implied Vol',
        aggfunc='first' # adjust to 'mean' or 'max' if multiple rows per leg may appear
    )

    # Ensure expected columns exist; fill missing with NaN
    for col in ['atm', 'p100', 'n100']:
        if col not in pivot.columns:
            pivot[col] = np.nan

    # Coerce to numeric safely
    pivot[['atm', 'p100', 'n100']] = pivot[['atm', 'p100', 'n100']].apply(pd.to_numeric, errors='coerce')

    # Skew = vol(+100) - vol(-100); if either leg missing -> NaN
    pivot['Skew'] = pivot['p100'].sub(pivot['n100'])

    # Build flattened output (always create 'Implied Vol' from 'atm')
    out = pivot.reset_index().rename(columns={'base_ticker': 'ticker'})
    out['Implied Vol'] = out['atm']

    # File metadata
    out['SourcePath'] = str(file_path)
    out['SourceFile'] = file_path.name

    # Do NOT set file_modified here; we'll set it later from the SourceFile date at 6pm ET -> UTC
    # Keep column present to preserve schema (optional)
    out['file_modified'] = pd.NaT

    # Apply mapping dictionary (replace ticker values)
    if ticker_map:
        out['ticker'] = out['ticker'].map(lambda x: ticker_map.get(x, x))

    # Final column order
    out = out[['ticker', 'Implied Vol', 'Skew', 'file_modified', 'SourcePath', 'SourceFile']]

    return out

def extract_trade_date_from_sourcefile(source_file: str) -> datetime.date:
    m = re.search(r'^(\d{8})', str(source_file)) or re.search(r'(\d{8})', str(source_file))
    if not m:
        raise ValueError(f"Could not find YYYYMMDD in SourceFile: {source_file}")
    return datetime.strptime(m.group(1), "%Y%m%d").date()

def set_6pm_newyork_from_sourcefile_then_utc(
    df: pd.DataFrame,
    source_col: str = 'SourceFile',
    timestamp_col: str = 'file_modified'
) -> pd.DataFrame:
    ny = ZoneInfo("America/New_York")
    utc = ZoneInfo("UTC")
    trade_dates = df[source_col].astype(str).map(extract_trade_date_from_sourcefile)
    six_pm_ny = pd.to_datetime(trade_dates).dt.tz_localize(ny) + pd.Timedelta("18:00:00")
    out = df.copy()
    out[timestamp_col] = six_pm_ny.dt.tz_convert(utc)
    return out

def interpolate_missing_tickers_on_days(
    df: pd.DataFrame,
    tenor_map: dict[str, float],
    restrict_days: list,
    timestamp_col: str = 'file_modified',
    value_cols: tuple[str, str] = ('Implied Vol', 'Skew'),
    group_tz: str = "UTC"
) -> pd.DataFrame:
    """
    Return ONLY the rows created by interpolation for the specified `restrict_days`.
    vol_col, skew_col = value_cols
    required_cols = {'ticker', timestamp_col, vol_col, skew_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"DataFrame missing required columns: {missing_cols}")
    """
    tz = ZoneInfo(group_tz)
    day_key = pd.to_datetime(df[timestamp_col]).dt.tz_convert(tz).dt.date
    universe = {t for t, x in tenor_map.items() if pd.notna(x)}

    out_rows = []
    df = df.assign(_day=day_key)
    df_restricted = df[df['_day'].isin(restrict_days)].copy()

    for day, g in df_restricted.groupby('_day'):
        g_use = g.groupby('ticker', as_index=False).agg({
            value_cols[0]: 'mean',
            value_cols[1]: 'mean',
            timestamp_col: 'first',
            'SourcePath': 'first',
            'SourceFile': 'first'
        })

        existing = [t for t in g_use['ticker'] if t in universe]
        missing = sorted(universe - set(existing))
        if not missing:
            continue

        X = np.array([tenor_map[t] for t in existing], dtype=float)
        if X.size == 0:
            continue

        Y_vol = g_use.set_index('ticker').loc[existing, value_cols[0]].astype(float).to_numpy()
        Y_skew = g_use.set_index('ticker').loc[existing, value_cols[1]].astype(float).to_numpy()

        order = np.argsort(X)
        Xs, Yvs, Yks = X[order], Y_vol[order], Y_skew[order]

        for t in missing:
            xt = float(tenor_map[t])
            vol_interp = np.interp(xt, Xs, Yvs, left=Yvs[0], right=Yvs[-1])
            skew_interp = np.interp(xt, Xs, Yks, left=Yks[0], right=Yks[-1])
            out_rows.append({
                'ticker': t,
                value_cols[0]: float(vol_interp),
                value_cols[1]: float(skew_interp),
                timestamp_col: g_use[timestamp_col].iloc[0],
                'SourcePath': '[INTERPOLATED]',
                'SourceFile': '[INTERPOLATED]'
            })

    add_df = pd.DataFrame(out_rows)
    if add_df.empty:
        return add_df
    
    add_df.drop_duplicates(subset=['ticker', timestamp_col], inplace=True)
    return add_df

def collect_vols(
    root_dir: str,
    out_csv: str = "VolSOFR_1M_concat.csv",
    ticker_map: dict | None = None,
    tenor_map: dict[str, float] | None = None,
    recursive: bool = False,
    write_out: bool = True
) -> pd.DataFrame:
    """
    Incremental updater:
      - Load existing out_csv if present.
      - Ingest ONLY new files (skip SourcePaths already present).
      - On new rows:
         * set timestamp from SourceFile date at 6pm America/New_York, then convert to UTC
         * interpolate missing tickers ONLY for the calendar dates found in new rows
      - Append new rows + new interpolations to existing; hard dedupe and sort.
      - Optionally write to out_csv and return the combined DataFrame.
    """
    files = collect_volsofr_csvs(root_dir, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No CSVs matching '*VolSOFR*.csv' found under: {root_dir}")

    out_path = Path(out_csv)

    # Load existing
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str)
        existing.columns = existing.columns.str.strip()
        # Ensure correct dtypes
        if 'file_modified' in existing.columns:
            existing['file_modified'] = pd.to_datetime(existing['file_modified'], errors='coerce')
        else:
            existing['file_modified'] = pd.NaT
        if 'ticker' not in existing.columns and 'Mnemonic' in existing.columns:
            existing.rename(columns={'Mnemonic': 'ticker'}, inplace=True)
        else:
            existing = pd.DataFrame(columns=['ticker', 'Implied Vol', 'Skew', 'file_modified', 'SourcePath', 'SourceFile'])
    else:
        existing = pd.DataFrame(columns=['ticker', 'Implied Vol', 'Skew', 'file_modified', 'SourcePath', 'SourceFile'])

    existing_paths = set(existing['SourcePath'].unique()) if 'SourcePath' in existing.columns else set()

    # Read only NEW files
    frames = []
    new_paths = set()
    for f in files:
        sp = str(f)
        if sp in existing_paths:
            continue
        try:
            df_new = load_and_filter(f, ticker_map=ticker_map)
            frames.append(df_new)
            new_paths.add(sp)
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

    # If nothing new -> return existing (and optionally rewrite for consistency)
    if not frames:
        combined = existing.copy()
        if write_out:
            combined.to_csv(out_csv, index=False)
        return combined

    # Concatenate NEW rows only
    new_rows = pd.concat(frames, ignore_index=True)

    # Set 6pm NY -> UTC on new rows ONLY
    new_rows = set_6pm_newyork_from_sourcefile_then_utc(
        new_rows, source_col='SourceFile', timestamp_col='file_modified'
    )

    # Rebuild combined as: old existing rows (non-new paths) + processed new_rows
    combined_existing = existing[~existing['SourcePath'].isin(new_paths)].copy()
    combined = pd.concat([combined_existing, new_rows], ignore_index=True)

    # Interpolate ONLY for the calendar dates appearing in new_rows
    if tenor_map:
        new_days = pd.to_datetime(new_rows['file_modified']).dt.tz_convert(ZoneInfo("UTC")).dt.date.unique().tolist()
        add_rows = interpolate_missing_tickers_on_days(
            df=combined,
            tenor_map=tenor_map,
            restrict_days=new_days,
            timestamp_col='file_modified',
            value_cols=('Implied Vol', 'Skew'),
            group_tz='UTC'
        )
        if not add_rows.empty:
            combined = pd.concat([combined, add_rows], ignore_index=True)

    # Hard dedupe and sort
    combined.drop_duplicates(subset=['ticker', 'file_modified'], keep='first', inplace=True)
    combined.sort_values(['file_modified', 'ticker'], inplace=True)

    if write_out:
        combined.to_csv(out_csv, index=False)
    return combined

if __name__ == "__main__":
    # >>> EDIT THESE <<<
    ROOT_DIR = r"\\us.bank-dns.com\mspmetro\MN14GL\Derivatives\Mortgage\PolyPathsCSVData"
    OUT_CSV = r"VolSOFR_1M_concat.csv"

    # Optional mapping of flattened tickers to your preferred labels
    TICKER_MAP = {
        "1Y": "USOSFR1 BGN Curncy", "2Y": "USOSFR2 BGN Curncy", "3Y": "USOSFR3 BGN Curncy",
        "4Y": "USOSFR4 BGN Curncy", "5Y": "USOSFR5 BGN Curncy", "6Y": "USOSFR6 BGN Curncy",
        "7Y": "USOSFR7 BGN Curncy", "10Y": "USOSFR10 BGN Curncy", "15Y": "USOSFR15 BGN Curncy",
        "20Y": "USOSFR20 BGN Curncy", "25Y": "USOSFR25 BGN Curncy", "30Y": "USOSFR30 BGN Curncy"
    }

    TENOR_YEARS = {
        "USOSFRA BGN Curncy": 1/12,   "USOSFRB BGN Curncy": 2/12,   "USOSFRC BGN Curncy": 3/12,
        "USOSFRD BGN Curncy": 4/12,   "USOSFRE BGN Curncy": 5/12,   "USOSFRF BGN Curncy": 6/12,
        "USOSFRG BGN Curncy": 7/12,   "USOSFRH BGN Curncy": 8/12,   "USOSFRK BGN Curncy": 11/12,
        "USOSFR1F BGN Curncy": 18/12, "USOSFR1 BGN Curncy": 1,      "USOSFR2 BGN Curncy": 2,
        "USOSFR3 BGN Curncy": 3,      "USOSFR4 BGN Curncy": 4,      "USOSFR5 BGN Curncy": 5,
        "USOSFR6 BGN Curncy": 6,      "USOSFR7 BGN Curncy": 7,      "USOSFR8 BGN Curncy": 8,
        "USOSFR9 BGN Curncy": 9,      "USOSFR10 BGN Curncy": 10,    "USOSFR12 BGN Curncy": 12,
        "USOSFR15 BGN Curncy": 15,    "USOSFR20 BGN Curncy": 20,    "USOSFR25 BGN Curncy": 25,
        "USOSFR30 BGN Curncy": 30,    "USOSFR40 BGN Curncy": 40,
    }

    combined = collect_vols(
        ROOT_DIR,
        OUT_CSV,
        ticker_map=TICKER_MAP,
        tenor_map=TENOR_YEARS, # interpolation guide
        recursive=False,
        write_out=True
    )
