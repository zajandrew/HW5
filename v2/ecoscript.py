import pandas as pd
from pathlib import Path
import numpy as np
import re
from zoneinfo import ZoneInfo

def filter_map_and_dedupe(
    df: pd.DataFrame,
    econ: list[str],
    auc: list[str],
    auc_dict: dict[str, str],
    timestamp_col: str = "Date Time",
    event_col: str = "Event",
    out_col: str = "event_type",
    case_insensitive: bool = False
) -> pd.DataFrame:
    """
    1) Filter rows:
       - Keep only rows where Event ∈ econ OR Event starts with any prefix in auc.
    
    2) Map new column `out_col`:
       - econ events -> 'econ'
       - auc prefix matches -> auc_dict[prefix] + '_mid'
         (If a prefix is missing in auc_dict, falls back to prefix + '_mid')
         
    3) Deduplicate:
       - Treat ALL econ events as one group ('econ'): keep unique timestamps across econ.
       - Auctions: dedupe on (event_type, timestamp) -> keep all distinct timestamps per prefix.
       
    Returns:
      DataFrame with columns [Event, Date Time, event_type], filtered and deduped.
    """
    # Basic checks
    if event_col not in df.columns or timestamp_col not in df.columns:
        raise ValueError(f"df must have columns ['{event_col}', '{timestamp_col}']")

    # Normalize event text
    s_event = df[event_col].astype(str).str.strip()
    econ_set = set(e.strip() for e in econ)

    # Build regex for prefix extraction (vectorized)
    flags = re.IGNORECASE if case_insensitive else 0
    if auc:
        pattern = re.compile(rf"^({'|'.join(map(re.escape, auc))})", flags)
        matched_prefix_full = s_event.str.extract(pattern, expand=False)
    else:
        matched_prefix_full = pd.Series(index=df.index, dtype="object")

    # Filter rows: econ exact OR prefix match
    mask = s_event.isin(econ_set) | matched_prefix_full.notna()
    kept = df.loc[mask, [event_col, timestamp_col]].copy()

    # Recompute matched prefix aligned to kept
    if auc:
        matched_prefix_kept = kept[event_col].astype(str).str.strip().str.extract(pattern, expand=False)
    else:
        matched_prefix_kept = pd.Series(index=kept.index, dtype="object")

    # event_type mapping
    # - econ -> 'econ'
    # - auc -> auc_dict[prefix] + '_mid' (fallback to prefix + '_mid' if missing in dict)
    auc_mapped = matched_prefix_kept.map(auc_dict)
    auc_mapped = np.where(
        pd.Series(auc_mapped, index=kept.index).notna(),
        pd.Series(auc_mapped, index=kept.index).astype(str),
        matched_prefix_kept.astype(str) + "_mid" # fallback if not in dict
    )

    kept[out_col] = np.where(
        kept[event_col].isin(econ_set),
        "econ",
        auc_mapped
    )

    # Parse timestamps to datetime
    if not np.issubdtype(kept[timestamp_col].dtype, np.datetime64):
        kept[timestamp_col] = pd.to_datetime(kept[timestamp_col], errors="coerce")
    kept = kept.dropna(subset=[timestamp_col])

    # Deduplicate:
    # - Econ: with event_type == 'econ', drop duplicates by timestamp (treat all econ equal)
    # - Auctions: drop duplicates by (event_type, timestamp)
    # Implemented uniformly by dropping duplicates on (event_type, timestamp)
    # because econ has a single event_type value ('econ'), which makes econ timestamps unique.
    kept = kept.drop_duplicates(subset=[out_col, timestamp_col], keep="first")

    # Sorted, tidy output
    out = (kept[[event_col, timestamp_col, out_col]]
           .sort_values([out_col, timestamp_col, event_col])
           .reset_index(drop=True))
    return out

def ensure_ny_then_utc(df: pd.DataFrame, timestamp_col: str = "Date Time") -> pd.DataFrame:
    """
    Treat existing timestamps as New York local time (tz-naive or already tz-aware),
    then convert to UTC. Returns a copy with tz-aware UTC timestamps.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in DataFrame.")

    ny = ZoneInfo("America/New_York")
    utc = ZoneInfo("UTC")

    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")

    # If timestamps are naive, localize them to NY; if already tz-aware, convert via NY then to UTC
    # (Going through NY ensures DST rules are applied as intended for local timestamps.)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(ny)
    else:
        # If they already carry a timezone (e.g., NY or something else),
        # first normalize to NY (for consistency), then convert to UTC.
        ts = ts.dt.tz_convert(ny)
    
    out[timestamp_col] = ts.dt.tz_convert(utc)
    return out

def synthesize_missing_auc_events_nearest(
    auc_df: pd.DataFrame,
    tenor_years: dict[str, float],
    timestamp_col: str = "Date Time",
    event_col: str = "Event",
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """
    For any ticker not present in auc_df but present in tenor_years:
      - Find the closest LOWER tenor neighbor among present tickers (max tenor <= target).
      - Find the closest UPPER tenor neighbor among present tickers (min tenor >= target).
      - For every timestamp where either closest neighbor has an event, create a synthetic row:
          Event = 'synthetic'
          event_type = missing ticker (key from tenor_years)
          Date Time = neighbor's timestamp (tz preserved)
    
    Ensures uniqueness: drops duplicate (event_type, Date Time) combinations.
    Returns a new DataFrame including synthetic rows.
    """
    # Basic schema checks
    for col in (timestamp_col, event_col, event_type_col):
        if col not in auc_df.columns:
            raise KeyError(f"Column '{col}' not found in auc_df.")

    df = auc_df.copy()

    # Normalize labels and timestamps
    df[event_type_col] = df[event_type_col].astype(str).str.strip()
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df[ts.notna()].copy()
    df[timestamp_col] = ts # tz-aware preserved if already set earlier

    # Tenor universe (strip keys, drop NaNs)
    ten_s = pd.Series(tenor_years, dtype="float").dropna()
    ten_s.index = ten_s.index.astype(str).str.strip()

    present = set(df[event_type_col].unique())
    universe = set(ten_s.index)
    missing = sorted(universe - present)
    if not missing:
        # Nothing to add
        return df

    # Tenors for present tickers (restrict to those with known tenor)
    present_tenors = {t: ten_s[t] for t in present if t in ten_s}

    # Precompute observed timestamps per present ticker (unique)
    ts_map = {
        t: pd.Index(df.loc[df[event_type_col] == t, timestamp_col]).unique()
        for t in present_tenors.keys()
    }

    synth_rows = []
    for tgt in missing:
        xt = ten_s.get(tgt)
        if pd.isna(xt):
            continue # skip if tenor missing
        
        # Closest LOWER (<= xt): max tenor
        lowers = [(t, v) for t, v in present_tenors.items() if v <= xt]
        lower = max(lowers, key=lambda x: x[1]) if lowers else None

        # Closest UPPER (>= xt): min tenor
        uppers = [(t, v) for t, v in present_tenors.items() if v >= xt]
        upper = min(uppers, key=lambda x: x[1]) if uppers else None

        # Union timestamps from the *closest* lower/upper neighbors only
        ts_union = pd.Index([])
        if lower:
            ts_union = ts_union.union(ts_map[lower[0]])
        if upper:
            ts_union = ts_union.union(ts_map[upper[0]])
        
        # If neither side exists, skip this missing ticker
        if ts_union.empty:
            continue

        # Emit synthetic rows at each unique timestamp
        for tstamp in ts_union:
            synth_rows.append({
                event_col: "synthetic",
                event_type_col: tgt,       # final ticker label from TENOR_YEARS
                timestamp_col: tstamp      # tz preserved (NY/UTC as already set upstream)
            })

    if not synth_rows:
        return df

    add_df = pd.DataFrame(synth_rows)

    # Combine and enforce uniqueness of (event_type, timestamp)
    result = pd.concat([df, add_df], ignore_index=True)
    result.drop_duplicates(subset=[event_type_col, timestamp_col], keep="first", inplace=True)

    # Stable sort for readability
    result.sort_values([timestamp_col, event_type_col, event_col], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result

def run_all():
    econ = ["S&P Global US Manufacturing PMI", "ISM Manufacturing", "S&P Global US Composite PMI",
            "ISM Services Index", "Change in Nonfarm Payrolls", "Unemployment Rate", "CPI YoY",
            "PPI Final Demand YoY", "Retail Sales Advance MoM", "U. of Mich. Sentiment", "Empire Manufacturing",
            "GDP Annualized QoQ", "PCE Price Index YoY", "FOMC Rate Decision (Upper Bound)"]
    auc = ["4W", "8W", "3M", "4M", "6M", "52W", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    auc_dict = {
        "4W" : "USOSFRA BGN Curncy",
        "8W" : "USOSFRB BGN Curncy",
        "3M" : "USOSFRC BGN Curncy",
        "4M" : "USOSFRD BGN Curncy",
        "6M" : "USOSFRF BGN Curncy",
        "52W" : "USOSFR1 BGN Curncy",
        "2Y" : "USOSFR2 BGN Curncy",
        "3Y" : "USOSFR3 BGN Curncy",
        "5Y" : "USOSFR5 BGN Curncy",
        "7Y" : "USOSFR7 BGN Curncy",
        "10Y" : "USOSFR10 BGN Curncy",
        "20Y" : "USOSFR20 BGN Curncy",
        "30Y" : "USOSFR30 BGN Curncy"
    }

    TENOR_YEARS = {
        "USOSFRA BGN Curncy": 1/12,   "USOSFRB BGN Curncy": 2/12,   "USOSFRC BGN Curncy": 3/12,
        "USOSFRD BGN Curncy": 4/12,   "USOSFRE BGN Curncy": 5/12,   "USOSFRF BGN Curncy": 6/12,
        "USOSFRG BGN Curncy": 7/12,   "USOSFRH BGN Curncy": 8/12,   "USOSFRI BGN Curncy": 9/12,
        "USOSFRJ BGN Curncy": 10/12,  "USOSFRK BGN Curncy": 11/12,  "USOSFR1 BGN Curncy": 1,
        "USOSFR1F BGN Curncy": 18/12, "USOSFR2 BGN Curncy": 2,      "USOSFR3 BGN Curncy": 3,
        "USOSFR4 BGN Curncy": 4,      "USOSFR5 BGN Curncy": 5,      "USOSFR6 BGN Curncy": 6,
        "USOSFR7 BGN Curncy": 7,      "USOSFR8 BGN Curncy": 8,      "USOSFR9 BGN Curncy": 9,
        "USOSFR10 BGN Curncy": 10,    "USOSFR12 BGN Curncy": 12,    "USOSFR15 BGN Curncy": 15,
        "USOSFR20 BGN Curncy": 20,    "USOSFR25 BGN Curncy": 25,    "USOSFR30 BGN Curncy": 30,
        "USOSFR40 BGN Curncy": 40,
    }

    # Point this to your folder (or use Path('.') if your notebook is already there)
    folder = Path(".")

    # List all CSV files in the folder
    csv_files = sorted(folder.glob("*.csv"))

    # Read and concatenate
    dfs = [pd.read_csv(f, index_col=None) for f in csv_files]
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Optional: drop common index-like columns that often sneak into CSVs
    for col in ["index", "Unnamed: 0"]:
        if col in combined.columns:
            combined = combined.drop(columns=col)

    # Ensure a clean default integer index
    combined_all = combined.reset_index(drop=True)

    filtered = filter_map_and_dedupe(combined_all, econ, auc, auc_dict)
    filtered = ensure_ny_then_utc(filtered, timestamp_col="Date Time")
    ecodata = filtered[filtered["event_type"] == "econ"].copy().reset_index(drop=True)
    aucdata = filtered[filtered["event_type"] != "econ"].copy().reset_index(drop=True)

    aucdata = synthesize_missing_auc_events_nearest(aucdata, TENOR_YEARS)

    ecodata.to_csv("eco_data.csv", index=False)
    aucdata.to_csv("auc_data.csv", index=False)

    return ecodata, aucdata

if __name__ == "__main__":
    run_all()
