"""
regime_manager.py

The Decision Engine. 
Ingests the 'Regime Multipliers' parquet and determines:
1. Which strategies are ACTIVE (Gating).
2. What Z-Score Adjustments to apply (Scoring).
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

# Import Config
import config as cr

class RegimeManager:
    def __init__(self, regime_path: Path):
        self.regime_path = regime_path
        self.df_regime = pd.DataFrame()
        self.lookup_cache = {}
        self._load_data()
        
    def _load_data(self):
        """Loads the regime multipliers/signals file."""
        if not self.regime_path.exists():
            print(f"[WARN] Regime file not found at {self.regime_path}. Defaulting to Neutral.")
            return
            
        df = pd.read_parquet(self.regime_path)
        
        # Ensure we have a datetime index for lookup
        if "decision_ts" in df.columns:
            df = df.set_index("decision_ts").sort_index()
        
        self.df_regime = df
        
    def get_state(self, ts: pd.Timestamp) -> Dict[str, float]:
        """
        Returns the raw signal state for a specific timestamp.
        Uses cached lookup for speed.
        """
        # Truncate to decision frequency (e.g., Hour)
        dts = ts.floor(cr.DECISION_FREQ)
        
        if dts in self.lookup_cache:
            return self.lookup_cache[dts]
            
        # Lookup in dataframe (with exact match, usually populated by merge previously)
        # Note: Strategy main loop usually handles alignment. 
        # Here we provide random access lookup.
        try:
            # We use 'asof' to find the most recent VALID regime data 
            # strictly BEFORE or AT this timestamp (depending on how file was built).
            # But remember: The file building script ALREADY lagged the signals.
            # So we can look up 'at' the timestamp.
            if dts in self.df_regime.index:
                row = self.df_regime.loc[dts]
                # If duplicate index, take last
                if isinstance(row, pd.DataFrame): 
                    row = row.iloc[-1]
                state = row.to_dict()
                self.lookup_cache[dts] = state
                return state
        except Exception:
            pass
            
        return {} # Empty if no data

    def check_trade_allowed(self, mode: str, state: Dict) -> bool:
        """
        GATING LOGIC: Returns True if 'mode' (curve/fly) is allowed 
        given the current regime 'state'.
        """
        if not state: return True # Default allow if no data
        
        # 1. Check Global Strategy Mode
        global_mode = cr.STRATEGY_MODE # 'curve', 'fly', 'both'
        if global_mode != "both" and global_mode != mode:
            return False
            
        # 2. Check "Both" Arbitration
        # If we are in "both" mode, we might only allow ONE type based on Hurst
        if global_mode == "both":
            arb_cfg = cr.REGIME_CONFIG.get("both", {}).get("priority", {})
            sig_name = arb_cfg.get("signal")
            split = arb_cfg.get("split_level")
            
            if sig_name and sig_name in state:
                val = state[sig_name]
                # e.g. Hurst > 0.55 -> Curve
                if val > split:
                    active = arb_cfg.get("above_split") # 'curve'
                else:
                    active = arb_cfg.get("below_split") # 'fly'
                
                if mode != active:
                    return False

        # 3. Check Specific Strategy Thresholds
        # e.g. "curve": {"threshold": {"hurst_max": (0.45, "greater")}}
        strat_cfg = cr.REGIME_CONFIG.get(mode, {})
        thresholds = strat_cfg.get("threshold", {})
        
        for sig, (thresh, operator) in thresholds.items():
            if sig not in state: continue
            val = state[sig]
            
            if operator == "greater" and val <= thresh: return False
            if operator == "less" and val >= thresh: return False
            
        return True

    def get_z_adjustment(self, mode: str, state: Dict, current_z_bar: float) -> float:
        """
        SCORING LOGIC: Returns the *Modified* Z-Entry threshold.
        """
        if not state: return current_z_bar
        
        strat_cfg = cr.REGIME_CONFIG.get(mode, {})
        multipliers = strat_cfg.get("multiplier", {})
        
        final_z = current_z_bar
        
        for sig, rule in multipliers.items():
            # rule is typically (trigger_val, op, impact)
            # e.g. (1.5, "multiply", 0.8)
            if sig not in state: continue
            val = state[sig]
            
            trigger, op, impact = rule
            
            # Simple logic: If Signal > Trigger, apply impact
            # (You can expand this to be directional if needed)
            if val > trigger:
                if op == "multiply":
                    final_z *= impact
                elif op == "add":
                    final_z += impact
                elif op == "subtract":
                    final_z -= impact
                    
        return final_z
