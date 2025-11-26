import time
import random
import threading
import sys
from pathlib import Path

# Hook for config
sys.path.append(str(Path(__file__).parent.parent))
import cr_config as cr
from .data_manager import log_ticks

class MockFeed:
    def __init__(self):
        self.live_map = {}
        self.running = False
        self.tickers = list(cr.TENOR_YEARS.keys())
        
        # Initialize reasonable starting rates (e.g., 4.00%)
        for t in self.tickers:
            self.live_map[t] = 4.00 + (cr.TENOR_YEARS[t] * 0.05) # Steep curve
            
    def start(self):
        self.running = True
        t = threading.Thread(target=self._run_loop)
        t.daemon = True
        t.start()
        
    def _run_loop(self):
        print("[FEED] Mock Feed Started...")
        while self.running:
            tick_batch = []
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            
            for t in self.tickers:
                # Random Walk: +/- 0.5 bps
                change = (random.random() - 0.5) * 0.005
                self.live_map[t] += change
                
                tick_batch.append({
                    'ts': ts,
                    'ticker': t,
                    'rate': round(self.live_map[t], 5)
                })
            
            # 1. Store locally for UI access
            # (In a real app, this might be a Redis cache)
            
            # 2. Log to DB for EOD
            log_ticks(tick_batch)
            
            time.sleep(1) # 1-second ticks

# Global instance
feed = MockFeed()
