import blpapi
import threading
import time
import sys
from datetime import datetime
from pathlib import Path

# Hook into parent directory for config
sys.path.append(str(Path(__file__).parent.parent))
import cr_config as cr
from .data_manager import log_ticks

class LiveFeed:
    def __init__(self):
        self.live_map = {}      # Shared memory for UI (Instant)
        self.last_db_log = {}   # Track last write time per ticker
        self.running = False
        self.db_buffer = []     # Buffer for SQL writes
        self.last_flush_time = time.time()
        self.lock = threading.Lock()
        
        # Connection Config
        self.host = "vpce-0c2e4f845a98f340e-t2ggznxv.vpce-svc-0a1ace7960b600239.us-east-2.vpce.amazonaws.com"
        self.port = 8194
        self.app_name = "USBANK:Rates_Data_Viewer"
        
        # Settings
        self.DB_LOG_INTERVAL = 5.0 # Seconds between DB writes per ticker

    def start(self):
        """Starts the Bloomberg Session and the Listener Thread."""
        self.session_options = blpapi.SessionOptions()
        self.session_options.setServerAddress(self.host, self.port, 0)
        self.session_options.setSessionIdentityOptions(
            blpapi.AuthOptions.createWithApp(self.app_name)
        )
        
        self.session = blpapi.Session(self.session_options)
        
        if not self.session.start():
            print("[FEED] Failed to start Bloomberg Session.")
            return
            
        if not self.session.openService("//blp/mktdata"):
            print("[FEED] Failed to open //blp/mktdata service.")
            return
            
        print("[FEED] Bloomberg Session Connected.")
        self._subscribe()
        
        # Start background thread
        self.running = True
        t = threading.Thread(target=self._run_loop)
        t.daemon = True
        t.start()
        
    def _subscribe(self):
        subscription_list = blpapi.SubscriptionList()
        
        for ticker in cr.TENOR_YEARS.keys():
            cid = blpapi.CorrelationId(ticker)
            subscription_list.add(ticker, "LAST_PRICE", "", cid)
            self.live_map[ticker] = 0.0
            self.last_db_log[ticker] = 0.0 # Initialize throttle tracker
            
        self.session.subscribe(subscription_list)
        print(f"[FEED] Subscribed to {len(cr.TENOR_YEARS)} tickers.")

    def _run_loop(self):
        while self.running:
            try:
                event = self.session.nextEvent(1000)
                for msg in event:
                    if msg.hasElement("LAST_PRICE"):
                        self._process_message(msg)
                
                # Periodic DB Flush (every 10 seconds)
                # This moves data from RAM buffer to SQLite
                if time.time() - self.last_flush_time > 10:
                    self._flush_to_db()
                    
            except Exception as e:
                print(f"[FEED ERROR] {e}")

    def _process_message(self, msg):
        try:
            last_price = msg.getElementAsFloat("LAST_PRICE")
            ticker = msg.correlationIds()[0].value()
            current_time = time.time()
            
            # 1. Update In-Memory (ALWAYS Instant for UI)
            self.live_map[ticker] = last_price
            
            # 2. Add to Buffer (THROTTLED for DB)
            # Only log if > 5 seconds have passed since last log for THIS ticker
            if (current_time - self.last_db_log.get(ticker, 0)) > self.DB_LOG_INTERVAL:
                
                ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                
                with self.lock:
                    self.db_buffer.append({
                        'ts': ts_str,
                        'ticker': ticker,
                        'rate': last_price
                    })
                
                # Update the last log time
                self.last_db_log[ticker] = current_time
                
        except Exception as e:
            print(f"[PARSE ERROR] {e}")

    def _flush_to_db(self):
        with self.lock:
            if not self.db_buffer:
                self.last_flush_time = time.time()
                return
            data_to_write = list(self.db_buffer)
            self.db_buffer.clear()
            
        try:
            log_ticks(data_to_write)
            # Optional: Print less frequently to avoid console spam
            # print(f"[FEED] Flushed {len(data_to_write)} ticks to DB.")
        except Exception as e:
            print(f"[DB ERROR] {e}")
            
        self.last_flush_time = time.time()

# Global Instance
feed = LiveFeed()
