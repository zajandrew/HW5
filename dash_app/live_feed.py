import blpapi
import threading
import time
import sys
from datetime import datetime
from pathlib import Path

# --- Robust Import Fix ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

import cr_config as cr
from .data_manager import log_ticks
import eod_process

class LiveFeed:
    def __init__(self):
        self.live_map = {}      
        self.last_db_log = {}   
        self.running = False
        self.db_buffer = []     
        self.last_flush_time = time.time()
        self.lock = threading.Lock()
        
        # State Flags
        self.eod_triggered = False
        self.market_open = False 
        
        # --- CONFIGURATION (MATCHING YOUR SCREENSHOT) ---
        self.host = "vpce-0c2e4f845a98f340e-t2ggznxv.vpce-svc-0a1ace7960b600239.us-east-2.vpce.amazonaws.com"
        self.port = 8194
        self.app_name = "USBANK:Rates_Data_Viewer"
        self.DB_LOG_INTERVAL = 5.0 
        
        # Time Windows (Local Hours)
        self.OPEN_HOUR = 7
        self.CLOSE_HOUR = 16

    def start(self):
        # Prevent double start
        if self.running: return

        self.session_options = blpapi.SessionOptions()
        self.session_options.setServerAddress(self.host, self.port, 0)
        self.session_options.setServerPort(self.port) # Redundant but safe matches your snippet
        
        # --- CRITICAL AUTHENTICATION BLOCK ---
        # 1. Create Auth Options with specific App Name
        authOptions = blpapi.AuthOptions.createWithApp(self.app_name)
        
        # 2. Create the exact Correlation ID
        self.auth_correlation_id = blpapi.CorrelationId("authCorrelation")
        
        # 3. Bind Identity to Session Options
        self.session_options.setSessionIdentityOptions(authOptions, self.auth_correlation_id)
        # -------------------------------------
        
        self.session = blpapi.Session(self.session_options)
        
        if not self.session.start():
            print("[FEED] Failed to start Bloomberg Session.")
            return
            
        if not self.session.openService("//blp/mktdata"):
            print("[FEED] Failed to open //blp/mktdata service.")
            return
            
        print("[FEED] Bloomberg Session Connected (Identity Verified).")
        self._subscribe()
        
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
            self.last_db_log[ticker] = 0.0
        self.session.subscribe(subscription_list)
        print(f"[FEED] Subscribed to {len(cr.TENOR_YEARS)} tickers.")

    def _run_loop(self):
        print(f"[FEED] Listener Loop Active. Schedule: {self.OPEN_HOUR}:00 - {self.CLOSE_HOUR}:00.")
        while self.running:
            try:
                # --- TIME & STATE MACHINE ---
                now = datetime.now()
                is_trading_hours = (self.OPEN_HOUR <= now.hour < self.CLOSE_HOUR)
                
                if is_trading_hours:
                    # MORNING STARTUP
                    if not self.market_open:
                        print(f"[SYSTEM] {now.strftime('%H:%M:%S')} - Market Open. DB Recording Started.")
                        self.market_open = True
                        self.eod_triggered = False 
                else:
                    # EVENING SHUTDOWN
                    if self.market_open:
                        print(f"[SYSTEM] {now.strftime('%H:%M:%S')} - Market Closed. DB Recording Stopped.")
                        self.market_open = False
                        self._flush_to_db() 
                        
                        if not self.eod_triggered:
                            print("[SYSTEM] Auto-Triggering EOD Process...")
                            self.eod_triggered = True
                            eod_thread = threading.Thread(target=eod_process.run_eod_main)
                            eod_thread.start()

                # --- EVENT PROCESSING ---
                event = self.session.nextEvent(1000)
                for msg in event:
                    if msg.hasElement("LAST_PRICE"):
                        self._process_message(msg)
                
                if self.market_open:
                    if time.time() - self.last_flush_time > 10:
                        self._flush_to_db()
                    
            except Exception as e:
                print(f"[FEED ERROR] {e}")
                time.sleep(5)

    def _process_message(self, msg):
        try:
            last_price = msg.getElementAsFloat("LAST_PRICE")
            ticker = msg.correlationIds()[0].value()
            current_time = time.time()
            
            # 1. Always Update RAM (For UI)
            self.live_map[ticker] = last_price
            
            # 2. Only Update DB if Market Open + Throttled
            if self.market_open:
                if (current_time - self.last_db_log.get(ticker, 0)) > self.DB_LOG_INTERVAL:
                    # UTC for Data Consistency
                    ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    with self.lock:
                        self.db_buffer.append({'ts': ts_str, 'ticker': ticker, 'rate': last_price})
                    self.last_db_log[ticker] = current_time
                
        except Exception:
            pass

    def _flush_to_db(self):
        with self.lock:
            if not self.db_buffer:
                self.last_flush_time = time.time()
                return
            data = list(self.db_buffer)
            self.db_buffer.clear()
        try:
            log_ticks(data)
        except Exception as e:
            print(f"[DB ERROR] {e}")
        self.last_flush_time = time.time()

feed = LiveFeed()
