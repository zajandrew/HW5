import blpapi
import threading
import time
import sys
from datetime import datetime
from pathlib import Path
from data_manager import log_ticks
import eod_process

# Hook into parent directory for config
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))
import cr_config as cr

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
        self.market_open = True
        
        # Config
        self.host = "x"
        self.port = 8194
        self.app_name = "x"
        self.DB_LOG_INTERVAL = 5.0 

    def start(self):
        self.session_options = blpapi.SessionOptions()
        self.session_options.setServerAddress(self.host, self.port, 0)
        
        # --- CORRECTED AUTHENTICATION LOGIC ---
        # Explicitly creating the Auth Options and Correlation ID as per your reference
        authOptions = blpapi.AuthOptions.createWithApp(self.app_name)
        self.auth_correlation_id = blpapi.CorrelationId("authCorrelation")
        
        self.session_options.setSessionIdentityOptions(authOptions, self.auth_correlation_id)
        # --------------------------------------
        
        self.session = blpapi.Session(self.session_options)
        
        if not self.session.start():
            print("[FEED] Failed to start Bloomberg Session.")
            return
            
        if not self.session.openService("//blp/mktdata"):
            print("[FEED] Failed to open //blp/mktdata service.")
            return
            
        print("[FEED] Bloomberg Session Connected (Identity Set).")
        self._subscribe()
        
        self.running = True
        t = threading.Thread(target=self._run_loop)
        t.daemon = True
        t.start()
        
    def _subscribe(self):
        subscription_list = blpapi.SubscriptionList()
        for ticker in cr.TENOR_YEARS.keys():
            # We map the event back to the ticker using CorrelationId(ticker)
            cid = blpapi.CorrelationId(ticker)
            subscription_list.add(ticker, "LAST_PRICE", "", cid)
            
            # Init state
            self.live_map[ticker] = 0.0
            self.last_db_log[ticker] = 0.0
            
        self.session.subscribe(subscription_list)
        print(f"[FEED] Subscribed to {len(cr.TENOR_YEARS)} tickers.")

    def _run_loop(self):
        print("[FEED] Listener loop started.")
        while self.running:
            try:
                # --- TIME CHECK ---
                now = datetime.now()
                # Check if it is past 16:00 (4 PM)
                if now.hour >= 16:
                    if self.market_open:
                        print("[SYSTEM] Market Closed (16:00). Stopping DB Writes.")
                        self.market_open = False
                        
                        # Flush any remaining ticks
                        self._flush_to_db()
                        
                        # TRIGGER AUTO-EOD
                        if not self.eod_triggered:
                            print("[SYSTEM] Auto-Triggering EOD Process...")
                            self.eod_triggered = True
                            # Run in separate thread
                            eod_thread = threading.Thread(target=eod_process.run_eod_main)
                            eod_thread.start()
                
                # --- EVENT LOOP ---
                event = self.session.nextEvent(1000)
                for msg in event:
                    # In a strict SAPI app, we might check for AUTHORIZATION_STATUS here
                    # using self.auth_correlation_id, but usually session.start() 
                    # handles the handshake sufficiently for data subscription.
                    
                    if msg.hasElement("LAST_PRICE"):
                        self._process_message(msg)
                
                # Only flush to DB if market is OPEN
                if self.market_open:
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
            
            # 2. Add to Buffer (ONLY if market is open)
            if self.market_open:
                if (current_time - self.last_db_log.get(ticker, 0)) > self.DB_LOG_INTERVAL:
                    ts_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    with self.lock:
                        self.db_buffer.append({'ts': ts_str, 'ticker': ticker, 'rate': last_price})
                    self.last_db_log[ticker] = current_time
                
        except Exception as e:
            # SAPI sometimes sends service messages that don't map to tickers
            pass

    def _flush_to_db(self):
        with self.lock:
            if not self.db_buffer:
                self.last_flush_time = time.time()
                return
            data_to_write = list(self.db_buffer)
            self.db_buffer.clear()
        try:
            log_ticks(data_to_write)
        except Exception as e:
            print(f"[DB ERROR] {e}")
        self.last_flush_time = time.time()

feed = LiveFeed()
