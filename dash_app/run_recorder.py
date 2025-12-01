import time
import sys
from pathlib import Path

# Fix imports to find parent modules
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# Import the feed and the DB initializer
from dash_app.live_feed import feed
from dash_app.data_manager import init_dbs

if __name__ == "__main__":
    # Check for optional test flag
    TEST_MODE = "--test" in sys.argv
    
    print("="*40)
    print("STARTING STANDALONE DATA RECORDER")
    print("This process will run 24/7.")
    print("It will record to SQLite from 07:00 to 16:00.")
    print("It will run EOD Batch at 16:00.")
    print(f"VISUAL TEST MODE: {'ON' if TEST_MODE else 'OFF'}")
    print("="*40)
    
    # 1. Initialize Databases (Creates tables if missing)
    print("[SYSTEM] Verifying Database Schema...")
    init_dbs()
    
    # 2. Start the feed thread
    feed.start()
    
    # 3. Keep the main thread alive forever
    try:
        while True:
            if TEST_MODE:
                # Visual Feedback Loop
                print(f"\n--- TICK SNAPSHOT {time.strftime('%H:%M:%S')} ---")
                
                active_tickers = sorted(feed.live_map.keys())
                
                if not active_tickers:
                    print("Waiting for subscriptions...")
                
                for k in active_tickers:
                    val = feed.live_map[k]
                    val_str = f"{val:.4f}" if val != 0.0 else "WAITING..."
                    print(f"{k:<30} : {val_str}")
                
                time.sleep(3)
            else:
                # Silent Production Loop
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Recorder stopped by user.")
