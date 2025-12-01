import time
import sys
from pathlib import Path

# Fix imports to find parent modules
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# Reuse the robust LiveFeed
from dash_app.live_feed import feed

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
    
    # Start the feed thread
    feed.start()
    
    # Keep the main thread alive forever
    try:
        while True:
            if TEST_MODE:
                # Every 3 seconds, wipe console (optional) or just print block
                # printing block is safer for logs.
                print(f"\n--- TICK SNAPSHOT {time.strftime('%H:%M:%S')} ---")
                
                # Sort keys for readable output
                active_tickers = sorted(feed.live_map.keys())
                
                if not active_tickers:
                    print("Waiting for subscriptions...")
                
                for k in active_tickers:
                    val = feed.live_map[k]
                    # Visual cue: If 0.0, we haven't received a tick yet
                    val_str = f"{val:.4f}" if val != 0.0 else "WAITING..."
                    print(f"{k:<30} : {val_str}")
                
                time.sleep(3)
            else:
                # Silent Mode (Production)
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Recorder stopped by user.")
