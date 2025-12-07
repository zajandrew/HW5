import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly_express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time

# --- Hook into Parent Directory ---
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

# --- Local Imports ---
from data_manager import (init_dbs, add_position, get_open_positions, 
                          get_all_positions, update_position_status, 
                          get_system_state, delete_position)
import live_engine as le
from live_feed import feed
import eod_process
import hybrid_filter as hf
import cr_config as cr

# --- SETUP ---
init_dbs()

# START MODE: Viewer Only (assuming recorder is running)
feed.start(log_to_db=False)

# --- GLOBAL STATE ---
# Stores (filename, modification_timestamp)
CURRENT_LOADED_STATE = (None, 0.0)

def get_latest_model_info():
    """
    Returns (filename_yymm, mod_time) of the latest model file.
    """
    suffix = cr.ENH_SUFFIX
    enh_dir = Path(cr.PATH_ENH)
    files = list(enh_dir.glob(f"*_enh{suffix}.parquet"))
    if not files: 
        return (None, 0.0)
    
    # Sort by YYMM name
    files.sort()
    latest_file = files[-1]
    
    # Get file stats
    stats = latest_file.stat()
    yymm = latest_file.name.split('_')[0]
    
    return (yymm, stats.st_mtime)

def check_and_reload_model():
    """
    Checks if the model file has been modified (Morning Update).
    Reloads if timestamp or filename has changed.
    """
    global CURRENT_LOADED_STATE
    
    new_yymm, new_mtime = get_latest_model_info()
    
    curr_yymm, curr_mtime = CURRENT_LOADED_STATE
    
    # Reload if:
    # 1. We have no model loaded
    # 2. The filename changed (New Month)
    # 3. The file was modified (Same Month, New Morning Data)
    if (new_yymm is not None) and (new_yymm != curr_yymm or new_mtime > curr_mtime):
        
        print(f"[SYSTEM] Detected New Model: {new_yymm} (Time: {datetime.fromtimestamp(new_mtime)})")
        success = le.load_midnight_model(new_yymm)
        
        if success:
            CURRENT_LOADED_STATE = (new_yymm, new_mtime)
            return f"Model Updated: {new_yymm} @ {datetime.fromtimestamp(new_mtime).strftime('%H:%M')}"
            
    return dash.no_update

# Initial Load
check_and_reload_model()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "RV Overlay DSS"

# --- HELPERS ---
def get_z_color(z):
    if z is None: return "#444"
    if z > 2.0: return "#00FF00" 
    if z > 0.5: return "#90EE90"
    if z < -2.0: return "#FF0000"
    if z < -0.5: return "#CD5C5C"
    return "#444"

def get_drift_color(bps):
    if bps > 0.1: return "#00FF00"  # Positive Carry
    if bps < -0.1: return "#FF0000" # Negative Carry
    return "#888" # Neutral

def format_tenor(ticker, float_years):
    if float_years is None: return "N/A"
    y = float(float_years)
    tol = 0.001
    if abs(y - 1/12) < tol: return "1M"
    if abs(y - 0.25) < tol: return "3M"
    if abs(y - 0.5) < tol:  return "6M"
    if abs(y - 1.0) < tol:  return "1Y"
    if abs(y - round(y)) < tol: return f"{int(round(y))}Y"
    return f"{y:.1f}Y"

def assign_bucket(tenor):
    if tenor < 2.0: return "Short (<2Y)"
    if tenor <= 7.0: return "Belly (2-7Y)"
    return "Long (>7Y)"

def aggregate_pnl_columns(df):
    if df.empty: return 0,0,0,0
    return (
        df['realized_pnl_cash'].sum(),
        df['realized_pnl_price'].sum(),
        df['realized_pnl_carry'].sum(),
        df['realized_pnl_roll'].sum()
    )

# --- LAYOUT ---
app.layout = html.Div([
    dcc.Interval(id='fast-interval', interval=2000, n_intervals=0), 
    dcc.Interval(id='slow-interval', interval=60000, n_intervals=0), 
    
    # Header
    dbc.NavbarSimple(
        children=[
            dbc.Badge("REGIME: SAFE", color="success", className="ms-2", id="badge-regime"),
            dbc.Badge("SHOCK: OFF", color="success", className="ms-2", id="badge-shock"),
            html.Span(id="model-version-display", className="ms-3 text-muted small"),
            dbc.Button("Force EOD", id="btn-run-eod", size="sm", color="secondary", className="ms-4")
        ],
        brand="Systematic Overlay Framework // Live Desk",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    
    # Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Execution Ticket (Overlay Switch)")),
        dbc.ModalBody(id="modal-body-content"),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-cancel-trade", className="ms-auto", n_clicks=0),
            dbc.Button("Execute Position", id="btn-submit-trade", color="success", n_clicks=0)
        ])
    ], id="modal-trade", is_open=False, size="lg"),

    # Tabs
    dbc.Tabs([
        # TAB 1: TICKER SCANNER
        dbc.Tab(label="Ticker Scanner", children=[
            dbc.Container([
                html.H5("Market Grid (Live Z-Scores & Drift)", className="mt-3"),
                html.Div(id="ticker-grid", className="d-flex flex-wrap gap-2"),
                html.Div(id="scanner-msg", className="text-danger mt-2")
            ], fluid=True)
        ]),
        
        # TAB 2: BLOTTER
        dbc.Tab(label="Blotter", children=[
            dbc.Container([
                # Open Positions
                dbc.Row([
                    dbc.Col(html.H5("Active Positions (Composite Score & PnL Breakdown)"), width=10),
                    dbc.Col(dbc.Button("Refresh", id="btn-refresh-blotter", size="sm"), width=2)
                ], className="mt-3"),
                dash_table.DataTable(
                    id='tbl-open-positions',
                    style_table={'overflowX': 'auto'},
                    style_cell={'backgroundColor': '#222', 'color': 'white'},
                    row_selectable='single'
                ),
                dbc.Row([
                    dbc.Col(dbc.Select(
                        id="close-reason-select",
                        options=[
                            {"label": "Profit Take", "value": "profit"},
                            {"label": "Stop Loss", "value": "stop"},
                            {"label": "Max Hold", "value": "max_hold"},
                            {"label": "Manual / Discretionary", "value": "manual"}
                        ],
                        placeholder="Select Close Reason"
                    ), width=3),
                    dbc.Col(dbc.Button("Close Selected Trade", id="btn-close-trade", color="warning", disabled=True), width=3),
                    dbc.Col(dbc.Button("DELETE Trade (Mistake)", id="btn-delete-trade", color="danger", disabled=True), width=3)
                ], className="mt-2 mb-4"),
                
                html.Hr(),
                
                # Closed Positions
                html.H5("Trade History (Closed)", className="mt-3"),
                dash_table.DataTable(
                    id='tbl-history-positions',
                    style_table={'overflowX': 'auto', 'maxHeight': '300px', 'overflowY': 'scroll'},
                    style_cell={'backgroundColor': '#333', 'color': '#ddd'},
                    page_size=20,
                    sort_action='native'
                )
            ], fluid=True)
        ]),

        # --- UPDATE TAB 3: SUMMARY ---
        dbc.Tab(label="Summary", children=[
            dbc.Container([
                html.H4("Performance Dashboard", className="mt-4"),
                
                # FIX: Add Controls for PnL Unit and Time Filter
                dbc.Row([
                    dbc.Col(dbc.RadioItems(
                        id="pnl-unit-toggle",
                        options=[{"label": "Cash ($)", "value": "cash"}, {"label": "Basis Points", "value": "bp"}],
                        value="cash",
                        inline=True
                    ), width=4),
                    dbc.Col(dbc.Select(
                        id="time-filter",
                        options=[
                            {"label": "Last 1 Month", "value": "1M"},
                            {"label": "Last 3 Months", "value": "3M"},
                            {"label": "YTD", "value": "YTD"},
                            {"label": "All Time", "value": "ALL"}
                        ],
                        value="YTD"
                    ), width=4)
                ], className="mb-3"),
        
                html.Div(id="summary-cards", className="d-flex gap-3 mb-4 flex-wrap"),
        
                # FIX: Add Graph Components
                dbc.Row([
                    dbc.Col(dcc.Graph(id='chart-cumulative-pnl'), width=6),
                    dbc.Col(dcc.Graph(id='chart-attribution'), width=6)
                ]),
        
                dbc.Row([
                    # FIX: Add ID for stats table
                    dbc.Col([
                        html.H5("Time Horizon Stats"), 
                        html.Div(id="stats-table-container") 
                    ], width=6),
                    dbc.Col([
                        html.H5("Risk Profile (Net vs Gross)"), 
                        html.Div(id="risk-profile-container")
                    ], width=6)
                ]),
                
                # FIX: Add this hidden div or visible div to handle the 'modify_trade' output
                html.Div(id="summary-content", style={"display": "none"}) 
            ], fluid=True)
        ]),
        
        # TAB 4: HEALTH
        dbc.Tab(label="System Health", children=[
            dbc.Container([
              html.H4("Regime & Signal Diagnostics", className="mt-4"),
              # FIX: Add Graph Component
              dcc.Graph(id='chart-signal-health'),
              
              html.Div(id="health-cards", className="d-flex gap-3 mb-4"),
              html.H5("Filter Inputs (Last 10 Days)"),
              dash_table.DataTable(
                  id='tbl-signal-history', 
                  style_table={'overflowX': 'auto'}, 
                  style_cell={'backgroundColor': '#222', 'color': 'white'}
              )
          ], fluid=True)
       ])
    ])
])

# --- CALLBACKS ---

# 1. Slow Interval (Model Reload & Header)
@app.callback(
    Output('model-version-display', 'children'),
    Input('slow-interval', 'n_intervals')
)
def auto_reload_model(n):
    msg = check_and_reload_model()
    return f"Model: {CURRENT_LOADED_STATE}"

# 2. Update Ticker Grid (Scanner)
@app.callback(
    Output('ticker-grid', 'children'),
    Output('badge-regime', 'children'),
    Output('badge-regime', 'color'),
    Output('badge-shock', 'children'),
    Output('badge-shock', 'color'),
    Input('fast-interval', 'n_intervals')
)
def update_grid(n):
    state = get_system_state()
    shock_days = state[2] if state else 0
    regime = state[3] if state else "SAFE"
    
    is_shocked = (shock_days > 0)
    is_unstable = (regime != 'SAFE')
    global_disable = is_shocked or is_unstable
    
    z_map = le.get_live_z_scores(feed.live_map)
    if not z_map:
        # Return a placeholder card saying "Waiting for Market Data..."
        placeholder = dbc.Card(dbc.CardBody("System Initializing / No Live Data"), className="m-2")
        return [placeholder], "REGIME: INIT", "secondary", "SHOCK: OFF", "secondary"
    
    sorted_tickers = sorted(z_map.keys(), key=lambda t: cr.TENOR_YEARS.get(t, 99))
    
    cards = []
    for t in sorted_tickers:
        data = z_map[t]
        z_val = data['z_live']
        tenor_label = format_tenor(t, data['tenor'])
        
        # Calculate Natural Drift (Rec direction +1) for display
        drift_bps = le.calc_live_drift(data['tenor'], 1.0, feed.live_map)
        
        z_color = get_z_color(z_val)
        drift_color = get_drift_color(drift_bps)
        
        # Smart Disable Logic
        if global_disable:
            can_pay, can_rec = False, False
        else:
            partners_pay = le.get_valid_partners(t, 'PAY', z_map, feed.live_map)
            partners_rec = le.get_valid_partners(t, 'REC', z_map, feed.live_map)
            can_pay = len(partners_pay) > 0
            can_rec = len(partners_rec) > 0
        
        pay_style = {'opacity': 0.5 if not can_pay else 1.0}
        rec_style = {'opacity': 0.5 if not can_rec else 1.0}
        
        if z_val == 0.0 and data['model_z'] != 0.0: z_color = "#444"

        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"{tenor_label}", className="card-title text-center"),
                html.P(f"Z: {z_val:.2f}", className="text-center small mb-1", style={"fontWeight": "bold"}),
                html.P(f"Drift: {drift_bps:.2f}bp", className="text-center small mb-2", style={"color": drift_color}),
                html.Div([
                    dbc.Button("PAY", id={'type': 'btn-pay', 'index': t}, 
                               color="success" if can_pay else "secondary", 
                               size="sm", className="me-1", disabled=not can_pay, style=pay_style), 
                    dbc.Button("REC", id={'type': 'btn-rec', 'index': t}, 
                               color="danger" if can_rec else "secondary", 
                               size="sm", disabled=not can_rec, style=rec_style)
                ], className="d-flex justify-content-center")
            ])
        ], style={"width": "120px", "border": f"2px solid {z_color}"})
        cards.append(card)
        
    regime_text = f"REGIME: {regime}"
    regime_color = "success" if not is_unstable else "danger"
    shock_text = "SHOCK: OFF" if not is_shocked else f"SHOCK: {shock_days} DAYS"
    shock_color = "success" if not is_shocked else "danger"
    
    return cards, regime_text, regime_color, shock_text, shock_color

# 3. Modal Logic (Fixed Ghost Click)
@app.callback(
    Output("modal-trade", "is_open"),
    Output("modal-body-content", "children"),
    Input({'type': 'btn-pay', 'index': dash.ALL}, 'n_clicks'),
    Input({'type': 'btn-rec', 'index': dash.ALL}, 'n_clicks'),
    Input("btn-cancel-trade", "n_clicks"),
    Input("btn-submit-trade", "n_clicks"),
    State("modal-trade", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n_pay, n_rec, n_cancel, n_submit, is_open):
    ctx = callback_context
    if not ctx.triggered: return is_open
    
    # Get the trigger property and value
    triggered_prop = ctx.triggered[0]
    trigger_id = triggered_prop['prop_id'].split('.')[0]
    trigger_value = triggered_prop['value']

    # --- FIX: IGNORE INITIALIZATION EVENTS ---
    # Dash fires dynamic components with n_clicks=0 or None on creation.
    # We must explicitly ignore these to prevent the "Ghost Pop-up".
    if trigger_value == 0 or trigger_value is None:
        return is_open, dash.no_update
    # ----------------------------------------

    if "btn-cancel" in trigger_id or "btn-submit" in trigger_id:
        return False, dash.no_update
        
    if "btn-pay" in trigger_id or "btn-rec" in trigger_id:
        import json
        info = json.loads(trigger_id)
        ticker_orig = info['index']
        tenor_orig_yrs = cr.TENOR_YEARS.get(ticker_orig)
        label_orig = format_tenor(ticker_orig, tenor_orig_yrs)
        intent = "PAY" if "btn-pay" in trigger_id else "REC"
        
        z_map = le.get_live_z_scores(feed.live_map)
        # Pass live_map to get drift/composite
        candidates = le.get_valid_partners(ticker_orig, intent, z_map, feed.live_map)
        
        if not candidates:
            return True, html.Div(f"No valid overlay candidates found for {label_orig}.")
            
        current_rate = feed.live_map.get(ticker_orig, 0.0)
        
        # Build options with Composite Score visibility
        options = []
        for c in candidates:
            # Format: Ticker | Comp: X.X | Drift: X.Xbp | Z-Imp: X.X
            lbl = (f"{format_tenor(c['ticker'], c['tenor'])} "
                   f"| Comp: {c['composite']:.2f} "
                   f"| Drift: {c['drift_bps']:.2f}bp "
                   f"| Z: {c['z_live']:.2f}")
            
            disabled = c.get('is_gated', False)
            options.append({'label': lbl, 'value': c['ticker'], 'disabled': disabled})
            
        best_ticker = candidates[0]['ticker']
        best_rate = feed.live_map.get(best_ticker, 0.0)
        
        if intent == "PAY":
            leg1_label = "LEG 1: PAY (Alternative)"
            leg2_label = f"LEG 2: REC (Original {label_orig})"
        else:
            leg1_label = "LEG 1: REC (Alternative)"
            leg2_label = f"LEG 2: PAY (Original {label_orig})"
        
        content = html.Div([
            html.H5(f"Original Req: {intent} Fixed {label_orig}"),
            html.P("Select candidate based on Composite Score (Z + Drift).", className="text-muted small"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Label(leg1_label, className="fw-bold text-success"),
                    dbc.Select(id="trade-partner", options=options, value=best_ticker),
                ]),
                dbc.Col([dbc.Label("Size (DV01 in k)"), dbc.Input(id="trade-size", type="number", value=10)])
            ]),
            dbc.Row([
                dbc.Col([dbc.Label(f"Rate Alternative (Override)"), dbc.Input(id="rate-override-1", type="number", value=round(best_rate, 5))]),
                dbc.Col([dbc.Label(f"Rate {label_orig} (Override)"), dbc.Input(id="rate-override-2", type="number", value=round(current_rate, 5))])
            ], className="mt-2"),
            dcc.Store(id="store-active-ticker", data=ticker_orig),
            dcc.Store(id="store-intent", data=intent)
        ])
        return True, content
    return is_open, dash.no_update

# 4. Submit Trade
@app.callback(
    Output("scanner-msg", "children"),
    Input("btn-submit-trade", "n_clicks"),
    State("store-active-ticker", "data"),
    State("store-intent", "data"),
    State("trade-partner", "value"),
    State("trade-size", "value"),
    State("rate-override-1", "value"),
    State("rate-override-2", "value"),
    prevent_initial_call=True
)
def submit_trade(n, ticker_orig, intent, ticker_alt, size, r_alt, r_orig):
    if not ticker_orig or not ticker_alt: return dash.no_update
    
    if intent == "PAY":
        t_pay, t_rec = ticker_alt, ticker_orig
        r_pay, r_rec = r_alt, r_orig
        pay_dir, rec_dir = -1.0, 1.0
    else:
        t_pay, t_rec = ticker_orig, ticker_alt
        r_pay, r_rec = r_orig, r_alt
        pay_dir, rec_dir = -1.0, 1.0
        
    tenor_pay = cr.TENOR_YEARS.get(t_pay)
    tenor_rec = cr.TENOR_YEARS.get(t_rec)
    z_map = le.get_live_z_scores(feed.live_map)
    z_spread = z_map.get(t_pay, {}).get('z_live', 0) - z_map.get(t_rec, {}).get('z_live', 0)
    model_z = z_map.get(t_pay, {}).get('model_z', 0) - z_map.get(t_rec, {}).get('model_z', 0)
    
    # Capture Entry Drift & Composite for Record Keeping
    drift_pay = le.calc_live_drift(tenor_pay, pay_dir, feed.live_map)
    drift_rec = le.calc_live_drift(tenor_rec, rec_dir, feed.live_map)
    total_drift = drift_pay + drift_rec
    
    DRIFT_WEIGHT = float(getattr(cr, "DRIFT_WEIGHT", 0.2))
    composite = z_spread + (DRIFT_WEIGHT * total_drift)
    
    trade = {
        'open_ts': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'OPEN',
        'ticker_pay': t_pay,
        'ticker_rec': t_rec,
        'tenor_pay': tenor_pay,
        'tenor_rec': tenor_rec,
        'dv01': float(size) * 1000,
        'entry_rate_pay': float(r_pay),
        'entry_rate_rec': float(r_rec),
        'entry_z_spread': z_spread,
        'model_z_score': model_z,
        'entry_drift_bps': total_drift,   # NEW FIELD
        'entry_composite_score': composite # NEW FIELD
    }
    add_position(trade)
    return f"Trade Executed: Pay {t_pay} / Rec {t_rec}"

# 5. Update Blotter
@app.callback(
    Output('tbl-open-positions', 'data'),
    Output('tbl-open-positions', 'columns'),
    Output('tbl-open-positions', 'style_data_conditional'),
    Output('tbl-history-positions', 'data'),
    Output('tbl-history-positions', 'columns'),
    Input('fast-interval', 'n_intervals'),
    Input('btn-refresh-blotter', 'n_clicks')
)
def update_blotter(n, click):
    all_trades = get_all_positions()
    if all_trades.empty: return [], [], [], [], []
    
    Z_EXIT, Z_STOP, MAX_HOLD = cr.Z_EXIT, cr.Z_STOP, cr.MAX_HOLD_DAYS
    DRIFT_WEIGHT = float(getattr(cr, "DRIFT_WEIGHT", 0.2))
    now_dt = datetime.now()
    
    # --- OPEN TRADES ---
    open_df = all_trades[all_trades['status'] == 'OPEN']
    open_data = []
    
    for _, row in open_df.iterrows():
        (c_tot, c_prc, c_cry, c_rol), (b_tot, b_prc, b_cry, b_rol) = le.calculate_live_pnl(row, feed.live_map, now_dt)
        
        # Calculate Live Composite Score
        z_map = le.get_live_z_scores(feed.live_map)
        zp = z_map.get(row['ticker_pay'], {}).get('z_live', 0)
        zr = z_map.get(row['ticker_rec'], {}).get('z_live', 0)
        curr_z = zp - zr
        
        # Live Drift
        drift_p = le.calc_live_drift(row['tenor_pay'], -1.0, feed.live_map)
        drift_r = le.calc_live_drift(row['tenor_rec'], 1.0, feed.live_map)
        curr_drift = drift_p + drift_r
        
        curr_comp = curr_z + (DRIFT_WEIGHT * curr_drift)

        entry_z = row['entry_z_spread']
        open_dt = pd.to_datetime(row['open_ts'])
        days_held = (now_dt - open_dt).days
        stop_z_level = entry_z + (np.sign(entry_z) * Z_STOP) if entry_z != 0 else Z_STOP

        bg_color = "#222"
        db_reason = row.get('close_reason')
        if db_reason and db_reason not in [None, 'None', '']:
             if db_reason == 'max_hold': flag, bg_color = "MAX HOLD", "#B8860B"
             elif db_reason == 'stop_loss': flag, bg_color = "STOP LOSS (EOD)", "#8B0000"
             elif db_reason == 'reversion': flag, bg_color = "TAKE PROFIT (EOD)", "#006400"
             else: flag = f"CHECK: {db_reason}"
        else:
             flag = "OPEN"
             if abs(curr_z) < Z_EXIT: flag, bg_color = "TAKE PROFIT", "#006400"
             elif abs(curr_z - entry_z) > Z_STOP: flag, bg_color = "STOP LOSS", "#8B0000"

        label_pay = format_tenor(row['ticker_pay'], row['tenor_pay'])
        label_rec = format_tenor(row['ticker_rec'], row['tenor_rec'])
        
        open_data.append({
            'trade_id': row['trade_id'],
            'open_date': row['open_ts'].split(' ')[0], 
            'pair': f"Pay {label_pay} / Rec {label_rec}",
            'dv01': f"{int(row['dv01']/1000)}k",
            'entry_z': round(entry_z, 2),  # Added
            'curr_z': round(curr_z, 2),
            'curr_drift': round(curr_drift, 2),
            'curr_comp': round(curr_comp, 2),
            'target_z': f"0.0 ± {Z_EXIT}",
            'total_pnl': round(c_tot, 0),
            'price_pnl': round(c_prc, 0),
            'carry_pnl': round(c_cry, 0),
            'roll_pnl': round(c_rol, 0),
            'aging': f"{days_held} / {MAX_HOLD}",
            'status': flag,
            '_row_color': bg_color 
        })

    # --- CLOSED TRADES ---
    hist_df = all_trades[all_trades['status'] == 'CLOSED'].sort_values('close_ts', ascending=False)
    hist_data = []
    
    for _, row in hist_df.iterrows():
        label_pay = format_tenor(row['ticker_pay'], row['tenor_pay'])
        label_rec = format_tenor(row['ticker_rec'], row['tenor_rec'])
        
        hist_data.append({
            'trade_id': row['trade_id'], # Added
            'open_date': str(row['open_ts']).split(' ')[0], # Added
            'close_ts': row['close_ts'],
            'pair': f"Pay {label_pay} / Rec {label_rec}",
            'dv01': f"{int(row['dv01']/1000)}k",
            'entry_z': round(row.get('entry_z_spread', 0), 2), # Added
            'exit_z': round(row.get('close_z_spread', 0), 2),  # Added (assuming column exists)
            'total_pnl': f"${row['realized_pnl_cash']:,.0f}",
            'price_pnl': f"${row.get('realized_pnl_price', 0):,.0f}",
            'carry_pnl': f"${row.get('realized_pnl_carry', 0):,.0f}",
            'roll_pnl': f"${row.get('realized_pnl_roll', 0):,.0f}",
            'reason': row['close_reason']
        })

    # --- COLUMNS DEFINITIONS ---
    
    # Updated Open Columns (reordered logically)
    open_cols_list = [
        'trade_id', 'open_date', 'pair', 'dv01', 
        'entry_z', 'curr_z', 'curr_drift', 'curr_comp', 'target_z', 
        'total_pnl', 'price_pnl', 'carry_pnl', 'roll_pnl', 
        'aging', 'status'
    ]
    open_cols = [{"name": i.replace('_', ' ').title(), "id": i} for i in open_cols_list]
    
    # Updated History Columns
    hist_cols_list = [
        'trade_id', 'open_date', 'close_ts', 'pair', 'dv01', 
        'entry_z', 'exit_z', 
        'total_pnl', 'price_pnl', 'carry_pnl', 'roll_pnl', 
        'reason'
    ]
    hist_cols = [{"name": i.replace('_', ' ').title(), "id": i} for i in hist_cols_list]

    style_cond = []
    for i, row in enumerate(open_data):
        if row['_row_color'] != "#222":
            style_cond.append({
                'if': {'filter_query': f'{{trade_id}} = {row["trade_id"]}'}, 
                'backgroundColor': row['_row_color'], 
                'color': 'white'
            })

    return open_data, open_cols, style_cond, hist_data, hist_cols

# 6. Modify Trade
@app.callback(
    Output('summary-content', 'children'), 
    Input('btn-close-trade', 'n_clicks'),
    Input('btn-delete-trade', 'n_clicks'),
    State('tbl-open-positions', 'selected_rows'),
    State('tbl-open-positions', 'data'),
    State('close-reason-select', 'value'),
    prevent_initial_call=True
)
def modify_trade(n_close, n_del, rows, data, reason):
    ctx = callback_context
    if not rows or not ctx.triggered: return dash.no_update
    
    row_idx = rows[0]
    ui_row = data[row_idx]
    trade_id = ui_row['trade_id']
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-delete-trade": 
        delete_position(trade_id)
        return "Trade Deleted"

    elif button_id == "btn-close-trade":
        all_pos = get_all_positions()
        matches = all_pos[all_pos['trade_id'] == trade_id]
        if matches.empty: return "Error: Trade Not Found"
        
        original_trade = matches.iloc[0]
        now_dt = datetime.now()
        
        (c_tot, c_prc, c_cry, c_rol), (b_tot, b_prc, b_cry, b_rol) = le.calculate_live_pnl(
            original_trade, feed.live_map, now_dt
        )
        
        c_dict = {'total': c_tot, 'price': c_prc, 'carry': c_cry, 'roll': c_rol}
        b_dict = {'total': b_tot, 'price': b_prc, 'carry': b_cry, 'roll': b_rol}
        
        update_position_status(
            trade_id, "CLOSED", reason or "manual", 
            now_dt.strftime("%Y-%m-%d %H:%M:%S"), 
            pnl_cash=c_dict, pnl_bp=b_dict
        )
        
    return "Action Complete"

# 7. EOD Trigger
@app.callback(
    Output('btn-run-eod', 'children'),
    Input('btn-run-eod', 'n_clicks'),
    prevent_initial_call=True
)
def run_eod_batch(n):
    eod_process.run_eod_main()
    return "Batch Completed"

@app.callback(
    Output('summary-cards', 'children'),
    Output('chart-cumulative-pnl', 'figure'),
    Output('chart-attribution', 'figure'),
    Output('stats-table-container', 'children'),
    Output('risk-profile-container', 'children'),
    Input('slow-interval', 'n_intervals'),
    Input('pnl-unit-toggle', 'value'),
    Input('time-filter', 'value'),
    Input('btn-close-trade', 'n_clicks')
)
def update_enhanced_summary(n, unit, time_filter, close_trig):
    all_trades = get_all_positions()
    if all_trades.empty: 
        empty_fig = go.Figure().update_layout(template='plotly_dark')
        return [], empty_fig, empty_fig, "No Data", "No Data"
    
    # 1. Unit Selection (Cash vs Bps)
    is_cash = (unit == 'cash')
    col_map = {
        'tot': 'realized_pnl_cash' if is_cash else 'realized_pnl_bp',
        'prc': 'realized_pnl_price' if is_cash else 'realized_pnl_price_bp',
        'cry': 'realized_pnl_carry' if is_cash else 'realized_pnl_carry_bp',
        'rol': 'realized_pnl_roll' if is_cash else 'realized_pnl_roll_bp'
    }
    fmt = "${:,.0f}" if is_cash else "{:,.1f} bp"
    
    # 2. Time Filtering
    all_trades['dt'] = pd.to_datetime(all_trades['open_ts'])
    now = datetime.now()
    
    if time_filter == '1M':
        start_date = now - timedelta(days=30)
    elif time_filter == '3M':
        start_date = now - timedelta(days=90)
    elif time_filter == 'YTD':
        start_date = datetime(now.year, 1, 1)
    else:
        start_date = datetime(1900, 1, 1)
        
    df_filtered = all_trades[all_trades['dt'] >= start_date].copy()
    
    # 3. Cards (Closed Trades Only for PnL)
    closed = df_filtered[df_filtered['status'] == 'CLOSED']
    
    def sum_col(c): return closed[c].sum() if not closed.empty else 0
    
    # Add Open PnL if needed? Usually summary focuses on Realized for charts, but Total for cards.
    # Let's stick to realized for charts, Total for Top Card.
    
    c_tot = sum_col(col_map['tot'])
    
    # Open PnL (Approx)
    open_trades = df_filtered[df_filtered['status'] == 'OPEN']
    o_tot = 0
    for _, row in open_trades.iterrows():
        (c_cash, _, _, _), (c_bp, _, _, _) = le.calculate_live_pnl(row, feed.live_map, now)
        o_tot += c_cash if is_cash else c_bp
        
    grand_total = c_tot + o_tot
    
    cards = [
        dbc.Card([dbc.CardBody([html.H6("Total PnL (Net)", className="card-title"), html.H4(fmt.format(grand_total), className="text-success" if grand_total>=0 else "text-danger")])], style={"width": "180px"}),
        dbc.Card([dbc.CardBody([html.H6("Realized Price", className="card-title"), html.H4(fmt.format(sum_col(col_map['prc'])), className="text-info")])], style={"width": "180px"}),
        dbc.Card([dbc.CardBody([html.H6("Realized Carry", className="card-title"), html.H4(fmt.format(sum_col(col_map['cry'])), className="text-warning")])], style={"width": "180px"}),
        dbc.Card([dbc.CardBody([html.H6("Realized Roll", className="card-title"), html.H4(fmt.format(sum_col(col_map['rol'])), className="text-warning")])], style={"width": "180px"}),
    ]
    
    # 4. Charts (Daily Aggregation)
    if not closed.empty:
        closed['day'] = pd.to_datetime(closed['close_ts']).dt.floor('D')
        daily = closed.groupby('day')[[col_map['tot'], col_map['prc'], col_map['cry'], col_map['rol']]].sum().reset_index()
        daily = daily.sort_values('day')
        daily['cum_pnl'] = daily[col_map['tot']].cumsum()
        
        # Chart 1: Cumulative
        fig_cum = px.line(daily, x='day', y='cum_pnl', title="Cumulative Realized PnL", template='plotly_dark')
        fig_cum.update_traces(line_color='#00FF00', line_width=3)
        
        # Chart 2: Attribution
        fig_attr = go.Figure()
        fig_attr.add_trace(go.Bar(x=daily['day'], y=daily[col_map['prc']], name='Price', marker_color='#1E90FF'))
        fig_attr.add_trace(go.Bar(x=daily['day'], y=daily[col_map['cry']], name='Carry', marker_color='#FFD700'))
        fig_attr.add_trace(go.Bar(x=daily['day'], y=daily[col_map['rol']], name='Roll', marker_color='#FFA500'))
        fig_attr.update_layout(barmode='relative', title="Daily PnL Attribution", template='plotly_dark')
    else:
        fig_cum = go.Figure().update_layout(template='plotly_dark', title="No Closed Trades")
        fig_attr = go.Figure().update_layout(template='plotly_dark', title="No Closed Trades")

    # 5. Stats Table
    stats_data = {
        "Metric": ["Trades Executed", "Win Rate", "Avg Trade", "Best Trade", "Worst Trade"],
        "Value": [
            len(closed),
            f"{(len(closed[closed[col_map['tot']] > 0]) / len(closed) * 100):.1f}%" if len(closed)>0 else "N/A",
            fmt.format(closed[col_map['tot']].mean()) if not closed.empty else "N/A",
            fmt.format(closed[col_map['tot']].max()) if not closed.empty else "N/A",
            fmt.format(closed[col_map['tot']].min()) if not closed.empty else "N/A",
        ]
    }
    stats_table = dbc.Table.from_dataframe(pd.DataFrame(stats_data), striped=True, bordered=True, dark=True)
    
    # 6. Risk Profile
    risk_data = []
    bucket_risk = {"Short": [0,0], "Belly": [0,0], "Long": [0,0]}
    for _, row in open_trades.iterrows():
        b_pay, b_rec = assign_bucket(row['tenor_pay']).split()[0], assign_bucket(row['tenor_rec']).split()[0]
        dv = row['dv01']
        bucket_risk[b_pay][0] -= dv; bucket_risk[b_pay][1] += dv
        bucket_risk[b_rec][0] += dv; bucket_risk[b_rec][1] += dv
        
    for b, (net, gross) in bucket_risk.items():
        risk_data.append({"Bucket": b, "Net DV01": f"{int(net/1000)}k", "Gross DV01": f"{int(gross/1000)}k"})
    risk_table = dbc.Table.from_dataframe(pd.DataFrame(risk_data), striped=True, bordered=True, dark=True)
    
    return cards, fig_cum, fig_attr, stats_table, risk_table

@app.callback(
    Output('chart-signal-health', 'figure'),
    Output('health-cards', 'children'),
    Output('tbl-signal-history', 'data'),
    Output('tbl-signal-history', 'columns'),
    Input('slow-interval', 'n_intervals')
)
def update_health_charts(n):
    signals = hf.get_or_build_hybrid_signals()
    if signals.empty: return go.Figure(), [], [], []
    
    # Chart
    recent_sig = signals.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_sig['decision_ts'], y=recent_sig['signal_health_z'], name="Health Z", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=recent_sig['decision_ts'], y=recent_sig['trendiness_abs'], name="Trendiness", line=dict(color='red', dash='dot')))
    fig.add_hline(y=cr.MIN_SIGNAL_HEALTH_Z, line_color="white", annotation_text="Health Limit")
    fig.update_layout(template='plotly_dark', title="Regime History (Last 60 Days)", yaxis_title="Z-Score")
    
    # Cards
    last = signals.iloc[-1]
    metrics = {
        "Health (Z)": (last.get('signal_health_z', -99), cr.MIN_SIGNAL_HEALTH_Z, "gt"),
        "Trend (Abs)": (last.get('trendiness_abs', 0), cr.MAX_TRENDINESS_ABS, "lt"),
        "Mean (Z)": (last.get('z_xs_mean_roll_z', 0), cr.MAX_Z_XS_MEAN_ABS_Z, "lt")
    }
    cards = []
    for name, (val, thresh, op) in metrics.items():
        is_good = (val > thresh) if op == "gt" else (abs(val) < thresh)
        col = "success" if is_good else "danger"
        cards.append(dbc.Card([dbc.CardBody([html.H6(name), html.H4(f"{val:.2f}", className=f"text-{col}"), html.Small(f"Limit: {thresh}")])], style={"width": "150px"}))
        
    cols = [{"name": i, "id": i} for i in ['decision_ts', 'signal_health_z', 'trendiness_abs', 'z_xs_mean_roll_z']]
    return fig, cards, recent_sig.tail(10).to_dict('records'), cols

# --- SCHEDULER LOGIC (NO CRON REQUIRED) ---
def run_schedule_loop():
    print("[SCHEDULER] Background thread started.")
    last_run = None
    while True:
        now = datetime.now()
        # Trigger at 07:01 AM
        if now.hour == 7 and now.minute == 1:
            today_str = now.strftime("%Y-%m-%d")
            if last_run != today_str:
                print(f"[SCHEDULER] Running Morning Batch for {today_str}...")
                try:
                    eod_process.run_eod_main()
                    last_run = today_str
                except Exception as e:
                    print(f"[SCHEDULER] Error: {e}")
        time.sleep(30) # Check every 30s

# Start the Scheduler Thread
t = threading.Thread(target=run_schedule_loop)
t.daemon = True
t.start()

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
