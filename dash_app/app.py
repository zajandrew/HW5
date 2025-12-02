import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

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
feed.start(log_to_db=False)

def get_latest_model_yymm():
    """Finds most recent _enh.parquet file to load as Midnight Model."""
    enh_dir = Path(cr.PATH_ENH)
    files = list(enh_dir.glob("*_enh.parquet"))
    if not files: return datetime.now().strftime("%y%m")
    files.sort() 
    return files[-1].name.split('_')[0]

CURRENT_YYMM = get_latest_model_yymm()
print(f"[SYSTEM] Initializing App with Model Month: {CURRENT_YYMM}")
le.load_midnight_model(CURRENT_YYMM)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "RV Overlay DSS"

# --- HELPERS ---
def get_z_color(z):
    if z is None: return "grey"
    if z > 2.0: return "#00FF00" # Bright Green
    if z > 0.5: return "#90EE90" # Light Green
    if z < -2.0: return "#FF0000" # Bright Red
    if z < -0.5: return "#CD5C5C" # Indian Red
    return "#FFD700" # Neutral

def format_tenor(ticker, float_years):
    if float_years is None: return "N/A"
    y = float(float_years)
    tol = 0.001
    if abs(y - 1/12) < tol: return "1M"
    if abs(y - 2/12) < tol: return "2M"
    if abs(y - 3/12) < tol: return "3M"
    if abs(y - 4/12) < tol: return "4M"
    if abs(y - 5/12) < tol: return "5M"
    if abs(y - 6/12) < tol: return "6M"
    if abs(y - 7/12) < tol: return "7M"
    if abs(y - 8/12) < tol: return "8M"
    if abs(y - 9/12) < tol: return "9M"
    if abs(y - 10/12) < tol: return "10M"
    if abs(y - 11/12) < tol: return "11M"
    if abs(y - 1.5) < tol: return "18M"
    if abs(y - round(y)) < tol: return f"{int(round(y))}Y"
    return f"{y:.1f}Y"

def assign_bucket(tenor):
    if tenor < 2.0: return "Short (<2Y)"
    if tenor <= 7.0: return "Belly (2-7Y)"
    return "Long (>7Y)"

def aggregate_pnl_columns(df):
    """Sums cash columns for summary."""
    if df.empty: return 0,0,0,0
    return (
        df['realized_pnl_cash'].sum(),
        df['realized_pnl_price'].sum(),
        df['realized_pnl_carry'].sum(),
        df['realized_pnl_roll'].sum()
    )

# --- LAYOUT ---
app.layout = html.Div([
    dcc.Interval(id='fast-interval', interval=2000, n_intervals=0), # 2s updates
    dcc.Interval(id='slow-interval', interval=10000, n_intervals=0), # 10s updates
    
    # Header
    dbc.NavbarSimple(
        children=[
            dbc.Badge("REGIME: SAFE", color="success", className="ms-2", id="badge-regime"),
            dbc.Badge("SHOCK: OFF", color="success", className="ms-2", id="badge-shock"),
            dbc.Button("Run EOD Batch", id="btn-run-eod", size="sm", color="secondary", className="ms-4")
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
                html.H5("Market Grid (Live Z-Scores)", className="mt-3"),
                html.Div(id="ticker-grid", className="d-flex flex-wrap gap-2"),
                html.Div(id="scanner-msg", className="text-danger mt-2")
            ], fluid=True)
        ]),
        
        # TAB 2: BLOTTER
        dbc.Tab(label="Blotter", children=[
            dbc.Container([
                # Open Positions
                dbc.Row([
                    dbc.Col(html.H5("Active Positions (Live PnL Breakdown)"), width=10),
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

        # TAB 3: SUMMARY
        dbc.Tab(label="Summary", children=[
            dbc.Container([
                html.H4("Performance Dashboard", className="mt-4"),
                html.Div(id="summary-cards", className="d-flex gap-3 mb-4 flex-wrap"),
                dbc.Row([
                    dbc.Col([
                        html.H5("Time Horizon Stats"), 
                        html.Div(id="time-stats-container")
                    ], width=6),
                    dbc.Col([
                        html.H5("Risk Profile (Net DV01)"), 
                        html.Div(id="risk-profile-container")
                    ], width=6)
                ])
            ], fluid=True)
        ]),
        
        # TAB 4: HEALTH
        dbc.Tab(label="System Health", children=[
            dbc.Container([
                html.H4("Regime & Signal Diagnostics", className="mt-4"),
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

# 1. Update Ticker Grid (SMART DISABLE)
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
    shock_days = state[2]
    regime = state[3]
    
    is_shocked = (shock_days > 0)
    is_unstable = (regime != 'SAFE')
    global_disable = is_shocked or is_unstable
    
    z_map = le.get_live_z_scores(feed.live_map)
    sorted_tickers = sorted(z_map.keys(), key=lambda t: cr.TENOR_YEARS.get(t, 99))
    
    cards = []
    for t in sorted_tickers:
        data = z_map[t]
        z_val = data['z_live']
        tenor_label = format_tenor(t, data['tenor'])
        color = get_z_color(z_val)
        
        # Smart Disable Logic
        if global_disable:
            can_pay, can_rec = False, False
        else:
            # Check availability of partners
            partners_pay = le.get_valid_partners(t, 'PAY', z_map)
            partners_rec = le.get_valid_partners(t, 'REC', z_map)
            can_pay = len(partners_pay) > 0
            can_rec = len(partners_rec) > 0
        
        pay_style = {'opacity': 0.5 if not can_pay else 1.0}
        rec_style = {'opacity': 0.5 if not can_rec else 1.0}
        
        # Visual cue if feed is dead (0.0)
        if z_val == 0.0 and data['model_z'] != 0.0: color = "#444"

        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"{tenor_label}", className="card-title text-center"),
                html.P(f"Z: {z_val:.2f}", className="text-center small mb-1"),
                html.Div([
                    dbc.Button("PAY", id={'type': 'btn-pay', 'index': t}, 
                               color="success" if can_pay else "secondary", 
                               size="sm", className="me-1", disabled=not can_pay, style=pay_style), 
                    dbc.Button("REC", id={'type': 'btn-rec', 'index': t}, 
                               color="danger" if can_rec else "secondary", 
                               size="sm", disabled=not can_rec, style=rec_style)
                ], className="d-flex justify-content-center")
            ])
        ], style={"width": "120px", "border": f"2px solid {color}"})
        cards.append(card)
        
    regime_text = f"REGIME: {regime}"
    regime_color = "success" if not is_unstable else "danger"
    shock_text = "SHOCK: OFF" if not is_shocked else f"SHOCK: {shock_days} DAYS"
    shock_color = "success" if not is_shocked else "danger"
    
    return cards, regime_text, regime_color, shock_text, shock_color

# 2. Modal Logic
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
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
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
        candidates = le.get_valid_partners(ticker_orig, intent, z_map)
        
        if not candidates:
            return True, html.Div(f"No valid overlay candidates found for {label_orig}.")
            
        current_rate = feed.live_map.get(ticker_orig, 0.0)
        options = []
        for c in candidates:
            lbl = f"{format_tenor(c['ticker'], c['tenor'])} ({c['ticker']}) | Z: {c['z_live']:.2f} | Imp: {c['spread_imp']:.2f}"
            options.append({'label': lbl, 'value': c['ticker']})
        best_ticker = candidates[0]['ticker']
        best_rate = feed.live_map.get(best_ticker, 0.0)
        
        # Switch Logic Text
        if intent == "PAY":
            leg1_label = "LEG 1: PAY (Alternative)"
            leg2_label = f"LEG 2: REC (Original {label_orig})"
        else:
            leg1_label = "LEG 1: REC (Alternative)"
            leg2_label = f"LEG 2: PAY (Original {label_orig})"
        
        content = html.Div([
            html.H5(f"Original Req: {intent} Fixed {label_orig}"),
            html.P("Strategy: Switch exposure to cheaper alternative.", className="text-muted small"),
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

# 3. Submit Trade
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
    else:
        t_pay, t_rec = ticker_orig, ticker_alt
        r_pay, r_rec = r_orig, r_alt
        
    tenor_pay = cr.TENOR_YEARS.get(t_pay)
    tenor_rec = cr.TENOR_YEARS.get(t_rec)
    z_map = le.get_live_z_scores(feed.live_map)
    z_spread = z_map.get(t_pay, {}).get('z_live', 0) - z_map.get(t_rec, {}).get('z_live', 0)
    model_z = z_map.get(t_pay, {}).get('model_z', 0) - z_map.get(t_rec, {}).get('model_z', 0)
    
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
        'model_z_score': model_z
    }
    add_position(trade)
    return f"Trade Executed: Pay {t_pay} / Rec {t_rec}"

# 4. Update Blotter (Both Tables + Breakdown)
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
    now_dt = datetime.now()
    
    # --- PROCESS OPEN TRADES ---
    open_df = all_trades[all_trades['status'] == 'OPEN']
    open_data = []
    
    for _, row in open_df.iterrows():
        # Live Engine Returns: (Cash_Tuple, BP_Tuple)
        (c_tot, c_prc, c_cry, c_rol), (b_tot, b_prc, b_cry, b_rol) = le.calculate_live_pnl(row, feed.live_map, now_dt)
        
        z_map = le.get_live_z_scores(feed.live_map)
        zp = z_map.get(row['ticker_pay'], {}).get('z_live', 0)
        zr = z_map.get(row['ticker_rec'], {}).get('z_live', 0)
        curr_z = zp - zr
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
            
            # PnL Cash (Displayed)
            'total_pnl': round(c_tot, 0),
            'price_pnl': round(c_prc, 0),
            'carry_pnl': round(c_cry, 0),
            'roll_pnl': round(c_rol, 0),
            
            # PnL BP (Hidden for DB pass-through)
            'total_bp': b_tot,
            'price_bp': b_prc,
            'carry_bp': b_cry,
            'roll_bp': b_rol,
            
            'curr_z': round(curr_z, 2),
            'target_z': f"0.0 ± {Z_EXIT}",
            'stop_z': round(stop_z_level, 2),
            'aging': f"{days_held} / {MAX_HOLD}",
            'status': flag,
            '_row_color': bg_color 
        })

    # --- PROCESS CLOSED HISTORY ---
    hist_df = all_trades[all_trades['status'] == 'CLOSED'].sort_values('close_ts', ascending=False)
    hist_data = []
    
    for _, row in hist_df.iterrows():
        label_pay = format_tenor(row['ticker_pay'], row['tenor_pay'])
        label_rec = format_tenor(row['ticker_rec'], row['tenor_rec'])
        hist_data.append({
            'close_ts': row['close_ts'],
            'pair': f"Pay {label_pay} / Rec {label_rec}",
            'dv01': f"{int(row['dv01']/1000)}k",
            'total_pnl': f"${row['realized_pnl_cash']:,.0f}",
            'price_pnl': f"${row.get('realized_pnl_price', 0):,.0f}",
            'carry_pnl': f"${row.get('realized_pnl_carry', 0):,.0f}",
            'roll_pnl': f"${row.get('realized_pnl_roll', 0):,.0f}",
            'reason': row['close_reason']
        })

    # Column Definitions
    open_cols_list = ['trade_id', 'status', 'aging', 'pair', 'dv01', 'total_pnl', 'price_pnl', 'carry_pnl', 'roll_pnl', 'curr_z', 'target_z', 'stop_z']
    open_cols = [{"name": i.replace('_', ' ').title(), "id": i} for i in open_cols_list]
    
    hist_cols_list = ['close_ts', 'pair', 'dv01', 'total_pnl', 'price_pnl', 'carry_pnl', 'roll_pnl', 'reason']
    hist_cols = [{"name": i.replace('_', ' ').title(), "id": i} for i in hist_cols_list]

    style_cond = []
    for i, row in enumerate(open_data):
        if row['_row_color'] != "#222":
            style_cond.append({'if': {'filter_query': f'{{trade_id}} = {row["trade_id"]}'}, 'backgroundColor': row['_row_color'], 'color': 'white'})

    return open_data, open_cols, style_cond, hist_data, hist_cols

# 5. Modify Trade (Save BPs to DB)
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
    trade_id = data[row_idx]['trade_id']
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-delete-trade": delete_position(trade_id)
    elif button_id == "btn-close-trade":
        row = data[row_idx]
        
        # Pass breakdown to DataManager
        c_dict = {'total': row['total_pnl'], 'price': row['price_pnl'], 'carry': row['carry_pnl'], 'roll': row['roll_pnl']}
        b_dict = {'total': row['total_bp'], 'price': row['price_bp'], 'carry': row['carry_bp'], 'roll': row['roll_bp']}
        
        update_position_status(
            trade_id, "CLOSED", reason or "manual", 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            pnl_cash=c_dict, pnl_bp=b_dict
        )
    return "Action Complete"

# 6. EOD Trigger
@app.callback(
    Output('btn-run-eod', 'children'),
    Input('btn-run-eod', 'n_clicks'),
    prevent_initial_call=True
)
def run_eod_batch(n):
    eod_process.run_eod_main()
    return "Batch Completed"

# --- SUMMARY & HEALTH ---
@app.callback(
    Output('summary-cards', 'children'),
    Output('time-stats-container', 'children'),
    Output('risk-profile-container', 'children'),
    Input('slow-interval', 'n_intervals'),
    Input('btn-close-trade', 'n_clicks')
)
def update_summary(n, close_trigger):
    all_trades = get_all_positions()
    if all_trades.empty: return [], "No Data", "No Data"
    
    closed = all_trades[all_trades['status'] == 'CLOSED']
    open_trades = all_trades[all_trades['status'] == 'OPEN']
    
    c_tot, c_prc, c_cry, c_rol = aggregate_pnl_columns(closed)
    
    # Calculate approx live PnL for Open Trades
    o_tot = 0
    now_dt = datetime.now()
    for _, row in open_trades.iterrows():
        (ct,_,_,_),_ = le.calculate_live_pnl(row, feed.live_map, now_dt)
        o_tot += ct
        
    grand_total = c_tot + o_tot
    
    def make_card(title, value, color="primary"):
        return dbc.Card([dbc.CardBody([html.H6(title, className="card-title"), html.H4(value, className=f"text-{color}")])], style={"width": "180px"})

    cards = [
        make_card("Total PnL", f"${grand_total:,.0f}", "success" if grand_total >=0 else "danger"),
        make_card("Realized Price", f"${c_prc:,.0f}", "info"),
        make_card("Realized Carry", f"${c_cry:,.0f}", "warning"),
        make_card("Realized Roll", f"${c_rol:,.0f}", "warning"),
    ]
    
    # Time Horizon Stats
    now = datetime.now()
    all_trades['dt'] = pd.to_datetime(all_trades['open_ts'])
    mask_mtd = (all_trades['dt'].dt.month == now.month) & (all_trades['dt'].dt.year == now.year)
    mask_ytd = (all_trades['dt'].dt.year == now.year)
    
    def calc_stat_pnl(mask):
        sub = all_trades[mask & (all_trades['status']=='CLOSED')]
        return sub['realized_pnl_cash'].sum()

    time_table = dbc.Table.from_dataframe(pd.DataFrame({
        "Period": ["Month to Date", "Year to Date", "All Time"],
        "Realized PnL": [f"${calc_stat_pnl(mask_mtd):,.0f}", f"${calc_stat_pnl(mask_ytd):,.0f}", f"${c_tot:,.0f}"],
        "Trades": [len(all_trades[mask_mtd]), len(all_trades[mask_ytd]), len(all_trades)]
    }), striped=True, bordered=True, dark=True)
    
    # Risk Profile
    bucket_risk = {"Short (<2Y)": 0, "Belly (2-7Y)": 0, "Long (>7Y)": 0}
    for _, row in open_trades.iterrows():
        b_pay = assign_bucket(row['tenor_pay'])
        b_rec = assign_bucket(row['tenor_rec'])
        bucket_risk[b_pay] += row['dv01']
        bucket_risk[b_rec] += row['dv01']
        
    risk_df = pd.DataFrame(list(bucket_risk.items()), columns=["Bucket", "Gross DV01"])
    risk_table = dbc.Table.from_dataframe(risk_df, striped=True, bordered=True, dark=True)
    
    return cards, time_table, risk_table

@app.callback(
    Output('health-cards', 'children'),
    Output('tbl-signal-history', 'data'),
    Output('tbl-signal-history', 'columns'),
    Input('slow-interval', 'n_intervals')
)
def update_health(n):
    signals = hf.get_or_build_hybrid_signals()
    if signals.empty: return [], [], []
    last = signals.iloc[-1]
    
    metrics = {
        "Signal Health (Z)": (last.get('signal_health_z', -99), cr.MIN_SIGNAL_HEALTH_Z, "gt"),
        "Trendiness (Abs)": (last.get('trendiness_abs', 0), cr.MAX_TRENDINESS_ABS, "lt"),
        "Curve Mean (Z)": (last.get('z_xs_mean_roll_z', 0), cr.MAX_Z_XS_MEAN_ABS_Z, "lt"),
        "Curve Vol (Z)": (last.get('z_xs_std_roll_z', 0), 2.0, "lt")
    }
    
    cards = []
    for name, (val, thresh, op) in metrics.items():
        is_good = (val > thresh) if op == "gt" else (abs(val) < thresh)
        col = "success" if is_good else "danger"
        cards.append(dbc.Card([dbc.CardBody([
            html.H6(name, className="card-title"),
            html.H4(f"{val:.2f}", className=f"text-{col}"),
            html.Small(f"Limit: {thresh}", className="text-muted")
        ])], style={"width": "180px"}))
        
    recent = signals.tail(10).sort_values('decision_ts', ascending=False)
    recent['decision_ts'] = recent['decision_ts'].astype(str)
    cols = [{"name": i, "id": i} for i in recent.columns if "z_" in i or "health" in i or "trend" in i or "ts" in i]
    
    return cards, recent.to_dict('records'), cols

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
