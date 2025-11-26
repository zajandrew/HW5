import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Import Local Modules
from data_manager import (init_dbs, add_position, get_open_positions, 
                          get_all_positions, update_position_status, 
                          get_system_state, delete_position)
import live_engine as le
from live_feed import feed  # Real Feed
import eod_process          # EOD Logic
import hybrid_filter as hf  # For loading signals

# Import Research Config
sys.path.append(str(Path(__file__).parent.parent))
import cr_config as cr

# --- SETUP ---
init_dbs()
feed.start()

def get_latest_model_yymm():
    """Scans PATH_ENH for the most recent parquet file."""
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
    if z > 2.0: return "#00FF00" 
    if z > 0.5: return "#90EE90" 
    if z < -2.0: return "#FF0000" 
    if z < -0.5: return "#CD5C5C" 
    return "#FFD700" 

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

# --- LAYOUT ---
app.layout = html.Div([
    dcc.Interval(id='fast-interval', interval=2000, n_intervals=0), # 2s updates
    dcc.Interval(id='slow-interval', interval=10000, n_intervals=0), # 10s updates (Health/Summary)
    
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
                dbc.Row([
                    dbc.Col(html.H5("Open Positions"), width=10),
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
                ], className="mt-2")
            ], fluid=True)
        ]),

        # TAB 3: SUMMARY (POPULATED)
        dbc.Tab(label="Summary", children=[
            dbc.Container([
                html.H4("Performance Summary", className="mt-4"),
                html.Div(id="summary-cards", className="d-flex gap-3 mb-4"),
                html.Hr(),
                html.H5("Closed Trades Ledger"),
                dash_table.DataTable(
                    id='tbl-closed-positions',
                    style_table={'overflowX': 'auto', 'maxHeight': '400px', 'overflowY': 'scroll'},
                    style_cell={'backgroundColor': '#222', 'color': 'white'},
                    page_size=10
                )
            ], fluid=True)
        ]),
        
        # TAB 4: HEALTH (POPULATED)
        dbc.Tab(label="System Health", children=[
            dbc.Container([
                html.H4("Regime & Signal Diagnostics", className="mt-4"),
                html.Div(id="health-cards", className="d-flex gap-3 mb-4"),
                html.H5("Last 10 Days Signal History"),
                dash_table.DataTable(
                    id='tbl-signal-history',
                    style_table={'overflowX': 'auto'},
                    style_cell={'backgroundColor': '#222', 'color': 'white'},
                )
            ], fluid=True)
        ])
    ])
])

# --- CALLBACKS ---

# 1. Update Ticker Grid & Badges
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
    trading_disabled = is_shocked or is_unstable
    
    z_map = le.get_live_z_scores(feed.live_map)
    sorted_tickers = sorted(z_map.keys(), key=lambda t: cr.TENOR_YEARS.get(t, 99))
    
    cards = []
    for t in sorted_tickers:
        data = z_map[t]
        z_val = data['z_live']
        tenor_label = format_tenor(t, data['tenor'])
        color = get_z_color(z_val)
        btn_style = {'opacity': 0.5} if trading_disabled else {}
        
        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"{tenor_label}", className="card-title text-center"),
                html.P(f"Z: {z_val:.2f}", className="text-center small mb-1"),
                html.Div([
                    dbc.Button("PAY", id={'type': 'btn-pay', 'index': t}, 
                               color="success" if not trading_disabled else "secondary", 
                               size="sm", className="me-1", disabled=trading_disabled, style=btn_style), 
                    dbc.Button("REC", id={'type': 'btn-rec', 'index': t}, 
                               color="danger" if not trading_disabled else "secondary", 
                               size="sm", disabled=trading_disabled, style=btn_style)
                ], className="d-flex justify-content-center")
            ])
        ], style={"width": "120px", "border": f"2px solid {color}"})
        cards.append(card)
        
    regime_text = f"REGIME: {regime}"
    regime_color = "success" if not is_unstable else "danger"
    shock_text = "SHOCK: OFF" if not is_shocked else f"SHOCK: {shock_days} DAYS"
    shock_color = "success" if not is_shocked else "danger"
    
    return cards, regime_text, regime_color, shock_text, shock_color

# 2. Modal Logic (Unchanged)
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

# 3. Submit Trade Logic (Unchanged)
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

# 4. Update Blotter
@app.callback(
    Output('tbl-open-positions', 'data'),
    Output('tbl-open-positions', 'columns'),
    Input('fast-interval', 'n_intervals'),
    Input('btn-refresh-blotter', 'n_clicks')
)
def update_blotter(n, click):
    df = get_open_positions()
    if df.empty: return [], []
    data = []
    Z_EXIT, Z_STOP = cr.Z_EXIT, cr.Z_STOP
    
    for _, row in df.iterrows():
        curr_pay = feed.live_map.get(row['ticker_pay'], row['entry_rate_pay'])
        curr_rec = feed.live_map.get(row['ticker_rec'], row['entry_rate_rec'])
        pnl_pay = (curr_pay - row['entry_rate_pay']) * 100 * row['dv01']
        pnl_rec = (row['entry_rate_rec'] - curr_rec) * 100 * row['dv01']
        total_pnl = pnl_pay + pnl_rec
        
        z_map = le.get_live_z_scores(feed.live_map)
        zp = z_map.get(row['ticker_pay'], {}).get('z_live', 0)
        zr = z_map.get(row['ticker_rec'], {}).get('z_live', 0)
        curr_z = zp - zr
        
        db_reason = row.get('close_reason')
        if db_reason and db_reason not in [None, 'None', '']:
             if db_reason == 'max_hold': flag = "MAX HOLD"
             elif db_reason == 'stop_loss': flag = "STOP LOSS (EOD)"
             elif db_reason == 'reversion': flag = "TAKE PROFIT (EOD)"
             else: flag = f"CHECK: {db_reason}"
        else:
             flag = "OPEN"
             if abs(curr_z) < Z_EXIT: flag = "TAKE PROFIT"
             if abs(curr_z - row['entry_z_spread']) > Z_STOP: flag = "STOP LOSS"
        
        label_pay = format_tenor(row['ticker_pay'], row['tenor_pay'])
        label_rec = format_tenor(row['ticker_rec'], row['tenor_rec'])
        
        data.append({
            'trade_id': row['trade_id'],
            'open_ts': row['open_ts'],
            'pair': f"Pay {label_pay} / Rec {label_rec}",
            'dv01': row['dv01'],
            'pnl_cash': round(total_pnl, 2),
            'entry_z': round(row['entry_z_spread'], 2),
            'curr_z': round(curr_z, 2),
            'status': flag
        })
    cols = [{"name": i, "id": i} for i in data[0].keys()]
    return data, cols

# 5. Buttons Enable/Action (Unchanged)
@app.callback(
    Output('btn-close-trade', 'disabled'),
    Output('btn-delete-trade', 'disabled'),
    Input('tbl-open-positions', 'selected_rows'),
    prevent_initial_call=True
)
def enable_buttons(rows): return not rows, not rows

@app.callback(
    Output('summary-content', 'children'), # Dummy output
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
        pnl = data[row_idx]['pnl_cash']
        update_position_status(trade_id, "CLOSED", reason or "manual", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pnl)
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

# --- NEW: SUMMARY & HEALTH TABS ---

@app.callback(
    Output('summary-cards', 'children'),
    Output('tbl-closed-positions', 'data'),
    Output('tbl-closed-positions', 'columns'),
    Input('slow-interval', 'n_intervals'),
    Input('btn-close-trade', 'n_clicks') # Update when we close a trade
)
def update_summary(n, close_trigger):
    # 1. Get Closed Trades
    all_trades = get_all_positions()
    if all_trades.empty: return [], [], []
    
    closed = all_trades[all_trades['status'] == 'CLOSED']
    open_trades = all_trades[all_trades['status'] == 'OPEN']
    
    realized_pnl = closed['realized_pnl_cash'].sum()
    
    # Approx Unrealized PnL (Recalculating briefly here)
    unrealized_pnl = 0.0
    for _, row in open_trades.iterrows():
        curr_pay = feed.live_map.get(row['ticker_pay'], row['entry_rate_pay'])
        curr_rec = feed.live_map.get(row['ticker_rec'], row['entry_rate_rec'])
        pnl_pay = (curr_pay - row['entry_rate_pay']) * 100 * row['dv01']
        pnl_rec = (row['entry_rate_rec'] - curr_rec) * 100 * row['dv01']
        unrealized_pnl += (pnl_pay + pnl_rec)
        
    total_pnl = realized_pnl + unrealized_pnl
    win_rate = (len(closed[closed['realized_pnl_cash'] > 0]) / len(closed) * 100) if not closed.empty else 0.0
    
    def make_card(title, value, color="primary"):
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="card-title"),
                html.H4(value, className=f"text-{color}")
            ])
        ], style={"width": "180px"})

    cards = [
        make_card("Total PnL", f"${total_pnl:,.0f}", "success" if total_pnl >=0 else "danger"),
        make_card("Realized PnL", f"${realized_pnl:,.0f}", "success" if realized_pnl >=0 else "danger"),
        make_card("Unrealized PnL", f"${unrealized_pnl:,.0f}", "info"),
        make_card("Win Rate", f"{win_rate:.1f}%", "warning"),
        make_card("Total Trades", f"{len(all_trades)}", "secondary")
    ]
    
    closed_data = closed.to_dict('records')
    closed_cols = [{"name": i, "id": i} for i in closed.columns]
    
    return cards, closed_data, closed_cols

@app.callback(
    Output('health-cards', 'children'),
    Output('tbl-signal-history', 'data'),
    Output('tbl-signal-history', 'columns'),
    Input('slow-interval', 'n_intervals')
)
def update_health(n):
    # Load Signals Parquet
    signals = hf.get_or_build_hybrid_signals()
    if signals.empty: return [], [], []
    
    last = signals.iloc[-1]
    
    # Metrics
    health_z = last.get('signal_health_z', -99)
    trendiness = last.get('trendiness_abs', 0)
    xs_mean_z = last.get('z_xs_mean_roll_z', 0)
    
    def make_card(title, value, threshold_desc, status_color):
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="card-title"),
                html.H4(f"{value:.2f}", className=f"text-{status_color}"),
                html.Small(threshold_desc, className="text-muted")
            ])
        ], style={"width": "220px"})

    # Logic colors
    health_col = "success" if health_z > cr.MIN_SIGNAL_HEALTH_Z else "danger"
    trend_col = "success" if trendiness < cr.MAX_TRENDINESS_ABS else "danger"
    
    cards = [
        make_card("Signal Health (Z)", health_z, f"Threshold > {cr.MIN_SIGNAL_HEALTH_Z}", health_col),
        make_card("Trendiness (Abs)", trendiness, f"Threshold < {cr.MAX_TRENDINESS_ABS}", trend_col),
        make_card("Curve Mean (Z)", xs_mean_z, f"Threshold < {cr.MAX_Z_XS_MEAN_ABS_Z}", "info")
    ]
    
    # Recent History Table
    recent = signals.tail(10).sort_values('decision_ts', ascending=False)
    recent['decision_ts'] = recent['decision_ts'].astype(str) # Serialize date
    
    # Filter columns for display
    disp_cols = ['decision_ts', 'signal_health_z', 'trendiness_abs', 'z_xs_mean_roll_z']
    data = recent[disp_cols].to_dict('records')
    cols = [{"name": i, "id": i} for i in disp_cols]
    
    return cards, data, cols

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
