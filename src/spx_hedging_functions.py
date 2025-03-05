from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import norm

from settings import config

pio.templates.default = "plotly_white"

# Load environment variables
DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def build_delta_map(
    option_chain,
    asof_date,
    contracts_to_display,
    show_detail=False,
    weighted=False,
    interpolate_missing=True,
    interpolation_args=None,
):
    if weighted:
        option_chain["delta"] = (
            option_chain["delta"]
            * option_chain["volume"]
            / option_chain["volume"].sum()
        )

    option_chain["expiration_date"] = pd.to_datetime(option_chain["expiration_date"])
    option_chain["days_to_expiration"] = (
        option_chain["expiration_date"] - pd.to_datetime(asof_date)
    ).dt.days
    option_chain = option_chain.set_index(
        ["strike", "expiration_date", "type"]
    ).sort_index()
    # option_chain

    strikes = option_chain.index.get_level_values("strike").unique()
    expiration_dates = (
        option_chain.index.get_level_values("expiration_date").unique().sort_values()
    )

    deltas = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [strikes, expiration_dates, ["Call", "Put"]],
            names=["strike", "expiration_date", "type"],
        ),
        columns=["delta"],
    )
    deltas.update(option_chain["delta"])
    deltas = deltas.unstack(level="type")
    # updated put deltas by adding 1
    deltas.loc[pd.IndexSlice[:, :], ("delta", "Put")] = (
        deltas.loc[pd.IndexSlice[:, :], ("delta", "Put")] + 1
    )
    # replace zeros and 1s with nan
    deltas = deltas.replace(0.0, np.nan).replace(1.0, np.nan)
    deltas[("delta", "mean")] = deltas.mean(axis=1)
    deltas.columns = deltas.columns.droplevel(0)
    deltas = deltas.astype("float64")
    deltas

    delta_map = deltas.pivot_table(
        index="strike", columns="expiration_date", values="mean"
    ).dropna(thresh=6, axis=0)
    # reverse index
    delta_map = delta_map.iloc[::-1]

    # interpolate missing values down the columns
    if interpolate_missing:
        non_interpolated_columns = 0

        while non_interpolated_columns < len(delta_map.columns):
            try:
                delta_map = (
                    delta_map.iloc[:, : contracts_to_display - non_interpolated_columns]
                    .interpolate(
                        method=interpolation_args["method"],
                        order=interpolation_args["order"],
                        axis=0,
                    )
                    .join(
                        delta_map.iloc[
                            :, contracts_to_display - non_interpolated_columns :
                        ]
                    )
                    .ffill(axis=1)
                )
                break  # Exit loop if no ValueError occurs
            except ValueError:
                non_interpolated_columns += 1

        # If the loop exits without successfully interpolating, handle it (optional)
        if non_interpolated_columns == len(delta_map.columns):
            raise ValueError("Interpolation failed for all column configurations.")

    if show_detail:
        # Remove the styling with highlight_closest_to_targets
        display(
            delta_map.style.format(
                "{:.1%}"
            )
        )

    return delta_map


def build_prediction_range(option_chain, asof_date, delta_map):
    # Initialize the `prediction_range` DataFrame with specified rows and the same columns as `delta_map`
    row_labels = ["80% Below", "50/50", "80% Above"]
    prediction_range = pd.DataFrame(index=row_labels, columns=delta_map.columns)

    # Define the target values and colors for reference
    targets = {0.80: "red", 0.50: "grey", 0.20: "green"}

    # drop the columns with many nan
    # delta_map = delta_map.dropna(thresh=contracts_to_display, axis=1)

    # Populate prediction_range based on the closest highlighted cells in delta_map
    for col in delta_map.columns:
        # Extract indices of the highlighted cells for each target value
        closest_to_80_idx = (delta_map[col] - 0.80).abs().dropna().idxmin()
        closest_to_50_idx = (delta_map[col] - 0.50).abs().dropna().idxmin()
        closest_to_20_idx = (delta_map[col] - 0.20).abs().dropna().idxmin()

        # Assign strike prices to prediction_range based on the target values
        prediction_range.loc["80% Above", col] = closest_to_80_idx
        prediction_range.loc["50/50", col] = closest_to_50_idx
        prediction_range.loc["80% Below", col] = closest_to_20_idx

    # Display the prediction_range DataFrame
    prediction_range = prediction_range.T
    # update the index to be the month start
    prediction_range.index = (
        pd.to_datetime(prediction_range.index).to_period("D").to_timestamp()
    )

    # add the term structure of forward prices
    fwd_prices_name = f"Fwd Prices {asof_date}"
    fwd_prices = option_chain.loc[:, ["expiration_date", "forwardprice"]].set_index(
        "expiration_date"
    )
    fwd_prices = fwd_prices.loc[~fwd_prices.index.duplicated(keep="last")]
    # add a column to the prediction_range
    prediction_range[fwd_prices_name] = fwd_prices.loc[:, "forwardprice"]

    prediction_range = prediction_range.astype("float64")
    prediction_range = prediction_range[~prediction_range.index.duplicated(keep="last")]
    prediction_range[fwd_prices_name] = prediction_range[fwd_prices_name].fillna(
        prediction_range["50/50"]
    )
    return prediction_range


def get_default_plot_layout(width=1200, height=600):
    """Returns default plot layout settings used across multiple functions"""
    return dict(
        width=width,
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )


def update_payoff_axes(
    fig, num_subplots=1, x_title="Brent Oil Price ($/Share)", x_range=[60, 100]
):
    """Updates axes with common payoff chart settings"""
    for i in range(1, num_subplots + 1):
        fig.update_yaxes(
            title_text="Payoff ($/Share)",
            range=[-15, 15],
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=0.8,
            row=1,
            col=i,
        )
        fig.update_xaxes(title_text=x_title, range=x_range, row=1, col=i)


def add_weekend_rangebreaks(fig):
    """Adds standard weekend breaks to plot x-axis"""
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(values=["2015-12-25", "2016-01-01"]),
        ]
    )


def get_annotation_location(trace_name):
    """Centralizes the logic for determining annotation locations"""
    x_loc = 0
    y_loc = 20

    if "Sim" not in trace_name:
        if "Below" in trace_name:
            y_loc = -35
        elif "Ceiling" in trace_name:
            y_loc = 15
            x_loc = 20
        elif "Above" in trace_name:
            y_loc = 35
    elif "Sim" in trace_name:
        if "Below" in trace_name:
            y_loc = -15
            x_loc = 20
        elif "Above" in trace_name:
            y_loc = 15
            x_loc = 20

    return {"x": x_loc, "y": y_loc}


def build_market_predictions(option_chain, asof_date, contracts_to_display, **kwargs):
    """
    Builds core market prediction data structures from option chain data.
    Returns delta map and prediction range for further analysis.
    """
    delta_map = build_delta_map(option_chain, asof_date, contracts_to_display, **kwargs)
    prediction_range = build_prediction_range(option_chain, asof_date, delta_map)
    return delta_map, prediction_range


def build_pareto_chart(ticker, asof_date, prediction_range, delta_map, contracts_to_display):
    """
    Creates a pareto chart visualization of market predictions.
    Returns the figure and formatted prediction range data for external display.
    """
    # Create pareto chart
    fig = px.line(
        prediction_range.head(contracts_to_display),
        markers=True,
        title=f"Market Implied Pareto Chart: {ticker} (as of {asof_date})",
        labels={"value": "Price", "expiration_date": "Expiration Date", "variable": ""},
    )

    fig.update_layout(
        **get_default_plot_layout(),
        xaxis=dict(dtick="D"),
        yaxis=dict(tickformat="$.2f"),
    )

    # Add annotations to each trace
    for trace in fig.data:
        if trace.name != "50/50":
            for i, value in enumerate(trace.y):
                fig.add_annotation(
                    x=trace.x[i],
                    y=value,
                    text=f"${value:.2f}",
                    showarrow=True,
                    arrowhead=0,
                    ax=0,
                    ay=[-20 if trace.name == "80% Below" else 20][0],
                )

    add_weekend_rangebreaks(fig)

    # Prepare prediction range table data
    prediction_range_reformat = prediction_range.copy()
    prediction_range_reformat.index = prediction_range_reformat.index.strftime("%Y-%m-%d")
    prediction_range_reformat.index.name = "Expiration Date"
    prediction_range_reformat = prediction_range_reformat.T
    prediction_range_reformat = prediction_range_reformat.iloc[:, :contracts_to_display]
    
    return {
        'figure': fig,
        'prediction_range': prediction_range_reformat
    }


def plot_delta_heatmap(ticker, asof_date, delta_map, contracts_to_display, heatmap_step):
    """
    Creates and displays a heatmap visualization of the delta map.
    """
    return build_delta_map_heatmap(
        ticker,
        asof_date, 
        delta_map,
        contracts_to_display,
        heatmap_step,
        dropna_threshold=contracts_to_display,
    )


def build_delta_map_heatmap(
    ticker,
    asof_date,
    delta_map,
    contracts_to_display=15,
    heatmap_step=3,
    dropna_threshold=5,
):
    # manage display size of the delta map
    # Prepare the delta map for display
    delta_map_reformat = delta_map.copy()
    delta_map_reformat.columns = delta_map_reformat.columns.strftime("%Y-%m-%d")
    delta_map_reformat = delta_map_reformat[
        (0.005 <= delta_map_reformat) & (delta_map_reformat <= 0.995)
    ].iloc[::heatmap_step, :contracts_to_display]
    delta_map_reformat.columns.name = "Expiration Date"
    delta_map_reformat.index.name = "Price >="
    delta_map_reformat.index = delta_map_reformat.index.map("${:,.2f}".format)

    # Filter rows and columns based on the commodity type
    result = delta_map_reformat.iloc[:, :]

    # Drop rows that are completely blank (all NaN)
    result = result.dropna(axis=0, thresh=dropna_threshold)

    # Apply background gradient styling to the numeric data
    styled_result = result.style.background_gradient(cmap="RdYlGn_r", axis=None)

    # Format numeric values and handle NaN separately
    def custom_formatter(val):
        if pd.isna(val):  # Handle NaN values
            return ""  # Return empty string for NaN
        return f"{val:.0%}"  # Format numeric values as percentages

    # Apply the custom formatter without modifying the original data
    styled_result = styled_result.format(custom_formatter)

    # Display the styled DataFrame
    display(styled_result)
    return styled_result


def simulate_prices(days, paths, S, r, sigma, dt):
    # Generate random normal values (Z) for each day and path
    rand_z = np.random.normal(0, 1, (days, paths))

    # Initialize the prices array
    prices = np.zeros((days, paths))
    prices[0, :] = S  # Set initial price at day 1 for all paths

    # Calculate prices using the GBM formula
    for day in range(1, days):
        drift = (r - 0.5 * sigma**2) * day * dt
        diffusion = sigma * np.sqrt(day * dt) * rand_z[day]
        prices[day] = S * np.exp(drift + diffusion)

    return prices


def get_fwd_prices(asof_date, option_chain, contracts_to_display):
    fwd_prices = option_chain.reset_index()[
        ["expiration_date", "forwardprice"]
    ].set_index(["expiration_date"])  # , 'forwardprice'])
    fwd_prices = fwd_prices[~fwd_prices.index.duplicated(keep="first")]
    fwd_prices = fwd_prices.sort_index().iloc[:contracts_to_display, :]
    # display(fwd_prices.style.set_caption(f"Forward prices as of {asof_date}").format("{:.2f}"))
    return fwd_prices


def monte_carlo_futures_simulation(
    strip_prices, iv_surface, asof_date, r=0.03, N=1000, dt=1/252, random_seed=123
):
    """
    Monte Carlo simulation for futures prices using a complete IV surface.

    Args:
        strip_prices (pd.Series): Current futures strip prices, indexed by time_to_expiry.
        iv_surface (pd.DataFrame): Fully interpolated IV surface (index=strikes, columns=time_to_expiry).
        r (float): Risk-free rate (default 0.03).
        N (int): Number of Monte Carlo paths (default 10000).
        dt (float): Time step in years (default daily granularity, 1/252).

    Returns:
        dict: Dictionary with time_to_expiry as keys and simulated price paths as values (2D arrays).
    """

    # Initialize results dictionary
    results = {}

    # Convert dates to time remaining
    strip_prices.index = dates_to_time_remaining(strip_prices.index, asof_date)

    # only use valid expiries in the IV surface (non-NaN)
    strip_prices = strip_prices.loc[iv_surface.columns, :]

    # Loop through each contract
    for T, F_0 in zip(strip_prices.index, strip_prices.values):
        # Find the nearest strike for F_0 in the IV surface
        nearest_strike = iv_surface.index[np.abs(iv_surface.index - F_0[0]).argmin()]
        volatility_curve = iv_surface.loc[nearest_strike]

        # Determine number of steps
        steps = np.maximum(int(T / dt), 1)

        # Generate Brownian motion
        np.random.seed(random_seed)  # Reproducibility
        dW = np.random.normal(scale=np.sqrt(dt), size=(steps, N))

        # Initialize price paths
        F_paths = np.zeros((steps, N))
        F_paths[0, :] = F_0

        # Simulate paths using GBM
        for t in range(1, steps):
            current_time = t * dt

            # sigma = np.interp(current_time, time_to_expiry, volatility_curve)

            sigma = (
                iv_surface.iloc[
                    iv_surface.index.get_indexer(F_paths[t - 1, :], method="nearest")
                ]
                .ffill()
                .bfill()[T]
                .values
            )
            # print(np.any(np.isnan(sigma)))
            decrement = 1
            while np.any(np.isnan(sigma)) and decrement < t:
                # print(f'Found NaNs in sigma at time {t}. Filling with nearest values from time {t-decrement}.')
                sigma = (
                    iv_surface.iloc[
                        iv_surface.index.get_indexer(
                            F_paths[t - 1 - decrement, :], method="nearest"
                        )
                    ]
                    .ffill()
                    .bfill()[T]
                    .values
                )
                decrement += 1

            F_paths[t, :] = F_paths[t - 1, :] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * dW[t, :]
            )
            # print(f'T: {T}, t: {t}, sigma: {sigma}\n F_paths[t-1, :]: {F_paths[t-1, :]}\n F_paths[t,:]: {F_paths[t,:]}')

        # Store results
        results[T] = F_paths

    return results


def simulate_futures(
    asof_date,
    fwd_prices,
    iv_surface,
    r=0.045,
    N=1000,
    dt=1/365.0,
    default_percentiles=np.arange(0.1, 1.0, 0.1),
    random_seed=123,
):
    """
    Simulate futures prices using Monte Carlo simulation.
    Returns the simulated futures data and visualization elements without displaying.
    """
    mcs_results = monte_carlo_futures_simulation(
        fwd_prices, iv_surface, asof_date, r, N, dt, random_seed
    )

    string_percentiles = list(map(lambda x: f"{x:.0%}", default_percentiles))

    simulated_futures = pd.DataFrame(
        index=mcs_results.keys(),
        columns=list(string_percentiles)
        + [f"Strip {pd.to_datetime(asof_date):%Y-%m-%d}"],
    )

    for contract in simulated_futures.index:
        simulation_results = (
            pd.DataFrame(mcs_results[contract][-1])
            .describe(percentiles=default_percentiles)
            .iloc[:, 0]
        )
        simulated_futures.loc[contract].update(simulation_results)
        
    # update with the current strip prices
    simulated_futures[f"Strip {pd.to_datetime(asof_date):%Y-%m-%d}"] = fwd_prices
    simulated_futures = simulated_futures.astype("float64")

    # Prepare display data
    display_data = simulated_futures.copy()
    display_data.index = time_remaining_to_dates(
        display_data.index, asof_date
    ).strftime("%Y-%m-%d")

    # Create figure
    fig = px.line(
        display_data,
        title="Monte Carlo Simulation of Futures Prices",
        labels={
            "value": "Price ($/Share)",
            "index": "Expiration Date",
            "variable": "",
        },
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    
    fig.update_layout(
        **get_default_plot_layout(),
        yaxis=dict(tickformat="$.2f"),
    )

    # Add annotations to each trace
    for trace in fig.data:
        if not (any(_ in trace.name for _ in ["%"])):
            for i, value in enumerate(trace.y):
                annot_loc = get_annotation_location(trace.name)
                fig.add_annotation(
                    x=trace.x[i],
                    y=value,
                    text=f"${value:.2f}",
                    showarrow=True,
                    arrowhead=0,
                    ay=annot_loc["y"],
                    ax=annot_loc["x"],
                )

    add_weekend_rangebreaks(fig)

    return {
        'simulated_futures': simulated_futures,
        'display_data': display_data,
        'figure': fig
    }


def calc_costless_collar(
    collar_strikes, option_chain, d, contracts_to_display, asof_date
):
    """
    Calculate the costless collar for given collar strikes and options data.
    Parameters:
    collar_strikes (pd.DataFrame): DataFrame containing the collar strikes information.
    option_chain (pd.DataFrame): DataFrame containing the options data.
    d (float): The percentage above the current price to determine the floor strikes.
    t (str): The time period for which the options data is considered.
    months_to_display (int): The number of months to display in the options data.
    Returns:
    pd.DataFrame: DataFrame containing the calculated costless collar strikes and their costs.
        Columns include:
        - 'floor_strike': The strike price of the floor option.
        - 'floor_px': The price of the floor option.
        - 'ceiling_strike': The strike price of the ceiling option.
        - 'ceiling_px': The price of the ceiling option.
        - 'net_cost': The net cost of the collar (floor price - ceiling price).
    """

    # show the costless ceiling for this choice of floor
    # from the option chain, get the call options that have the same cost as the provided floor

    options_data = option_chain.copy()
    options_data["expiration_date"] = dates_to_time_remaining(
        options_data[["expiration_date"]].values.squeeze(), asof_date
    )
    options_data.set_index(["expiration_date", "type", "strike"], inplace=True)

    put_prices = options_data.loc[pd.IndexSlice[:, "Put", :], "midprice"].to_frame()
    put_prices.index = put_prices.index.droplevel(1)
    put_prices = put_prices.unstack(level=0)
    put_prices.columns = put_prices.columns.droplevel(0)
    put_prices = put_prices.iloc[:, :]

    floor_strikes = collar_strikes[f"{d:.0%} Above"]

    # extract one entry per expiry
    put_prices = np.diag(
        put_prices.iloc[
            put_prices.index.get_indexer(floor_strikes.values, method="nearest")
        ]
    )

    # build the floor prices and strikes series
    floor_prices = pd.Series(put_prices).interpolate(method="cubic").ffill()
    floor_prices.index = floor_strikes.index

    # get the ceiling prices and strikes
    ceiling_prices = pd.Series().reindex_like(floor_prices)
    ceiling_strikes = pd.Series().reindex_like(floor_prices)
    call_prices = options_data.loc[pd.IndexSlice[:, "Call", :], "midprice"].to_frame()
    call_prices.index = call_prices.index.droplevel(1)
    call_prices = call_prices.unstack(level=0)
    call_prices.columns = call_prices.columns.droplevel(0)
    call_prices = call_prices.iloc[:, :]

    costless_collar = pd.DataFrame(
        columns=["floor_strike", "floor_px", "ceiling_strike", "ceiling_px", "net_cost"]
    )

    for i, (t, floor_px) in enumerate(zip(floor_prices.index, floor_prices.values)):
        ceiling_strike = np.abs(call_prices.loc[:, t] - floor_px).idxmin()
        ceiling_price = call_prices.loc[ceiling_strike, t]
        # print(f'floor_strike: {floor_strikes.iloc[i]}, floor_px: {floor_px}\nceiling_strike: {ceiling_strike}, ceiling_price: {ceiling_price}')
        ceiling_prices.loc[floor_prices.index[i]] = ceiling_price
        ceiling_strikes.loc[floor_prices.index[i]] = ceiling_strike
        costless_collar.loc[floor_prices.index[i]] = [
            floor_strikes.iloc[i],
            floor_px,
            ceiling_strike,
            ceiling_price,
            floor_px - ceiling_price,
        ]

    costless_collar.index = collar_strikes.index

    collar_strikes = pd.concat(
        [collar_strikes, costless_collar.loc[:, "ceiling_strike"]], axis=1
    )
    collar_strikes.rename(columns={"ceiling_strike": "Costless Ceiling"}, inplace=True)
    collar_strikes = collar_strikes.astype("float64")
    collar_strikes.index = time_remaining_to_dates(collar_strikes.index, asof_date)

    return costless_collar, collar_strikes

def build_collar_strikes(
    d, delta_map, simulated_futures, contracts_to_display, asof_date, option_chain
):
    """
    Build collar strikes data and visualization without displaying.
    Returns the data and figure for external display.
    """
    # extract the columns for the simulated futures for this choice of [d, 1-d]
    sim_d = simulated_futures.loc[
        :,
        [
            f"{d:.0%}",
            f"{1 - d:.0%}",
            simulated_futures.columns[simulated_futures.columns.get_indexer(["Strip"])][0],
        ],
    ]

    collar_strikes = pd.DataFrame(columns=[f"{d:.0%} Above", f"{d:.0%} Below"])
    collar_strikes[f"{d:.0%} Below"] = np.abs(delta_map - 1 + d).idxmin()
    collar_strikes[f"{d:.0%} Above"] = np.abs(delta_map - d).idxmin()

    collar_strikes = pd.concat([collar_strikes, sim_d], axis=1)
    collar_strikes.rename(
        columns={
            f"{d:.0%}": f"Sim. {d:.0%} Below",
            f"{1 - d:.0%}": f"Sim. {d:.0%} Above",
        },
        inplace=True,
    )

    costless_collar, collar_strikes = calc_costless_collar(
        collar_strikes, option_chain, d, contracts_to_display, asof_date
    )

    fig = px.line(
        collar_strikes,
        title=f"{d * 100:.0f} / {(1 - d) * 100:.0f} Collar Strikes",
        labels={"value": "Price ($/Share)", "index": "Expiration Date", "variable": ""},
    )

    fig.update_layout(
        **get_default_plot_layout(),
        xaxis=dict(
            range=[
                collar_strikes.index[:contracts_to_display].min(),
                collar_strikes.index[:contracts_to_display].max(),
            ]
        ),
        yaxis=dict(
            tickformat="$.2f",
            range=[
                collar_strikes.iloc[:contracts_to_display, :].min().min() - 50,
                collar_strikes.iloc[:contracts_to_display, :].max().max() + 50,
            ],
        ),
    )

    # Add annotations to each trace
    for trace in fig.data:
        if not (any(_ in trace.name for _ in ["50/50", "Sim"])):
            for i, value in enumerate(trace.y):
                annot_loc = get_annotation_location(trace.name)
                fig.add_annotation(
                    x=trace.x[i],
                    y=value,
                    text=f"${value:.2f}",
                    showarrow=True,
                    arrowhead=0,
                    ay=annot_loc["y"],
                    ax=annot_loc["x"],
                )

    add_weekend_rangebreaks(fig)

    return {
        'collar_strikes': collar_strikes,
        'costless_collar': costless_collar,
        'figure': fig,
        'net_cost': costless_collar['net_cost'].sum()
    }


def build_put_iv_surface(option_chain, fwd_prices, asof_date):
    # extract the OTM options for the IV surface. We use OTM options due to higher liquidity, lower bid-ask spread, and less early exercise risk, which complicates the IV calculation. NTM options (near-the-money, within 5% - 10% of the underlying) are prioritized to anchor the vol surface

    options_data = option_chain.copy()
    options_data["expiration_date"] = dates_to_time_remaining(
        options_data[["expiration_date"]].values.squeeze(), asof_date
    )
    options_data.set_index(["expiration_date", "type", "strike"], inplace=True)

    puts = options_data.loc[pd.IndexSlice[:, "Put", :], "implied_volatility"].to_frame()
    puts.index = puts.index.droplevel(1)
    puts = puts.unstack(level=0)
    puts.columns = puts.columns.droplevel(0)

    contracts_to_display = fwd_prices.shape[0]
    puts = puts.iloc[:, :contracts_to_display]
    puts = puts.sort_index()

    # we find that the IVs are not complete in the dataset, so we filter by options with available prices
    put_prices = options_data.loc[pd.IndexSlice[:, "Put", :], "midprice"].to_frame()
    put_prices.index = put_prices.index.droplevel(1)
    put_prices = put_prices.unstack(level=0)
    put_prices.columns = put_prices.columns.droplevel(0)

    contracts_to_display = fwd_prices.shape[0]
    put_prices = put_prices.iloc[:, :contracts_to_display]
    put_prices = put_prices.sort_index()

    puts = puts.loc[put_prices.index, :].dropna(how="all", axis=1)

    # we want to filter out only the OTM puts
    # we can do this by eliminating strikes above the strip, for a given time to expiry
    otm_mask = pd.DataFrame(index=puts.index, columns=fwd_prices.values.squeeze())
    for c in otm_mask.columns:
        otm_mask[c] = otm_mask.index <= c

    puts = puts.loc[puts.index <= otm_mask.sum(axis=0).idxmax(), :].iloc[
        :, :contracts_to_display
    ]

    # drop duplicate rows
    puts = puts.loc[~puts.index.duplicated(keep="first")]

    # Assume `puts` has index as strikes and columns as time_to_expiry
    # Flatten the DataFrame to get known points and values
    strikes = puts.index.values
    maturities = puts.columns.values

    puts

    # Create a 2D array of coordinates (strike, maturity) for known IV values
    known_points = np.array(
        [
            (strike, maturity)
            for strike in strikes
            for maturity in maturities
            if not np.isnan(puts.loc[strike, maturity])
        ]
    )
    known_points

    # Extract corresponding IV values
    known_values = np.array(
        [
            puts.loc[strike, maturity]
            for strike in strikes
            for maturity in maturities
            if not np.isnan(puts.loc[strike, maturity])
        ]
    )
    known_values

    # Create all possible grid points (even where IV is missing)
    target_points = np.array(
        [(strike, maturity) for strike in strikes for maturity in maturities]
    )
    target_points

    # Interpolate IVs on the target grid
    interpolated_values = griddata(
        points=known_points,  # Known (strike, maturity) pairs
        values=known_values.reshape(-1, 1),  # Known IV values
        xi=target_points,  # Target grid points for interpolation
        method="cubic",  # Interpolation method ('linear', 'nearest', or 'cubic')
    )

    # smooth out the surface with a Gaussian filter
    interpolated_values = gaussian_filter(interpolated_values, sigma=2)

    # Reshape interpolated values back into the DataFrame format
    interpolated_puts = pd.DataFrame(
        interpolated_values.reshape(puts.shape), index=puts.index, columns=puts.columns
    )

    interpolated_puts.ffill(axis=1).bfill(axis=1, inplace=True)

    # Define realistic IV range
    min_iv = 0.01  # 1%
    max_iv = 2.0  # 200%

    # Filter values outside the range
    put_iv_surface = interpolated_puts.clip(lower=min_iv, upper=max_iv)

    put_iv_surface = put_iv_surface.dropna(axis=0, how="any")

    # put_price_surface = put_prices.loc[put_iv_surface.index, put_iv_surface.columns].dropna(axis=0, how='any')

    return puts, put_iv_surface


def build_call_iv_surface(option_chain, fwd_prices, asof_date):
    # Extract OTM calls for the IV surface due to better liquidity, lower bid-ask spread, and fewer early exercise risks
    options_data = option_chain.copy()
    options_data["expiration_date"] = dates_to_time_remaining(
        options_data[["expiration_date"]].values.squeeze(), asof_date
    )
    options_data.set_index(["expiration_date", "type", "strike"], inplace=True)

    calls = options_data.loc[
        pd.IndexSlice[:, "Call", :], "implied_volatility"
    ].to_frame()
    calls.index = calls.index.droplevel(1)
    calls = calls.unstack(level=0)
    calls.columns = calls.columns.droplevel(0)

    contracts_to_display = fwd_prices.shape[0]
    calls = calls.iloc[:, :contracts_to_display]
    calls = calls.sort_index()

    # Filter out calls with available market prices
    call_prices = options_data.loc[pd.IndexSlice[:, "Call", :], "midprice"].to_frame()
    call_prices.index = call_prices.index.droplevel(1)
    call_prices = call_prices.unstack(level=0)
    call_prices.columns = call_prices.columns.droplevel(0)

    call_prices = call_prices.iloc[:, :contracts_to_display]
    call_prices = call_prices.sort_index()

    calls = calls.loc[call_prices.index, :].dropna(how="all", axis=1)

    # Filter out only the OTM calls (strikes above the forward price for each maturity)
    otm_mask = pd.DataFrame(
        index=calls.index, columns=fwd_prices.values.squeeze().squeeze()
    )
    for c in otm_mask.columns:
        otm_mask[c] = otm_mask.index >= c  # Calls are OTM if strike >= forward price

    calls = calls.loc[calls.index >= otm_mask.sum(axis=0).idxmin(), :].iloc[
        :, :contracts_to_display
    ]

    # Drop duplicate rows
    calls = calls.loc[~calls.index.duplicated(keep="first")]

    # Prepare strike and maturity values
    strikes = calls.index.values
    maturities = calls.columns.values

    # Extract known IV values for interpolation
    known_points = np.array(
        [
            (strike, maturity)
            for strike in strikes
            for maturity in maturities
            if not np.isnan(calls.loc[strike, maturity])
        ]
    )
    known_values = np.array(
        [
            calls.loc[strike, maturity]
            for strike in strikes
            for maturity in maturities
            if not np.isnan(calls.loc[strike, maturity])
        ]
    )

    # Create grid for interpolation
    target_points = np.array(
        [(strike, maturity) for strike in strikes for maturity in maturities]
    )

    # Interpolate missing IV values
    interpolated_values = griddata(
        points=known_points,
        values=known_values.reshape(-1, 1),
        xi=target_points,
        method="cubic",
    )

    # Apply smoothing
    interpolated_values = gaussian_filter(interpolated_values, sigma=2)

    # Reshape interpolated values into a DataFrame
    interpolated_calls = pd.DataFrame(
        interpolated_values.reshape(calls.shape),
        index=calls.index,
        columns=calls.columns,
    )

    interpolated_calls.ffill(axis=1).bfill(axis=1, inplace=True)

    # Define IV bounds
    min_iv = 0.01
    max_iv = 2.0

    # Clip IV values within realistic range
    call_iv_surface = interpolated_calls.clip(lower=min_iv, upper=max_iv)

    call_iv_surface = call_iv_surface.dropna(axis=0, how="any")

    # Extract corresponding call price surface
    call_price_surface = call_prices.loc[
        call_iv_surface.index, call_iv_surface.columns
    ].dropna(axis=0, how="any")

    return calls, call_iv_surface


def apply_iv_smoothing(iv_surface, sigma=2, clip_bounds=(0.01, 1.0)):
    """
    Apply Gaussian smoothing along the strike price dimension.

    Parameters:
    iv_surface (pd.DataFrame): IV surface DataFrame (index = strike, columns = time to expiry).
    sigma (float): Standard deviation for Gaussian kernel. Higher = more smoothing.
    clip_bounds (tuple): Lower and upper bounds for IV values.

    Returns:
    pd.DataFrame: Smoothed IV surface.
    """
    # clip IV values to realistic bounds
    iv_surface = iv_surface.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    # Smooth IV surface along the strike price dimension
    smoothed_iv = iv_surface.apply(
        lambda col: gaussian_filter1d(col, sigma=sigma), axis=0
    )
    return smoothed_iv


def build_iv_and_price_surface(
    option_chain,
    fwd_prices,
    asof_date,
    strike_range,
    expiry_range,
    smoothing=False,
    smoothing_params={"sigma": 2, "clip_bounds": (0.01, 1.0)},
):
    """
    Builds implied volatility (IV) and price surfaces for options based on the given option chain and forward prices.
    Parameters:
    option_chain (DataFrame): The option chain data containing option prices and other relevant information.
    fwd_prices (DataFrame): The forward prices for the underlying asset.
    asof_date (datetime): The date as of which the surfaces are being built.
    strike_range (tuple): The range of strikes to consider for the surfaces.
    expiry_range (tuple): The range of expiries to consider for the surfaces.
    smoothing (bool, optional): Whether to apply smoothing to the surfaces. Default is False.
    smoothing_params (dict, optional): Parameters for the smoothing function. Default is {'sigma': 2, 'clip_bounds': (0.01, 1.0)}.
    Returns:
    tuple: A tuple containing the IV surface and the price surface.
    """

    puts, put_iv_surface = build_put_iv_surface(option_chain, fwd_prices, asof_date)
    put_price_surface = build_put_price_surface(option_chain, fwd_prices, asof_date)

    calls, call_iv_surface = build_call_iv_surface(option_chain, fwd_prices, asof_date)
    call_price_surface = build_call_price_surface(option_chain, fwd_prices, asof_date)

    iv_surface = build_iv_surface(
        put_iv_surface, call_iv_surface, strike_range, expiry_range
    )
    price_surface = build_price_surface(
        put_price_surface, call_price_surface, strike_range, expiry_range
    )

    if smoothing:
        iv_surface = apply_iv_smoothing(iv_surface, **smoothing_params)

    return iv_surface, price_surface


def build_iv_surface(
    put_iv_surface, call_iv_surface, strike_range=None, expiry_range=(0.0, 2.0)
):
    iv_surface = pd.concat((put_iv_surface, call_iv_surface), axis=0)
    # trim the IV surface strikes to desired range
    if strike_range is not None:
        iv_surface = iv_surface.loc[
            (iv_surface.index >= strike_range[0])
            & (iv_surface.index <= strike_range[1])
        ]
    # trim the IV surface expiries to desired range
    if expiry_range is not None:
        iv_surface = iv_surface.loc[
            :,
            (iv_surface.columns >= expiry_range[0])
            & (iv_surface.columns <= expiry_range[1]),
        ]

    # interpolate the IV surface
    iv_surface = iv_surface.interpolate(method="cubic", axis=0)

    min_iv = 0.00  # 0%
    max_iv = 2.0  # 200%

    # Filter values outside the range
    iv_surface = iv_surface.clip(lower=min_iv, upper=max_iv)

    return iv_surface


def build_put_price_surface(option_chain, fwd_prices, asof_date):
    options_data = option_chain.copy()
    options_data["expiration_date"] = dates_to_time_remaining(
        options_data[["expiration_date"]].values.squeeze(), asof_date
    )
    options_data.set_index(["expiration_date", "type", "strike"], inplace=True)

    put_price_surface = options_data.loc[
        pd.IndexSlice[:, "Put", :], "midprice"
    ].to_frame()
    put_price_surface.index = put_price_surface.index.droplevel(1)
    put_price_surface = put_price_surface.unstack(level=0)
    put_price_surface.columns = put_price_surface.columns.droplevel(0)

    months_to_display = fwd_prices.shape[0]
    put_price_surface = put_price_surface.iloc[:, :months_to_display]

    put_price_surface = put_price_surface.sort_index()

    # we want to filter out only the OTM put_price_surface
    # we can do this by eliminating strikes above the strip, for a given time to expiry
    otm_mask = pd.DataFrame(
        index=put_price_surface.index, columns=fwd_prices.values.squeeze()
    )
    for c in otm_mask.columns:
        otm_mask[c] = otm_mask.index <= c

    put_price_surface = put_price_surface.loc[
        put_price_surface.index <= otm_mask.sum(axis=0).idxmax(), :
    ].iloc[:, :months_to_display]

    # drop duplicate rows

    put_price_surface = put_price_surface.loc[
        ~put_price_surface.index.duplicated(keep="first")
    ]

    # Assume `put_price_surface` has index as strikes and columns as time_to_expiry
    # Flatten the DataFrame to get known points and values
    strikes = put_price_surface.index.values
    maturities = put_price_surface.columns.values

    # Create a 2D array of coordinates (strike, maturity) for known IV values
    known_points = np.array(
        [
            (strike, maturity)
            for strike in strikes
            for maturity in maturities
            if not np.isnan(put_price_surface.loc[strike, maturity])
        ]
    )

    # Extract corresponding IV values
    known_values = np.array(
        [
            put_price_surface.loc[strike, maturity]
            for strike in strikes
            for maturity in maturities
            if not np.isnan(put_price_surface.loc[strike, maturity])
        ]
    )

    # Create all possible grid points (even where IV is missing)
    target_points = np.array(
        [(strike, maturity) for strike in strikes for maturity in maturities]
    )

    # Interpolate IVs on the target grid
    interpolated_values = griddata(
        points=known_points,  # Known (strike, maturity) pairs
        values=known_values,  # Known IV values
        xi=target_points,  # Target grid points for interpolation
        method="cubic",  # Interpolation method ('linear', 'nearest', or 'cubic')
    )

    # smooth out the surface with a Gaussian filter
    interpolated_values = gaussian_filter(interpolated_values, sigma=2)

    # Reshape interpolated values back into the DataFrame format
    interpolated_put_prices = pd.DataFrame(
        interpolated_values.reshape(put_price_surface.shape),
        index=put_price_surface.index,
        columns=put_price_surface.columns,
    )

    interpolated_put_prices.ffill(axis=1).bfill(axis=1, inplace=True)

    # Define realistic price range
    min_price = 0.00  # 1%

    # Filter values outside the range
    put_price_surface = interpolated_put_prices.clip(lower=min_price).dropna(
        axis=0, how="any"
    )

    return put_price_surface


def build_call_price_surface(option_chain, fwd_prices, asof_date):
    options_data = option_chain.copy()
    options_data["expiration_date"] = dates_to_time_remaining(
        options_data[["expiration_date"]].values.squeeze(), asof_date
    )
    options_data.set_index(["expiration_date", "type", "strike"], inplace=True)

    call_price_surface = options_data.loc[
        pd.IndexSlice[:, "Call", :], "midprice"
    ].to_frame()
    call_price_surface.index = call_price_surface.index.droplevel(1)
    call_price_surface = call_price_surface.unstack(level=0)
    call_price_surface.columns = call_price_surface.columns.droplevel(0)

    months_to_display = fwd_prices.shape[0]
    call_price_surface = call_price_surface.iloc[:, :months_to_display]
    call_price_surface = call_price_surface.sort_index()

    # we want to filter out only the OTM call_price_surface
    # we can do this by eliminating strikes above the strip, for a given time to expiry
    otm_mask = pd.DataFrame(
        index=call_price_surface.index, columns=fwd_prices.values.squeeze()
    )
    for c in otm_mask.columns:
        otm_mask[c] = otm_mask.index >= c

    call_price_surface = call_price_surface.loc[
        call_price_surface.index >= otm_mask.sum(axis=0).idxmin(), :
    ].iloc[:, :months_to_display]
    # drop duplicate rows

    call_price_surface = call_price_surface.loc[
        ~call_price_surface.index.duplicated(keep="first")
    ]

    # Assume `call_price_surface` has index as strikes and columns as time_to_expiry
    # Flatten the DataFrame to get known points and values
    strikes = call_price_surface.index.values
    maturities = call_price_surface.columns.values

    # Create a 2D array of coordinates (strike, maturity) for known IV values
    known_points = np.array(
        [
            (strike, maturity)
            for strike in strikes
            for maturity in maturities
            if not np.isnan(call_price_surface.loc[strike, maturity])
        ]
    )

    # Extract corresponding IV values
    known_values = np.array(
        [
            call_price_surface.loc[strike, maturity]
            for strike in strikes
            for maturity in maturities
            if not np.isnan(call_price_surface.loc[strike, maturity])
        ]
    )

    # Create all possible grid points (even where IV is missing)
    target_points = np.array(
        [(strike, maturity) for strike in strikes for maturity in maturities]
    )

    # Interpolate IVs on the target grid
    interpolated_values = griddata(
        points=known_points,  # Known (strike, maturity) pairs
        values=known_values,  # Known IV values
        xi=target_points,  # Target grid points for interpolation
        method="cubic",  # Interpolation method ('linear', 'nearest', or 'cubic')
    )

    # smooth out the surface with a Gaussian filter
    interpolated_values = gaussian_filter(interpolated_values, sigma=2)

    # Reshape interpolated values back into the DataFrame format
    interpolated_call_prices = pd.DataFrame(
        interpolated_values.reshape(call_price_surface.shape),
        index=call_price_surface.index,
        columns=call_price_surface.columns,
    )

    interpolated_call_prices.ffill(axis=1).bfill(axis=1, inplace=True)

    # Define realistic price range
    min_price = 0.00  # 1%

    # Filter values outside the range
    call_price_surface = interpolated_call_prices.clip(lower=min_price).dropna(
        axis=0, how="any"
    )

    return call_price_surface


def build_price_surface(
    put_price_surface, call_price_surface, strike_range=None, expiry_range=(0.0, 2.0)
):
    price_surface = pd.concat((put_price_surface, call_price_surface), axis=0)
    # trim the IV surface strikes to desired range
    if strike_range is not None:
        price_surface = price_surface.loc[
            (price_surface.index >= strike_range[0])
            & (price_surface.index <= strike_range[1])
        ]
    # trim the IV surface expiries to desired range
    if expiry_range is not None:
        price_surface = price_surface.loc[
            :,
            (price_surface.columns >= expiry_range[0])
            & (price_surface.columns <= expiry_range[1]),
        ]

    return price_surface


def plot_vol_price_charts(
    iv_surface,
    price_surface,
    strike_range,
    expiry_range,
    iv_surface_title="Implied Volatility Surface",
    price_surface_title="Price Surface",
    eye_params=dict(x=2.0, y=0.6, z=0.9),
):
    """
    Plots the implied volatility surface and price surface charts using Plotly.
    Parameters:
    iv_surface (pd.DataFrame): DataFrame containing the implied volatility surface data.
    price_surface (pd.DataFrame): DataFrame containing the price surface data.
    expiry_range (list): List containing the start and end of the expiry range [start, end].
    strike_range (list): List containing the start and end of the strike range [start, end].
    iv_surface_title (str, optional): Title for the implied volatility surface plot. Defaults to 'Implied Volatility Surface'.
    price_surface_title (str, optional): Title for the price surface plot. Defaults to 'Price Surface'.
    Returns:
    None: This function displays the plot and does not return any value.
    """

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(iv_surface_title, price_surface_title),
        specs=[[{"type": "surface"}, {"type": "surface"}]],
    )

    # Add traces
    fig.add_trace(
        go.Surface(
            z=iv_surface.values,
            x=iv_surface.columns,
            y=iv_surface.index,
            showscale=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Surface(
            z=price_surface.values,
            x=price_surface.columns,
            y=price_surface.index,
            showscale=False,
        ),
        row=1,
        col=2,
    )

    # Layout for first subplot (IV Surface)
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickformat=",.1f",
                range=[expiry_range[0], expiry_range[1]],
                title="Time to Expiry (Years)",
                autorange="reversed",
            ),
            yaxis=dict(
                tickformat=",.0f",
                range=[strike_range[0], strike_range[1]],
                title="Strike Price",
            ),
            zaxis=dict(tickformat=",.0%", title="Implied Volatility"),
            camera=dict(eye=eye_params),
        ),
        scene2=dict(
            xaxis=dict(
                tickformat=",.1f",
                range=[expiry_range[0], expiry_range[1]],
                title="Time to Expiry (Years)",
                autorange="reversed",
            ),
            yaxis=dict(
                tickformat=",.0f",
                range=[strike_range[0], strike_range[1]],
                title="Strike Price",
            ),
            zaxis=dict(tickformat=",.2f", title="Price ($/Share)"),
            camera=dict(eye=eye_params),
        ),
        **get_default_plot_layout(),
        title_text=f"{iv_surface_title} and {price_surface_title}",
    )

    # Show figure
    fig.show(renderer="notebook")


def dates_to_time_remaining(dates, asof_date):
    return pd.Index((pd.to_datetime(dates) - pd.to_datetime(asof_date)).days / 365.0)


def time_remaining_to_dates(time_remaining, asof_date):
    return pd.DatetimeIndex(
        pd.to_datetime(asof_date) + pd.to_timedelta(time_remaining * 365, unit="D")
    ).normalize()


def black_scholes_price(S, K, r, sigma, T, option_type="call"):
    """
    Calculate the Black-Scholes price for a European option.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity in years.
        option_type (str): 'call' or 'put'.

    Returns:
        float: Option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")


def get_payoff_axis_settings(domain=None):
    """Returns common payoff axis settings"""
    settings = dict(
        title="Payoff",
        tickformat=".2f",
        dtick=2.5,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=0.8,
        range=[-10, 10],
    )
    if domain:
        settings.update(dict(domain=domain))
    return settings


def build_simulated_futures(
    asof_date,
    fwd_prices,
    iv_surface,
    months_to_display,
    comdty_code="CY",
    r=0.045,
    N=1000,
    dt=1/365.0,
    show_charts=False,
):
    # match indexes
    fwd_prices.index = iv_surface.columns
    # deduplicate the index
    iv_surface = iv_surface[~iv_surface.index.duplicated(keep="last")].ffill().bfill()

    simulated_futures = simulate_futures(
        asof_date,
        fwd_prices,
        iv_surface,
        r,
        N,
        dt,
        default_percentiles=np.arange(0.1, 1.0, 0.1),
        random_seed=123,
    )

    display_simulated_futures = simulated_futures['display_data'].copy()
    display_simulated_futures.index = time_remaining_to_dates(
        display_simulated_futures.index, asof_date
    ).strftime("%b %Y")

    fig = px.line(
        display_simulated_futures,
        title="Monte Carlo Simulation of Futures Prices",
        labels={"value": "Price ($/Share)", "index": "Expiration Date", "variable": ""},
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )
    fig.update_layout(
        **get_default_plot_layout(),
        yaxis=dict(tickformat="$.2f"),
        # Remove duplicate legend settings
    )

    # Add annotations to each trace
    for trace in fig.data:
        if not (any(_ in trace.name for _ in ["%"])):
            for i, value in enumerate(trace.y):
                annot_loc = get_annotation_location(trace.name)
                fig.add_annotation(
                    x=trace.x[i],
                    y=value,
                    text=f"${value:.2f}",
                    showarrow=True,
                    arrowhead=0,
                    ay=annot_loc["y"],
                    ax=annot_loc["x"],
                )

    add_weekend_rangebreaks(fig)
    display(display_simulated_futures.T.style.format("{:.2f}"))
    fig.show()

    return simulated_futures['simulated_futures']


def show_call_option_payoff(S, K, sigma, r, T):
    S_range = np.arange(S - 10, S + 20, 0.5)  # Underlying price range

    # Compute option price at S = S
    option_price_at_S0 = black_scholes_price(S, K, r, sigma, T, "call")

    # Compute call option price for each underlying price
    call_prices = np.array([black_scholes_price(S, K, r, sigma, T, "call") for S in S_range])

    # Compute payoffs
    payoff_no_cost = np.maximum(S_range - K, 0)
    payoff_with_cost = payoff_no_cost - option_price_at_S0

    long_call_payoff = np.concatenate(
        (payoff_no_cost.reshape(-1, 1), payoff_with_cost.reshape(-1, 1)), axis=1
    )

    # Seller payoffs (negative of buyer)
    seller_payoff_no_cost = -payoff_no_cost
    seller_payoff_with_cost = -payoff_with_cost

    short_call_payoff = np.concatenate(
        (seller_payoff_no_cost.reshape(-1, 1), seller_payoff_with_cost.reshape(-1, 1)),
        axis=1,
    )

    # Create subplots
    fig = go.Figure()

    # Left Chart: Call Option Buyer Payoff
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoff_no_cost,
            mode="lines",
            name="Payoff (No Cost)",
            line=dict(color="purple"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoff_with_cost,
            mode="lines",
            name="Payoff (With Cost)",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[S],
            y=[-option_price_at_S0],
            mode="markers+text",
            name="Premium Paid",
            text=[f"Price: -${option_price_at_S0:.2f}"],
            textposition="top right",
            marker=dict(color="blue", size=8),
        )
    )

    # Right Chart: Call Option Seller Payoff
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=seller_payoff_no_cost,
            mode="lines",
            name="Payoff (No Cost)",
            line=dict(color="purple"),
            xaxis="x2",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=seller_payoff_with_cost,
            mode="lines",
            name="Payoff (With Cost)",
            line=dict(color="orange"),
            xaxis="x2",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[S],
            y=[option_price_at_S0],
            mode="markers+text",
            name="Premium Received",
            text=[f"Price: +${option_price_at_S0:.2f}"],
            textposition="top right",
            marker=dict(color="red", size=8),
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Layout adjustments
    fig.update_layout(
        title_text="Call Option Payoff: Buyer vs Seller",
        xaxis=dict(title="Underlying Price", domain=[0, 0.45]),
        yaxis=dict(**get_payoff_axis_settings()),
        xaxis2=dict(title="Underlying Price", domain=[0.55, 1], anchor="y2"),
        yaxis2=dict(**get_payoff_axis_settings(), anchor="x2"),
        **get_default_plot_layout(),
    )

    # Show figure
    fig.show()

    return pd.DataFrame(
        index=S_range,
        data=np.concatenate((long_call_payoff, short_call_payoff), axis=1),
        columns=[
            "Long Call (no cost)",
            "Long Call (with cost)",
            "Short Call (no cost)",
            "Short Call (with cost)",
        ],
    )


def show_put_option_payoff(S, K, sigma, r, T):
    S_range = np.arange(S - 20, S + 10, 0.5)  # Underlying price range

    # Compute option price at S
    option_price_at_S0 = black_scholes_price(S, K, r, sigma, T, "put")

    # Compute put option price for each underlying price
    put_prices = np.array([black_scholes_price(S, K, r, sigma, T, "put") for S in S_range])

    # Compute payoffs
    payoff_no_cost = np.maximum(K - S_range, 0)
    payoff_with_cost = payoff_no_cost - option_price_at_S0

    long_put_payoff = np.concatenate(
        (payoff_no_cost.reshape(-1, 1), payoff_with_cost.reshape(-1, 1)), axis=1
    )

    # Seller payoffs (negative of buyer)
    seller_payoff_no_cost = -payoff_no_cost
    seller_payoff_with_cost = -payoff_with_cost

    short_put_payoff = np.concatenate(
        (seller_payoff_no_cost.reshape(-1, 1), seller_payoff_with_cost.reshape(-1, 1)),
        axis=1,
    )

    # Create subplots
    fig = go.Figure()

    # Left Chart: Put Option Buyer Payoff
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoff_no_cost,
            mode="lines",
            name="Payoff (No Cost)",
            line=dict(color="purple"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoff_with_cost,
            mode="lines",
            name="Payoff (With Cost)",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[S],
            y=[-option_price_at_S0],
            mode="markers+text",
            name="Premium Paid",
            text=[f"Price: -${option_price_at_S0:.2f}"],
            textposition="top right",
            marker=dict(color="blue", size=8),
        )
    )

    # Right Chart: Put Option Seller Payoff
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=seller_payoff_no_cost,
            mode="lines",
            name="Payoff (No Cost)",
            line=dict(color="purple"),
            xaxis="x2",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=seller_payoff_with_cost,
            mode="lines",
            name="Payoff (With Cost)",
            line=dict(color="orange"),
            xaxis="x2",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[S],
            y=[option_price_at_S0],
            mode="markers+text",
            name="Premium Received",
            text=[f"Price: +${option_price_at_S0:.2f}"],
            textposition="top right",
            marker=dict(color="red", size=8),
            xaxis="x2",
            yaxis="y2",
        )
    )

    # Layout adjustments
    fig.update_layout(
        title_text="Put Option Payoff: Buyer vs Seller",
        xaxis=dict(title="Underlying Price", domain=[0, 0.45]),
        yaxis=dict(**get_payoff_axis_settings()),
        xaxis2=dict(title="Underlying Price", domain=[0.55, 1], anchor="y2"),
        yaxis2=dict(**get_payoff_axis_settings(), anchor="x2"),
        **get_default_plot_layout(),
    )

    # Show figure
    fig.show()

    return pd.DataFrame(
        index=S_range,
        data=np.concatenate((long_put_payoff, short_put_payoff), axis=1),
        columns=[
            "Long Put (no cost)",
            "Long Put (with cost)",
            "Short Put (no cost)",
            "Short Put (with cost)",
        ],
    )


def build_costless_collar(put_payoffs, call_payoffs, put_strike=70, call_strike=90.25):
    collar_payoff = (
        pd.concat(
            (put_payoffs.filter(like="Long"), call_payoffs.filter(like="Short")),
            axis=1,
        )
        .ffill()
        .bfill()
    )

    collar_payoff["Collar (no cost)"] = (
        collar_payoff["Long Put (no cost)"] + collar_payoff["Short Call (no cost)"]
    )
    collar_payoff["Collar (with cost)"] = (
        collar_payoff["Long Put (with cost)"] + collar_payoff["Short Call (with cost)"]
    )

    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Long Put", "Short Call", "Collar")
    )

    # Add traces for each component
    for i, (name, data) in enumerate(collar_payoff.items()):
        fig.add_trace(
            go.Scatter(x=collar_payoff.index, y=data, mode="lines", name=name),
            row=1,
            col=(i // 2 + 1),
        )

    # Add price marker and strike lines
    current_price = 80  # Default current price
    for col in [1, 2, 3]:
        # Add current price marker
        fig.add_trace(
            go.Scatter(
                x=[current_price],
                y=[0],
                mode="markers",
                name="Current Price",
                line=dict(dash="dash", color="black"),
            ),
            row=1,
            col=col,
        )
        
        # Add put strike line
        fig.add_trace(
            go.Scatter(
                x=[put_strike, put_strike],
                y=[-15, 15],
                mode="lines",
                line=dict(dash="dash", color="red"),
            ),
            row=1,
            col=col,
        )
        
        # Add call strike line
        fig.add_trace(
            go.Scatter(
                x=[call_strike, call_strike],
                y=[-15, 15],
                mode="lines",
                line=dict(dash="dash", color="green"),
            ),
            row=1,
            col=col,
        )

    update_payoff_axes(fig, num_subplots=3)

    fig.update_layout(
        title="Construction of A Collar", 
        **get_default_plot_layout(), 
        showlegend=False
    )

    fig.show()
    return collar_payoff
