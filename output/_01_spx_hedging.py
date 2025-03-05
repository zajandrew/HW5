#!/usr/bin/env python
# coding: utf-8

# # Case Study: Hedging A Long-Only SPX Portfolio With Costless Collars    

# In[ ]:


import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio

import pull_options_data
import spx_hedging_functions
from settings import config

pd.set_option("display.max_columns", None)
pio.templates.default = "plotly_white"


warnings.filterwarnings("ignore")


# In[ ]:


# Load environment variables
DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")

START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


# In[ ]:


asof_date = "2023-08-02"
interpolation_args = {"method": "cubic", "order": 3}
# enter the range of costless collar to construct
d = 0.80
contracts_to_display = 15
r = 0.045
N = 1000
days_in_year = 365.0

strike_range = (1000, 6000)
expiry_range = (0.01, 2.0)

dt = 1 / days_in_year

option_chain = pull_options_data.load_spx_option_data(
    asof_date=asof_date, start_date=START_DATE, end_date=END_DATE
)
# do we need to check if forward prices for calls and puts are the same? no, because the sql_query() function is pulling the forward price from a single other dataset and joining tables


# In[ ]:


# Build market predictions
delta_map, prediction_range = spx_hedging_functions.build_market_predictions(
    option_chain=option_chain,
    asof_date=asof_date,
    contracts_to_display=contracts_to_display,
    interpolation_args=interpolation_args,
)
# Display prediction range table
prediction_range_reformat = prediction_range.copy()
prediction_range_reformat.index = prediction_range_reformat.index.strftime("%Y-%m-%d")
prediction_range_reformat.index.name = "Expiration Date"
prediction_range_reformat = prediction_range_reformat.T
prediction_range_reformat = prediction_range_reformat.iloc[:, :contracts_to_display]
prediction_range_reformat = prediction_range_reformat.style.format("{:.2f}")
prediction_range_reformat


# In[ ]:


# Create pareto chart visualization
pareto_data = spx_hedging_functions.build_pareto_chart(
    ticker="SPX",
    asof_date=asof_date,
    prediction_range=prediction_range,
    delta_map=delta_map,
    contracts_to_display=contracts_to_display,
)

# Display pareto chart
pareto_data['figure'].show()


# In[ ]:


# Display prediction range table
pareto_data['prediction_range'].style.format("{:.2f}")


# In[ ]:


# Create delta heatmap visualization
delta_heatmap = spx_hedging_functions.plot_delta_heatmap(
    ticker="SPX",
    asof_date=asof_date,
    delta_map=delta_map,
    contracts_to_display=contracts_to_display,
    heatmap_step=3,
)


# In[ ]:


# get the forward prices
fwd_prices = spx_hedging_functions.get_fwd_prices(asof_date, option_chain, contracts_to_display)
iv_surface, price_surface = spx_hedging_functions.build_iv_and_price_surface(
    option_chain,
    fwd_prices,
    asof_date,
    strike_range,
    expiry_range,
    smoothing=True,
    smoothing_params={"sigma": 2, "clip_bounds": (0.05, 0.75)},
)


# In[ ]:


spx_hedging_functions.plot_vol_price_charts(
    iv_surface=iv_surface,
    price_surface=price_surface,
    strike_range=[np.floor(iv_surface.index.min()), np.ceil(iv_surface.index.max())],
    expiry_range=[iv_surface.columns.min(), iv_surface.columns.max()],
    iv_surface_title="SPX Implied Volatility",
    price_surface_title="Price Surface as of " + asof_date,
)


# In[ ]:


default_percentiles = np.unique(
    np.round(np.sort(np.append(np.arange(0.1, 1.0, 0.1), np.array([d, 1 - d]))), 3)
)
print(f"Constructing a {d:.0%} / {(1 - d):.0%} collar...")


# In[ ]:


# build the delta map
delta_map = spx_hedging_functions.build_delta_map(
    option_chain,
    asof_date,
    contracts_to_display,
    show_detail=False,
    weighted=False,
    interpolate_missing=True,
    interpolation_args=interpolation_args,
)


# In[ ]:


# simulate a range of futures, including the collar strike %iles
futures_data = spx_hedging_functions.simulate_futures(
    asof_date, fwd_prices, iv_surface, r, N, dt, default_percentiles, random_seed=123
)
# Display simulation results
futures_data['figure'].show()


# In[ ]:


futures_data['display_data'].T.style.format("{:.2f}")


# In[ ]:


simulated_futures = futures_data['simulated_futures']
delta_map.columns = spx_hedging_functions.dates_to_time_remaining(delta_map.columns, asof_date)
delta_map = delta_map.loc[:, simulated_futures.index]

# Build collar strikes data and visualization
collar_data = spx_hedging_functions.build_collar_strikes(
    d,
    delta_map,
    simulated_futures=simulated_futures,
    contracts_to_display=contracts_to_display,
    asof_date=asof_date,
    option_chain=option_chain,
)


# In[ ]:


# Display the figure
collar_data['figure'].show()


# In[ ]:


# Display formatted collar strikes table
display_collar_strikes = collar_data['collar_strikes'].copy()
display_collar_strikes.index = display_collar_strikes.index.strftime("%Y-%m-%d")
display_collar_strikes.T.style.format("{:.2f}")



# In[ ]:


# Display net cost
print(f"Net cost to hedge this set of collars: ${collar_data['net_cost']:.4f} per unit")


# In[ ]:




