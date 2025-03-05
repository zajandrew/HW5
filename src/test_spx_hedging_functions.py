import numpy as np
import pandas as pd

import pull_options_data
import spx_hedging_functions
from settings import config
from pandas import Timestamp, NaT

def test_load_data():
    df = pull_options_data.load_spx_option_data(
        asof_date="2023-08-02",
        start_date=config("START_DATE"),
        end_date=config("END_DATE"),
    )
    # import misc_tools
    # print(misc_tools.df_to_literal(df.tail()))
    df_expected = pd.DataFrame(
    {
        'secid': [108105.0, 108105.0, 108105.0, 108105.0, 108105.0],
        'optionid': [155477343.0, 155271050.0, 155715629.0, 155271051.0, 155271052.0],
        'date': [Timestamp('2023-08-02 00:00:00'), Timestamp('2023-08-02 00:00:00'), Timestamp('2023-08-02 00:00:00'), Timestamp('2023-08-02 00:00:00'), Timestamp('2023-08-02 00:00:00')],
        'expiration_date': [Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00')],
        'last_date': [NaT, Timestamp('2023-08-01 00:00:00'), NaT, NaT, Timestamp('2023-08-01 00:00:00')],
        'type': ['Put', 'Put', 'Put', 'Put', 'Put'],
        'strike': [5700.0, 5800.0, 5900.0, 6000.0, 6200.0],
        'best_bid': [933.1, 1027.9, 1119.5, 1212.5, 1399.4],
        'best_offer': [981.1, 1075.9, 1167.5, 1260.5, 1447.4],
        'volume': [0.0, 0.0, 0.0, 0.0, 0.0],
        'implied_volatility': [0.118289, 0.126093, 0.124332, 0.123955, 0.115528],
        'delta': [-0.941359, -0.944855, -0.958905, -0.968043, -0.983428],
        'gamma': [0.000199, 0.000176, 0.000132, 0.0001, 3.9e-05],
        'theta': [218.5383, 224.4454, 240.1798, 252.5082, 276.4534],
        'vega': [435.5181, 410.4616, 303.8221, 228.1846, 83.18876],
        'contract_size': [100.0, 100.0, 100.0, 100.0, 100.0],
        'amsettlement': [0.0, 0.0, 0.0, 0.0, 0.0],
        'forwardprice': [4703.416938, 4703.416938, 4703.416938, 4703.416938, 4703.416938],
        'expiration': [Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00'), Timestamp('2024-06-28 00:00:00')],
        'midprice': [957.1, 1051.9, 1143.5, 1236.5, 1423.4]
    }, index=[13129, 13130, 13131, 13132, 13133]
    )
    assert df.tail().equals(df_expected)

def test_collar_net_cost():
    """
    Test that the collar net cost calculation matches expected value
    when using default parameters from 01_spx.py
    """
    # Setup the same parameters as in 01_spx.py
    asof_date = "2023-08-02"
    interpolation_args = {"method": "cubic", "order": 3}
    d = 0.80
    contracts_to_display = 15
    r = 0.045
    N = 1000
    days_in_year = 365.0
    dt = 1 / days_in_year

    strike_range = (1000, 6000)
    expiry_range = (0.01, 2.0)

    # Load data
    option_chain = pull_options_data.load_spx_option_data(
        asof_date=asof_date,
        start_date=config("START_DATE"),
        end_date=config("END_DATE"),
    )

    # Build market predictions
    delta_map, prediction_range = spx_hedging_functions.build_market_predictions(
        option_chain=option_chain,
        asof_date=asof_date,
        contracts_to_display=contracts_to_display,
        interpolation_args=interpolation_args,
    )

    # Get forward prices and build surfaces
    fwd_prices = spx_hedging_functions.get_fwd_prices(
        asof_date, option_chain, contracts_to_display
    )
    iv_surface, price_surface = spx_hedging_functions.build_iv_and_price_surface(
        option_chain,
        fwd_prices,
        asof_date,
        strike_range,
        expiry_range,
        smoothing=True,
        smoothing_params={"sigma": 2, "clip_bounds": (0.05, 0.75)},
    )

    # Build delta map
    delta_map = spx_hedging_functions.build_delta_map(
        option_chain,
        asof_date,
        contracts_to_display,
        show_detail=False,
        weighted=False,
        interpolate_missing=True,
        interpolation_args=interpolation_args,
    )

    # Simulate futures
    default_percentiles = pd.unique(
        pd.Series(
            np.round(
                np.sort(np.append(np.arange(0.1, 1.0, 0.1), np.array([d, 1 - d]))),
                3,
            )
        )
    )

    futures_data = spx_hedging_functions.simulate_futures(
        asof_date,
        fwd_prices,
        iv_surface,
        r,
        N,
        dt,
        default_percentiles,
        random_seed=123,
    )

    simulated_futures = futures_data["simulated_futures"]
    delta_map.columns = spx_hedging_functions.dates_to_time_remaining(
        delta_map.columns, asof_date
    )
    delta_map = delta_map.loc[:, simulated_futures.index]

    # Build collar strikes
    collar_data = spx_hedging_functions.build_collar_strikes(
        d,
        delta_map,
        simulated_futures=simulated_futures,
        contracts_to_display=contracts_to_display,
        asof_date=asof_date,
        option_chain=option_chain,
    )

    # Test the net cost
    expected_net_cost = -0.3750
    actual_net_cost = collar_data["net_cost"]

    assert abs(actual_net_cost - expected_net_cost) < 0.0001, (
        f"Expected net cost {expected_net_cost}, but got {actual_net_cost}"
    )
