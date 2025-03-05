"""
This file contains our SQL Query to Wharton Research Dataservices (WRDS).

We connect to the database, and then submit a query per year. This code was
written by Viren Desai for FINM 32900 by Jeremy Bejarano
"""

import time
from pathlib import Path

import pandas as pd
import wrds

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))
DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")

START_DATE = config("START_DATE")
END_DATE = config("END_DATE")


description_opt_Met = {
    "secid": "Security ID",
    "cusip": "CUSIP Number",
    "date": "date",
    "symbol": "symbol",
    "exdate": "Expiration date of the Option",
    "last_date": "Date of last trade",
    "cp_flag": "C=call, P=Put",
    "strike_price": "Strike Price of the Option TIMES 1000",
    "best_bid": "Highest Closing Bid Across All Exchanges",
    "best_offer": "Lowest Closing Ask Across All Exchanges",
    "open_interest": "Open Interest for the Option",
    "impl_volatility": "Implied Volatility of the Option",
    "exercise_style": "(A)merican, (E)uropean, or ? (exercise_style)",
}


def sql_query(secid=108105, year=1996, start="1996-01-01", end="2012-01-31"):
    sql_query = f"""
        SELECT 
            a.secid, 
            a.optionid, 
            a.date, 
            a.exdate, 
            a.last_date, 
            a.cp_flag,
            a.strike_price,
            a.best_bid,
            a.best_offer,
            a.volume,
            a.impl_volatility,
            a.delta,
            a.gamma,
            a.theta,
            a.vega,
            a.contract_size,
            b.amsettlement, 
            b.forwardprice, 
            b.expiration
        FROM optionm_all.opprcd{year} AS a
        JOIN optionm_all.fwdprd{year} AS b 
            ON a.secid = b.secid AND a.date = b.date AND a.exdate = b.expiration
        WHERE a.secid = '{secid}' 
          AND '{start}' <= a.date 
          AND a.date <= '{end}' 
          AND a.am_settlement = 0 
          AND b.amsettlement = 0
    """
    return sql_query


def pull_all_optm_data(wrds_username, secid=108105, start_date="1996-01-01", end_date="2012-01-31"):
    """
    Pull options data from WRDS for the specified security and date range.
    
    Parameters
    ----------
    wrds_username : str
        WRDS username for authentication.
    secid : int, optional
        Security ID, defaults to 108105 (SPX).
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format, defaults to '1996-01-01'.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format, defaults to '2012-01-31'.
    
    Returns
    -------
    pandas.DataFrame
        Raw options data from WRDS.
    """
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Connect to WRDS
    db = wrds.Connection(wrds_username=wrds_username)
    
    # Get the range of years to query
    start_year = start_date.year
    end_year = end_date.year
    
    # Initialize an empty list to store DataFrames for each year
    dfs = []
    
    # Query data for each year in the range
    for year in range(start_year, end_year + 1):
        # Adjust start and end dates for the current year
        year_start = max(start_date, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end_date, pd.Timestamp(f"{year}-12-31"))
        
        # Generate SQL query for the current year
        query = sql_query(
            secid=secid,
            year=year,
            start=year_start.strftime("%Y-%m-%d"),
            end=year_end.strftime("%Y-%m-%d")
        )
        
        # Execute query and append results to the list
        df_year = db.raw_sql(query)
        
        if not df_year.empty:
            dfs.append(df_year)
    
    # Close the database connection
    db.close()
    
    # Combine all yearly DataFrames
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        
        # Convert date columns to datetime
        date_columns = ["date", "exdate", "last_date", "expiration"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Divide strike_price by 1000 to get the actual strike price
        if "strike_price" in df.columns:
            df["strike_price"] = df["strike_price"] / 1000
        
        return df
    else:
        return pd.DataFrame()


def load_all_optm_data(data_dir=DATA_DIR, secid=108105, start_date="1996-01-01", end_date="2012-01-31"):
    """
    Load options data from disk.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory where data is stored, defaults to DATA_DIR.
    secid : int, optional
        Security ID, defaults to 108105 (SPX).
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format, defaults to '1996-01-01'.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format, defaults to '2012-01-31'.
    
    Returns
    -------
    pandas.DataFrame
        Options data loaded from disk.
    """
    # Convert dates to datetime for consistent formatting
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create filename based on date range
    start_year_month = start_date.strftime("%Y-%m")
    end_year_month = end_date.strftime("%Y-%m")
    file_name = f"spx_options_{start_year_month}_{end_year_month}.parquet"
    file_path = Path(data_dir) / file_name
    
    # Load data from disk
    return pd.read_parquet(file_path)


def load_spx_option_data(asof_date, start_date=START_DATE, end_date=END_DATE):
    """
    Load SPX options data for a specific date.
    
    Parameters
    ----------
    asof_date : str or datetime
        Date for which to load options data.
    start_date : str, optional
        Start date of the data range, defaults to START_DATE.
    end_date : str, optional
        End date of the data range, defaults to END_DATE.
    
    Returns
    -------
    pandas.DataFrame
        Filtered options data for the specified date.
    """
    # Convert asof_date to datetime if it's a string
    asof_date = pd.to_datetime(asof_date)
    
    # Load the full dataset
    spx_options_data = load_all_optm_data(
        data_dir=DATA_DIR,
        secid=108105,
        start_date=start_date,
        end_date=end_date
    )
    
    # Extract only the data for the asof_date
    try:
        filtered_data = spx_options_data.loc[spx_options_data["date"] == asof_date]
        if filtered_data.empty:
            raise KeyError("No data for exact date")
    except KeyError:
        # If this specific asof_date is not available, get the nearest date
        # preceding this
        filtered_data = (
            spx_options_data.loc[spx_options_data["date"] <= asof_date]
            .sort_values("date", ascending=False)
            .iloc[0:1]  # Get first row as DataFrame instead of Series
        )
        if filtered_data.empty:
            raise ValueError(f"No data available on or before {asof_date}")

    filtered_data.reset_index(drop=True, inplace=True)
    
    return filtered_data


def prepare_option_chain(options_data):
    """
    Cleans and prepares options data for analysis.
    
    Parameters
    ----------
    options_data : pandas.DataFrame
        Raw options data to be cleaned.
    
    Returns
    -------
    pandas.DataFrame
        Cleaned options data with: - Added 'midprice' column (average of
        best_bid and best_offer) - Renamed columns for compatibility with other
        functions - Converted option type values from 'C'/'P' to 'Call'/'Put'
    """
    # Create a copy to avoid modifying the original
    cleaned_data = options_data.copy()
    
    # Create a new midprice column (average of best bid and best offer)
    cleaned_data["midprice"] = (
        cleaned_data["best_bid"] + cleaned_data["best_offer"]
    ) / 2

    # Rename columns to work with the provided functions
    rename_columns = {
        "cp_flag": "type",
        "impl_volatility": "implied_volatility",
        "strike_price": "strike",
        "exdate": "expiration_date",
    }
    cleaned_data = cleaned_data.rename(columns=rename_columns)

    # Change the values in the "type" column from 'C' and 'P' to 'Call' and
    # 'Put'
    cleaned_data["type"] = cleaned_data["type"].apply(
        lambda x: "Call" if x == "C" else "Put"
    )

    return cleaned_data


def _demo():
    df = load_spx_option_data(
        asof_date="2023-08-02",
        start_date=START_DATE,
        end_date=END_DATE,
    )
    df.tail()
    


if __name__ == "__main__":
    # Pull the data from WRDS
    df = pull_all_optm_data(
        wrds_username=WRDS_USERNAME,
        secid=108105,
        start_date=str(START_DATE),
        end_date=str(END_DATE),
    )
    
    option_chain = prepare_option_chain(df)

    # Create the filename based on date range
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    start_year_month = start_date.strftime("%Y-%m")
    end_year_month = end_date.strftime("%Y-%m")
    file_name = f"spx_options_{start_year_month}_{end_year_month}.parquet"
    
    # Ensure the data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the data to disk
    file_path = DATA_DIR / file_name
    option_chain.to_parquet(file_path)
