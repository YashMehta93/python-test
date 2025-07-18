import requests
import pandas as pd
import logging


def fetch_data(symbol, from_date, to_date, interval):
    """
    Fetch OHLCV data from the API and return it as a pandas DataFrame.

    Args:
        symbol (str): Stock symbol.
        from_date (str): Start date for fetching data (format: YYYY-MM-DD).
        to_date (str): End date for fetching data (nullable, format: YYYY-MM-DD).
        interval (str): Time interval for OHLCV data (e.g., '5M', '15M').

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'].
    """
    url = "http://localhost:8080/dataset/load"
    headers = {"Content-Type": "application/json"}
    payload = {
        "symbol": symbol,
        "fromDate": from_date,
        "toDate": to_date,
        "interval": interval,
    }

    try:
        # Make the POST request to fetch data
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_data = response.json()

        # Extract candle data
        candle_list = response_data.get("data", {}).get("candleList", [])
        if not candle_list:
            raise ValueError("No data returned for the given request.")

        # Convert to DataFrame
        df = pd.DataFrame(candle_list)

        # Rename and convert columns to appropriate data types
        df.rename(
            columns={
                "ts": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "oi": "open_interest",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert timestamp to datetime
        df.set_index("timestamp", inplace=True)  # Set timestamp as index

        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Error processing response: {e}")
        raise
