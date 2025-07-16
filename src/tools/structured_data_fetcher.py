import yfinance as yf
import pandas as pd

class StructuredDataFetcher:
    """
    Fetches historical currency exchange rate data using the yfinance library.
    """

    def fetch_data(self, currency_pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical data for a given currency pair.

        Args:
            currency_pair (str): The currency pair in "BASE/QUOTE" format (e.g., "USD/CNY").
            start_date (str): The start date for the data range (YYYY-MM-DD).
            end_date (str): The end date for the data range (YYYY-MM-DD).

        Returns:
            pd.DataFrame: A DataFrame containing the historical data, with the date as the index.
                          Returns an empty DataFrame if the data cannot be fetched.
        """
        ticker = self._convert_to_ticker(currency_pair)
        if not ticker:
            print(f"Invalid currency pair format: {currency_pair}")
            return pd.DataFrame()

        try:
            print(f"Fetching data for ticker: {ticker} from {start_date} to {end_date}")
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                print(f"No data found for ticker '{ticker}' in the specified date range.")
                return pd.DataFrame()
            print("Data fetched successfully.")
            return data
        except Exception as e:
            print(f"An error occurred while fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def _convert_to_ticker(self, currency_pair: str) -> str:
        """
        Converts a standard currency pair string to a yfinance-compatible ticker.
        Example: "USD/CNY" -> "CNY=X"
        """
        parts = currency_pair.strip().upper().split('/')
        if len(parts) != 2 or not all(p for p in parts):
            return ""
        # yfinance expects the format to be "QUOTE=X" for the BASE/QUOTE pair.
        # For example, for USD/CNY, you look up "CNY=X" to get how many CNY one USD is.
        return f"{parts[1]}=X"

if __name__ == '__main__':
    # --- Demonstration ---
    fetcher = StructuredDataFetcher()
    
    # Example 1: Fetch USD to CNY exchange rate
    currency_pair_to_fetch = "USD/CNY"
    start_date_range = "2023-01-01"
    end_date_range = "2023-03-31"
    
    print(f"\n--- Attempting to fetch data for {currency_pair_to_fetch} ---")
    historical_data = fetcher.fetch_data(
        currency_pair=currency_pair_to_fetch,
        start_date=start_date_range,
        end_date=end_date_range
    )
    
    if not historical_data.empty:
        print(f"\nSuccessfully fetched data for {currency_pair_to_fetch}:")
        print(historical_data.head())
    else:
        print(f"\nFailed to fetch data for {currency_pair_to_fetch}.")

    # Example 2: Invalid Ticker
    print("\n--- Attempting to fetch data for an invalid pair 'USD/INVALID' ---")
    invalid_data = fetcher.fetch_data("USD/INVALID", "2023-01-01", "2023-01-31")
    if invalid_data.empty:
        print("Correctly handled invalid pair by returning an empty DataFrame.")