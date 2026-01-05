"""
Data Fetching Utility for C-CFM Examples

Fetches economic and financial data from FRED and Yahoo Finance
for demonstrating the Causal Conditional Flow Matching framework.

This module provides 20 variables covering:
- Market data (equities, volatility, commodities)
- Interest rates and credit spreads
- Macroeconomic indicators

Data is fetched for a fixed time period (2010-2023) to ensure reproducibility.
"""

import numpy as np
import polars as pl
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings


# Fixed date range for reproducibility
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"


# Variable definitions
YAHOO_TICKERS = {
    # Market indices
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "EFA": "EAFE International ETF",

    # Volatility
    "^VIX": "VIX Volatility Index",

    # Bonds
    "TLT": "20+ Year Treasury Bond ETF",
    "HYG": "High Yield Corporate Bond ETF",
    "LQD": "Investment Grade Corporate Bond ETF",

    # Commodities
    "GLD": "Gold ETF",
    "USO": "Oil ETF",

    # Sector ETFs
    "XLF": "Financial Sector ETF",
    "XLE": "Energy Sector ETF",
}

FRED_SERIES = {
    # Interest Rates
    "DFF": "Federal Funds Rate",
    "DGS10": "10-Year Treasury Rate",
    "DGS2": "2-Year Treasury Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",

    # Credit Spreads
    "BAMLH0A0HYM2": "High Yield OAS Spread",

    # Macro indicators
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "CPI All Items",
    "GDPC1": "Real GDP",
}


def fetch_yahoo_data(
    tickers: Optional[List[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE
) -> pl.DataFrame:
    """Fetch daily price data from Yahoo Finance.

    Args:
        tickers: List of Yahoo Finance tickers (None = use defaults)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        df: Polars DataFrame with date index and adjusted close prices
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    if tickers is None:
        tickers = list(YAHOO_TICKERS.keys())

    print(f"Fetching Yahoo Finance data for {len(tickers)} tickers...")

    # Fetch data
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if len(tickers) == 1:
        # Single ticker returns Series, wrap in DataFrame
        prices = data['Close'].to_frame()
        prices.columns = tickers
    else:
        prices = data['Close']

    # Convert to Polars
    prices_reset = prices.reset_index()
    df = pl.from_pandas(prices_reset)

    # Rename date column
    df = df.rename({'Date': 'date'})

    print(f"  Retrieved {len(df)} daily observations")

    return df.with_columns(pl.col("date").cast(pl.Date))


def fetch_fred_data(
    series: Optional[List[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    api_key: Optional[str] = None
) -> pl.DataFrame:
    """Fetch economic data from FRED.

    Args:
        series: List of FRED series IDs (None = use defaults)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        api_key: FRED API key (optional, but recommended for reliability)

    Returns:
        df: Polars DataFrame with date index and series values
    """
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("fredapi is required. Install with: pip install fredapi")

    if series is None:
        series = list(FRED_SERIES.keys())

    # Initialize FRED client
    if api_key:
        fred = Fred(api_key=api_key)
    else:
        # Try to get from environment or use demo mode
        import os
        api_key = os.environ.get('FRED_API_KEY')
        if api_key:
            fred = Fred(api_key=api_key)
        else:
            warnings.warn(
                "No FRED API key provided. Some series may fail. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            # Return empty DataFrame with expected structure
            return pl.DataFrame({'date': []})

    print(f"Fetching FRED data for {len(series)} series...")

    dfs = []
    for series_id in series:
        try:
            data = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            df_series = pl.DataFrame({
                'date': data.index.tolist(),
                series_id: data.values.tolist()
            })
            dfs.append(df_series)
            print(f"  ✓ {series_id}: {len(data)} observations")
        except Exception as e:
            print(f"  ✗ {series_id}: Failed ({e})")

    if not dfs:
        return pl.DataFrame({'date': []})

    # Join all series on date
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, on='date', how='outer', coalesce=True)

    # Sort by date
    result = result.sort('date')

    print(f"  Retrieved {len(result)} observations total")

    return result.with_columns(pl.col("date").cast(pl.Date))


def fetch_all_data(
    yahoo_tickers: Optional[List[str]] = None,
    fred_series: Optional[List[str]] = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    fred_api_key: Optional[str] = None
) -> pl.DataFrame:
    """Fetch combined market and economic data.

    Args:
        yahoo_tickers: Yahoo Finance tickers (None = use defaults)
        fred_series: FRED series IDs (None = use defaults)
        start_date: Start date
        end_date: End date
        fred_api_key: FRED API key

    Returns:
        df: Combined Polars DataFrame
        fast_vars: Dict mapping ticker/series to description (market/fast variables)
        slow_vars: Dict mapping ticker/series to description (macro/slow variables)
    """
    # Fetch Yahoo data (daily)
    yahoo_df = fetch_yahoo_data(yahoo_tickers, start_date, end_date)

    # Fetch FRED data (mixed frequencies)
    fred_df = fetch_fred_data(fred_series, start_date, end_date, fred_api_key)

    if len(fred_df) == 0:
        # FRED failed, just use Yahoo data
        combined = yahoo_df

        # All Yahoo are "fast" market variables
        fast_vars = {k: v for k, v in YAHOO_TICKERS.items()
                     if k in yahoo_df.columns or k.replace('^', '') in str(yahoo_df.columns)}
        slow_vars = {}
    else:
        # Join on date (FRED is typically lower frequency, so we forward-fill)
        combined = yahoo_df.join(fred_df, on='date', how='left', coalesce=True)

        # Forward fill FRED data (monthly/quarterly data)
        for col in fred_df.columns:
            if col != 'date':
                combined = combined.with_columns(
                    pl.col(col).forward_fill()
                )

    # Drop rows with any NaN (after forward fill)
    combined = combined.drop_nulls()

    print(f"\nCombined dataset: {len(combined)} rows, {len(combined.columns)} columns")

    return combined


def compute_returns(df: pl.DataFrame, price_cols: List[str]) -> pl.DataFrame:
    """Compute log returns for price columns.

    Args:
        df: DataFrame with price data
        price_cols: Columns containing prices to convert to returns

    Returns:
        df: DataFrame with returns instead of prices
    """
    result = df.clone()

    for col in price_cols:
        if col in result.columns:
            # Compute log returns
            result = result.with_columns(
                (pl.col(col).log() - pl.col(col).log().shift(1)).alias(col)
            )

    # Drop first row (NaN from differencing)
    result = result.slice(1)

    return result


def prepare_cfm_data(
    df: pl.DataFrame,
    fast_cols: List[str],
    slow_cols: List[str],
    compute_returns_for: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Prepare data for C-CFM training.

    Args:
        df: Combined DataFrame
        fast_cols: Column names for fast (market) variables
        slow_cols: Column names for slow (macro) variables
        compute_returns_for: Columns to convert from prices to returns

    Returns:
        X: Numpy array of shape (T, D)
        fast_names: List of fast variable names
        slow_names: List of slow variable names
    """
    # Compute returns if specified
    if compute_returns_for:
        df = compute_returns(df, compute_returns_for)

    # Extract columns (excluding date)
    all_cols = [c for c in fast_cols + slow_cols if c in df.columns]
    X = df.select(all_cols).to_numpy()

    fast_names = [c for c in fast_cols if c in df.columns]
    slow_names = [c for c in slow_cols if c in df.columns]

    return X.astype(np.float64), fast_names, slow_names


def create_sample_dataset() -> Tuple[np.ndarray, List[str], List[str]]:
    """Create a sample dataset without requiring API access.

    Generates synthetic data that mimics the structure of real
    economic/financial data for testing and demonstration.

    Returns:
        X: Numpy array of shape (1000, 20)
        fast_vars: List of fast variable names
        slow_vars: List of slow variable names
    """
    np.random.seed(42)
    n_samples = 1000

    # Simulate correlated market data
    # Fast variables (market, high frequency)
    fast_vars = [
        'SPY_ret', 'QQQ_ret', 'IWM_ret', 'EFA_ret',
        'VIX', 'TLT_ret', 'HYG_ret', 'LQD_ret',
        'GLD_ret', 'USO_ret', 'XLF_ret', 'XLE_ret'
    ]

    # Slow variables (macro, low frequency)
    slow_vars = [
        'Fed_Funds', 'Treasury_10Y', 'Treasury_2Y', 'Term_Spread',
        'HY_Spread', 'Unemployment', 'CPI', 'GDP_Growth'
    ]

    n_fast = len(fast_vars)
    n_slow = len(slow_vars)

    # Generate correlated fast variables (returns-like)
    # Use a factor model
    market_factor = np.random.randn(n_samples)
    vol_factor = np.abs(np.random.randn(n_samples))

    fast_data = np.zeros((n_samples, n_fast))
    for i in range(n_fast):
        if i == 4:  # VIX - special case
            fast_data[:, i] = 15 + 10 * vol_factor + 2 * np.random.randn(n_samples)
        else:
            beta = 0.5 + 0.5 * np.random.rand()
            fast_data[:, i] = beta * market_factor * 0.01 + 0.005 * np.random.randn(n_samples)

    # Generate slow variables (levels with trends)
    slow_data = np.zeros((n_samples, n_slow))

    # Fed Funds - mean reverting
    slow_data[:, 0] = 2.0 + np.cumsum(0.01 * np.random.randn(n_samples))

    # 10Y Treasury
    slow_data[:, 1] = slow_data[:, 0] + 1.5 + 0.5 * np.random.randn(n_samples)

    # 2Y Treasury
    slow_data[:, 2] = slow_data[:, 0] + 0.5 + 0.3 * np.random.randn(n_samples)

    # Term spread (10Y - 2Y)
    slow_data[:, 3] = slow_data[:, 1] - slow_data[:, 2]

    # HY Spread - counter-cyclical
    slow_data[:, 4] = 4.0 - 0.5 * market_factor + 0.5 * np.random.randn(n_samples)

    # Unemployment - persistent
    slow_data[:, 5] = 5.0 + np.cumsum(0.02 * np.random.randn(n_samples))
    slow_data[:, 5] = np.clip(slow_data[:, 5], 3, 12)

    # CPI - trending
    slow_data[:, 6] = 200 + 0.02 * np.arange(n_samples) + np.cumsum(0.1 * np.random.randn(n_samples))

    # GDP Growth - cyclical
    slow_data[:, 7] = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(n_samples) / 100) + 0.5 * np.random.randn(n_samples)

    # Combine
    X = np.hstack([fast_data, slow_data])

    print(f"Created sample dataset: {X.shape[0]} samples, {X.shape[1]} variables")
    print(f"  Fast variables: {n_fast}")
    print(f"  Slow variables: {n_slow}")

    return X, fast_vars, slow_vars


if __name__ == "__main__":
    # Test the data fetcher
    print("Testing data fetcher...\n")

    # First, try synthetic data (always works)
    print("=" * 50)
    print("Creating synthetic sample dataset")
    print("=" * 50)
    X, fast, slow = create_sample_dataset()
    print(f"Shape: {X.shape}")
    print(f"Fast vars: {fast}")
    print(f"Slow vars: {slow}")

    # Then try real data (requires API access)
    print("\n" + "=" * 50)
    print("Attempting to fetch real data...")
    print("=" * 50)
    try:
        df, fast_dict, slow_dict = fetch_all_data()
        print("\nData preview:")
        print(df.head())
    except Exception as e:
        print(f"Could not fetch real data: {e}")
        print("This is expected if yfinance/fredapi are not installed or no API key is set.")
