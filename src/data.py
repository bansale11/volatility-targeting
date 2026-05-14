"""
Data pipeline: download, cache, validate, and prepare daily OHLCV for vol targeting.

Adjusted prices are used throughout so that splits and dividends do not appear
as artificial return shocks.  A local parquet cache avoids re-hitting the
network on reruns; the cache key encodes the ticker and start date so stale
files are not silently reused after a parameter change.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DOWNLOAD_RETRIES = 4
_RETRY_BACKOFF_S = 30   # seconds to wait between retries; doubles each attempt


def _download_with_retry(
    ticker: str,
    start: str,
    end: str | None,
    retries: int = _DOWNLOAD_RETRIES,
    backoff: int = _RETRY_BACKOFF_S,
) -> pd.DataFrame:
    """
    Download OHLCV from yfinance with exponential-backoff retry.

    Yahoo Finance rate-limits unauthenticated requests.  A brief pause between
    attempts is usually sufficient to clear the limit.  We raise only after all
    retries are exhausted so that transient 429 errors don't abort the notebook.
    """
    for attempt in range(retries):
        logger.info("Downloading %s (attempt %d/%d)", ticker, attempt + 1, retries)
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if not raw.empty:
            return raw
        wait = backoff * (2 ** attempt)
        if attempt < retries - 1:
            logger.warning(
                "yfinance returned empty data for %s (rate limit?). "
                "Retrying in %d s …", ticker, wait,
            )
            time.sleep(wait)
    raise ValueError(
        f"yfinance returned no data for '{ticker}' after {retries} attempts. "
        "Check your internet connection or try again in a few minutes."
    )


DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "data_cache"
_MAX_FFILL_DAYS = 3       # forward-fill gaps no longer than this many trading days
_LARGE_MOVE_THRESH = 0.25  # warn (not drop) on |return| > 25%


def load_ohlcv(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Return a daily OHLCV + log-return DataFrame for *ticker*.

    The returned frame has a DatetimeIndex and columns:
        Open, High, Low, Close, Volume, log_return

    where ``Close`` is the split- and dividend-adjusted price (via yfinance's
    ``auto_adjust=True``).  ``log_return`` is NaN on the first row; all
    downstream estimators handle this correctly.

    Parameters
    ----------
    ticker        : Ticker symbol recognised by yfinance (default "SPY")
    start         : ISO date string for the first bar to download
    end           : ISO date string for the last bar (default: today)
    cache_dir     : Directory for parquet cache files
    force_download: Re-download even if a cached file exists
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start_slug = start.replace("-", "")
    end_slug   = (end or "latest").replace("-", "")
    cache_file = cache_dir / f"{ticker}_{start_slug}_{end_slug}.parquet"

    if cache_file.exists() and not force_download:
        logger.info("Loading %s from cache: %s", ticker, cache_file)
        df = pd.read_parquet(cache_file)
    else:
        raw = _download_with_retry(ticker, start=start, end=end)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        # Validate before writing the cache: if validation raises, the partial
        # file is removed so the next run retries the download rather than
        # silently loading bad data.
        try:
            df = _validate_and_clean(df, ticker)
        except Exception:
            cache_file.unlink(missing_ok=True)
            raise
        df.to_parquet(cache_file)
        logger.info("Saved to cache: %s", cache_file)
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df

    df = _validate_and_clean(df, ticker)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df



def _validate_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Validate raw OHLCV, forward-fill short gaps, and warn on anomalies.

    We forward-fill gaps of 1–3 trading days (data vendor glitches, observed
    holidays) but raise on longer gaps where genuine market suspension is
    possible.  Extreme daily moves (>25%) are flagged rather than removed
    because they may be legitimate — 2020-03-16 SPY fell ~12%.
    """
    if df.empty:
        raise ValueError(f"Empty DataFrame for {ticker}")

    # Coerce index to DatetimeIndex if needed (parquet round-trip safety)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    n_missing = df["Close"].isna().sum()
    if n_missing > 0:
        logger.warning(
            "%d missing Close values for %s — forward-filling Close up to %d days",
            n_missing, ticker, _MAX_FFILL_DAYS,
        )
        # Fill only Close; forward-filling Volume and OHLC columns would
        # produce misleading zero-range bars for GK/YZ estimators and
        # artificially inflate volume in downstream analysis.
        df["Close"] = df["Close"].ffill(limit=_MAX_FFILL_DAYS)
        remaining = df["Close"].isna().sum()
        if remaining > 0:
            raise ValueError(
                f"{remaining} unfillable missing Close prices for {ticker}. "
                f"Gaps exceed the {_MAX_FFILL_DAYS}-day forward-fill limit."
            )

    raw_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    large_moves = raw_returns[raw_returns.abs() > _LARGE_MOVE_THRESH]
    if not large_moves.empty:
        logger.warning(
            "%d days with |return| > %.0f%% for %s: %s",
            len(large_moves),
            _LARGE_MOVE_THRESH * 100,
            ticker,
            [str(d.date()) for d in large_moves.index],
        )

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    if n_years < 15:
        logger.warning(
            "Only %.1f years of data for %s; 15+ years recommended for vol targeting",
            n_years, ticker,
        )

    return df
