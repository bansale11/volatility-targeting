"""
Volatility estimators returning annualized realized-volatility series.

Four estimators are provided, in increasing sophistication:

1. rolling_std_vol  — close-to-close sample std; the market standard for
                      daily risk management.  Simple, interpretable, and
                      unbiased.  Downside: equal weight on all observations
                      in the window.

2. ewma_vol         — RiskMetrics 1994 exponential smoother (λ=0.94).
                      Down-weights old observations geometrically, which makes
                      it faster to react to vol regime shifts than a rolling
                      window of comparable span.  The effective half-life at
                      λ=0.94 is ~11 days.

3. garman_klass_vol — uses the full daily OHLC range.  About 5× more
                      efficient than close-to-close for the same number of
                      days, but ignores overnight gaps.  Acceptable for
                      large-cap ETFs (SPY overnight gaps are small).

4. yang_zhang_vol   — combines the overnight gap (open vs prior close),
                      intraday range (Rogers-Satchell), and close-to-close
                      components.  Minimum-variance unbiased estimator for
                      GBM with overnight jumps (Yang & Zhang 2000).  The
                      most information-efficient of the four.

All functions are pure (no I/O) and return a pd.Series of the same length
as the input with NaN during the warmup period.  Annualisation uses √252.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALISE = np.sqrt(252)


def rolling_std_vol(
    returns: pd.Series,
    window: int = 20,
    ddof: int = 1,
) -> pd.Series:
    """
    Annualised close-to-close rolling volatility.

    ddof=1 is the sample (unbiased) standard deviation.  We pass it
    explicitly because it is easy to miss and the difference matters most
    for short windows — the 20-day estimator with ddof=0 overstates vol
    by a factor of sqrt(20/19) ≈ 1.026 relative to ddof=1.
    """
    return (
        returns
        .rolling(window=window, min_periods=window)
        .std(ddof=ddof)
        .mul(ANNUALISE)
        .rename(f"vol_roll{window}")
    )


def ewma_vol(
    returns: pd.Series,
    lambda_: float = 0.94,
    min_periods: int = 20,
) -> pd.Series:
    """
    Annualised EWMA volatility — RiskMetrics 1994 convention.

    Recursive variance: σ²_t = λ·σ²_{t-1} + (1−λ)·r²_t

    Equivalent to an exponentially weighted mean of squared returns with
    decay factor λ.  min_periods suppresses the noisy early estimates that
    result from a cold-start (a single extreme return would otherwise
    dominate the first few vol readings).

    Uses pandas ewm(adjust=False) which implements the recursion exactly,
    initialising at the first squared return.
    """
    alpha = 1.0 - lambda_
    variance = (
        returns
        .pow(2)
        .ewm(alpha=alpha, min_periods=min_periods, adjust=False)
        .mean()
    )
    return (
        np.sqrt(variance)
        .mul(ANNUALISE)
        .rename(f"vol_ewma{lambda_}")
    )


def garman_klass_vol(
    ohlc: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Annualised Garman-Klass range-based volatility.

    Per-day variance:
        σ²_GK = 0.5·[ln(H/L)]² − (2·ln2−1)·[ln(C/O)]²

    This is the minimum-variance estimator for intraday data when there are
    no overnight gaps.  For SPY, overnight gaps are a second-order effect,
    so GK is a reasonable approximation.  Do not use this for single stocks
    with frequent gaps (e.g., around earnings).
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])

    daily_var = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    annual_var = (
        daily_var
        .rolling(window=window, min_periods=window)
        .mean()
        .clip(lower=0)   # GK can go negative when High==Low (zero-range bars)
        .mul(252)
    )
    return np.sqrt(annual_var).rename(f"vol_gk{window}")


def yang_zhang_vol(
    ohlc: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Annualised Yang-Zhang volatility (Yang & Zhang 2000).

    Combines three independent variance components:

        σ²_YZ = σ²_overnight + k·σ²_close + (1−k)·σ²_RS

    where:
        σ²_overnight = sample var of ln(Open_t / Close_{t-1})
        σ²_close     = sample var of ln(Close_t / Close_{t-1})
        σ²_RS        = mean Rogers-Satchell intraday variance
                     = E[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]
        k            = 0.34 / (1.34 + (n+1)/(n−1))   (optimal weighting)

    YZ is the minimum-variance unbiased estimator for geometric Brownian
    motion with overnight gaps, making it the best choice when data quality
    is high (all four prices available and free of errors).

    The variance can theoretically turn slightly negative due to floating-
    point arithmetic; we clip to zero before taking the square root.
    """
    log_open  = np.log(ohlc["Open"])
    log_high  = np.log(ohlc["High"])
    log_low   = np.log(ohlc["Low"])
    log_close = np.log(ohlc["Close"])
    log_close_prev = log_close.shift(1)

    overnight = log_open  - log_close_prev   # ln(Open_t / Close_{t-1})
    close_ret = log_close - log_close_prev   # ln(Close_t / Close_{t-1})

    # Rogers-Satchell per-day variance (drift-free, no look-ahead)
    rs_daily = (
        (log_high - log_close) * (log_high - log_open)
        + (log_low - log_close) * (log_low - log_open)
    )

    # pandas .var(ddof=1) computes mean and deviations over the same window
    # in a single pass, which is both numerically correct and look-ahead-free.
    # The earlier hand-rolled approach subtracted a rolling mean computed at
    # position t, then applied a second rolling sum — this centred historical
    # deviations on a mean that included future data relative to those rows.
    def _roll_sample_var(s: pd.Series) -> pd.Series:
        return s.rolling(window=window, min_periods=window).var(ddof=1)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_overnight = _roll_sample_var(overnight)
    var_close     = _roll_sample_var(close_ret)
    var_rs        = rs_daily.rolling(window=window, min_periods=window).mean()

    yz_var = (var_overnight + k * var_close + (1 - k) * var_rs).clip(lower=0)
    return np.sqrt(yz_var * 252).rename(f"vol_yz{window}")
