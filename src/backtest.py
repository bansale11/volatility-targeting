"""
Simulation engine for the vol-targeted strategy vs buy-and-hold.

Design principles:
  - Thin: accepts pre-computed weights and returns; all estimation logic
    lives in volatility.py and sizing.py so it can be tested independently.
  - Transaction costs proportional to absolute weight change (turnover).
    This matches how institutional desks account for market impact on deep,
    liquid index products like SPY.  More granular models (bid-ask spread,
    market impact) require intraday data and are not warranted here.
  - Both strategies are evaluated over the same date range (post-warmup)
    so the equity curves start from a common reference point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_BPS: float = 1.5   # basis points per unit of |Δweight|


def run_backtest(
    log_returns: pd.Series,
    weights: pd.Series,
    cost_bps: float = DEFAULT_COST_BPS,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Simulate the vol-targeted strategy alongside a buy-and-hold benchmark.

    Parameters
    ----------
    log_returns     : Daily log returns of the underlying.  NaN on the first
                      row is expected and handled.
    weights         : Position weights from src/sizing.py (already lagged
                      by one day).  NaN rows are excluded from the backtest.
    cost_bps        : Transaction cost in basis points per unit of |Δweight|.
                      At 1.5 bps, rebalancing from w=0.8 to w=1.2 costs
                      0.4 × 0.00015 = 0.006% of capital.
    initial_capital : Starting portfolio value (default 1.0).

    Returns
    -------
    DataFrame indexed by date with columns:
        log_return          raw underlying log return
        weight              applied position weight (NaN during warmup)
        turnover            |weight_t − weight_{t-1}|
        cost                transaction cost as fraction of capital
        strat_gross_return  weight × simple_return (pre-cost)
        strat_net_return    gross return − cost
        bah_return          buy-and-hold simple return
        strat_equity        cumulative vol-targeted portfolio value
        bah_equity          cumulative buy-and-hold portfolio value

    Notes
    -----
    We convert log returns to simple returns for equity accumulation.  For
    daily holding periods this approximation is negligible (the difference
    is ½σ² ≈ 0.001% per day for a 15%-vol strategy), but using simple
    returns avoids compounding artefacts when multiplying by a weight that
    can vary between 0 and 2.
    """
    simple_returns = np.exp(log_returns) - 1

    if weights.first_valid_index() is None:
        raise ValueError("weights series has no valid (non-NaN) values")

    combined = pd.DataFrame(
        {
            "log_return": log_returns,
            "simple_return": simple_returns,
            "weight": weights,
        }
    ).dropna(subset=["weight", "simple_return"])   # guard both: NaN weight = warmup;
                                                   # NaN simple_return = unfilled gap

    cost_per_unit = cost_bps / 10_000

    # Build turnover in a separate Series to avoid SettingWithCopyWarning;
    # the first row assumes the portfolio was flat before the backtest starts.
    turnover = combined["weight"].diff().abs()
    turnover.iloc[0] = combined["weight"].iloc[0]

    combined = combined.assign(
        turnover=turnover,
        cost=lambda df: df["turnover"] * cost_per_unit,
        strat_gross_return=lambda df: df["weight"] * df["simple_return"],
    ).assign(
        strat_net_return=lambda df: df["strat_gross_return"] - df["cost"],
        bah_return=lambda df: df["simple_return"],
    ).assign(
        strat_equity=lambda df: initial_capital * (1 + df["strat_net_return"]).cumprod(),
        bah_equity=lambda df: initial_capital * (1 + df["bah_return"]).cumprod(),
    )

    return combined.drop(columns=["simple_return"])
