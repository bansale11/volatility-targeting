"""
Position weight calculation for volatility targeting.

The core idea: if realised volatility doubles, cut the position in half so
that the ex-ante risk contribution stays roughly constant.  The weight is
therefore the ratio of the target volatility to the most recent measured
volatility.

    w_t = σ_target / σ_{t−1}

The −1 lag is the single most important implementation detail.  The weight
applied to day-t returns must be computed from volatility data available at
the market open of day t, which means the vol estimate can only use data
through the close of day t−1.  Failing to lag introduces look-ahead bias
that would not be present in live trading and inflates the Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TARGET_VOL: float = 0.15   # 15% annualised
DEFAULT_MAX_LEVERAGE: float = 2.0  # 2× notional cap


def compute_weights(
    realized_vol: pd.Series,
    target_vol: float = DEFAULT_TARGET_VOL,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> pd.Series:
    """
    Return daily position weights w_t = target_vol / σ_{t−1}, capped at
    max_leverage and floored at 0 (long-only).

    Parameters
    ----------
    realized_vol  : Annualised daily vol series from src/volatility.py.
                    NaN values during the warmup period propagate correctly.
    target_vol    : Desired annualised portfolio volatility (default 15%)
    max_leverage  : Upper bound on position weight, expressed as a multiple
                    of capital (default 2×).  In calm markets, σ_realised
                    can fall well below σ_target, driving the raw weight
                    above 1.  The cap prevents extreme leverage from
                    compounding drawdowns when volatility snaps back.

    Returns
    -------
    pd.Series of daily weights, NaN during the warmup period, aligned to
    the same index as realized_vol.
    """
    # The lag is the line that prevents look-ahead bias.
    # vol on day t is only known at close of day t, so we can only act on it
    # from day t+1 onwards.
    lagged_vol = realized_vol.shift(1)

    raw_weight = target_vol / lagged_vol
    weights = raw_weight.clip(lower=0.0, upper=max_leverage)
    weights.name = "weight"
    return weights


def compute_riskparity_weights(
    returns_df: pd.DataFrame,
    target_vol: float = DEFAULT_TARGET_VOL,
    window: int = 60,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
) -> pd.DataFrame:
    """
    Inverse-volatility risk parity weights scaled to a target portfolio vol.

    For each day t (using only data through t−1 after the final lag):

    1. Rolling per-asset vol  →  inverse-vol weights normalised to sum = 1
    2. Rolling pairwise covariances  →  portfolio vol  σ_p = √(w'Σw)
    3. Scale: w_final = w × (target_vol / σ_p), capped at max_leverage
    4. One-day lag (no look-ahead)

    Parameters
    ----------
    returns_df   : DataFrame of log returns, one column per asset.
    target_vol   : Desired annualised portfolio vol (default 15%).
    window       : Rolling window in trading days (default 60).
    max_leverage : Upper bound on sum of absolute weights (default 2×).

    Returns
    -------
    DataFrame with the same columns as returns_df, NaN during warmup,
    lagged by one day so weights on day t use only data through day t−1.
    """
    tickers = list(returns_df.columns)

    # Step 1: Rolling per-asset vol and inverse-vol normalised weights
    vols = (
        returns_df
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
        .mul(np.sqrt(252))
    )
    inv_vol = 1.0 / vols
    weights_raw = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Step 2: Portfolio variance via rolling pairwise covariances (vectorised)
    port_var = pd.Series(0.0, index=returns_df.index)
    for t1 in tickers:
        for t2 in tickers:
            cov_series = (
                returns_df[t1]
                .rolling(window=window, min_periods=window)
                .cov(returns_df[t2], ddof=1)
                .mul(252)
            )
            port_var = port_var + weights_raw[t1] * weights_raw[t2] * cov_series

    portfolio_vol = np.sqrt(port_var.clip(lower=0))

    # Step 3: Scale to target vol, cap leverage
    scale = (target_vol / portfolio_vol).clip(upper=max_leverage)
    weights_scaled = weights_raw.mul(scale, axis=0)

    total = weights_scaled.sum(axis=1)
    overflow = total > max_leverage
    if overflow.any():
        scale_down = (max_leverage / total).where(overflow, other=1.0)
        weights_scaled = weights_scaled.mul(scale_down, axis=0)

    # Step 4: Lag by one day — mandatory, mirrors compute_weights()
    return weights_scaled.shift(1)


def trend_filter(
    prices: pd.DataFrame | pd.Series,
    window: int = 200,
) -> pd.DataFrame | pd.Series:
    """
    Binary trend signal: 1 (in) if price > window-day moving average, else 0 (out).

    Lagged by one day so the signal on day t uses only prices through day t−1.
    Multiply risk parity (or any) weights by this signal to zero out positions
    in assets that are in a downtrend.

    Parameters
    ----------
    prices : Close price series or DataFrame (one column per asset).
    window : Look-back window in trading days (default 200, ~10 months).

    Returns
    -------
    Same shape as prices, dtype float (0.0 or 1.0), NaN during warmup.
    """
    ma = prices.rolling(window=window, min_periods=window).mean()
    signal = (prices > ma).astype(float)
    signal[prices.isna() | ma.isna()] = float("nan")
    return signal.shift(1)
