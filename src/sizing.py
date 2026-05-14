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
