"""
Performance metrics for the volatility targeting backtest.

All annualisation uses 252 trading days.  Risk-free rate defaults to 0%;
pass rf > 0 to compute excess-return Sharpe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Geometric (CAGR) annualised return.

    Expects simple (not log) daily returns.  NaN values are dropped before
    both the row count and the cumulative product so that the annualisation
    exponent (periods_per_year / n) uses the actual number of trading days
    in the series, not the total index length including any NaN warmup rows.
    """
    clean = returns.dropna()
    n = len(clean)
    if n == 0:
        return np.nan
    total_growth = (1 + clean).prod()
    return float(total_growth ** (periods_per_year / n) - 1)


def annualized_vol(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised volatility of simple daily returns (sample std, ddof=1)."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualised Sharpe ratio.

    (annualised_return − risk_free_rate) / annualised_vol

    With rf=0 this is equivalent to the information ratio relative to cash.
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_v = annualized_vol(returns, periods_per_year)
    if ann_v == 0:
        return np.nan
    return (ann_ret - risk_free_rate) / ann_v


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Rolling drawdown from the running maximum of the equity curve.

    Returns non-positive values; −0.20 means a 20% loss from the prior peak.
    """
    running_max = equity_curve.cummax()
    return (equity_curve / running_max - 1).rename("drawdown")


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (non-positive scalar)."""
    return float(drawdown_series(equity_curve).min())


def calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Calmar ratio: annualised return divided by absolute max drawdown.

    Higher is better.  A natural complement to Sharpe for drawdown-conscious
    strategies: a strategy that improves the Calmar without hurting Sharpe
    has genuinely improved the risk-adjusted profile.
    """
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return np.nan
    return ann_ret / abs(mdd)


def summary_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    label: str = "",
    risk_free_rate: float = 0.0,
) -> dict[str, object]:
    """Return a single-strategy performance dict."""
    return {
        "Strategy": label,
        "Ann. Return (%)": round(annualized_return(returns) * 100, 2),
        "Ann. Vol (%)": round(annualized_vol(returns) * 100, 2),
        "Sharpe": round(sharpe_ratio(returns, risk_free_rate), 3),
        "Max DD (%)": round(max_drawdown(equity_curve) * 100, 2),
        "Calmar": round(calmar_ratio(returns, equity_curve), 3),
    }


def build_summary_table(
    results: dict[str, tuple[pd.Series, pd.Series]],
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Build a multi-strategy performance comparison table.

    Parameters
    ----------
    results : dict mapping label → (returns_series, equity_curve_series)

    Returns
    -------
    DataFrame with one row per strategy and standard metric columns,
    indexed by strategy label.
    """
    rows = [
        summary_metrics(ret, eq, label=label, risk_free_rate=risk_free_rate)
        for label, (ret, eq) in results.items()
    ]
    return pd.DataFrame(rows).set_index("Strategy")
