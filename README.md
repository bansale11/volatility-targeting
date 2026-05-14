# Volatility Targeting

A single-asset volatility-targeting strategy implemented on SPY (2005–present), with four realised-volatility estimators, a full daily backtest with transaction costs, and standard risk/return diagnostics.

Run `notebook.ipynb` to reproduce all results. The performance table below is populated by the backtest.

---

## Performance (run notebook to populate)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| Vol Target – Roll 20d | 12.48% | 16.32% | **0.765** | −36.50% | 0.342 |
| Vol Target – Roll 60d | 11.39% | 16.13% | 0.706 | −34.05% | 0.334 |
| Vol Target – EWMA (λ=0.94) | 11.54% | 15.64% | 0.738 | −35.06% | 0.329 |
| Vol Target – Yang-Zhang 20d | 12.22% | 26.78% | 0.456 | −74.81% | 0.163 |
| Buy-and-Hold | 11.14% | 19.05% | 0.585 | −55.19% | 0.202 |

*SPY 2005-04-01 to 2026-05-13 (21 years). Transaction costs: 1.5 bps per unit of |Δweight|. Risk-free rate: 0%.*

The Yang-Zhang estimator fails in this backtest: it collapses to near-zero on dividend-adjustment artefact days in the yfinance OHLCV data, driving the weight to the 2× cap just before large moves. Close-to-close estimators (rolling std, EWMA) are immune because they depend only on the Close column, which is consistently adjusted. See the Conclusions section of the notebook for a full discussion.

---

## Why This Works: Volatility Persistence

The strategy rests on a single empirical fact: daily equity returns are approximately unpredictable, but their *variance* is not.  This asymmetry was formalised by Engle (1982) with ARCH and extended by Bollerslev (1986) with GARCH, but it is visible in any autocorrelation plot.  The first-order autocorrelation of SPY daily returns is close to zero; the first-order autocorrelation of squared returns (a proxy for daily variance) is typically 0.15–0.30 and remains significantly positive for lags of several weeks.

The intuition is that high-volatility environments cluster.  A 3% down day is much more likely to be followed by another high-volatility day than a calm one.  This persistence is the precondition for vol targeting to have any predictive value at all.  If volatility were i.i.d., yesterday's realised vol would be no better than the long-run average as an estimate of tomorrow's risk, and the strategy would reduce to a constant leverage fraction with extra turnover cost.

The GARCH(1,1) model makes this quantitative.  The conditional variance follows:

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

For a fitted SPY model the persistence parameter (α + β) is typically 0.97–0.99, implying that a variance shock has a half-life of several months.  The rolling-window and EWMA estimators used here are crude but theoretically grounded approximations to this conditional variance.

## Risk Non-Stationarity in Fixed-Notional Portfolios

A fund that maintains a constant dollar notional in a single underlying is implicitly running time-varying risk.  When SPY volatility doubles — as it did in late 2008 and again in March 2020 — the dollar loss for the same notional position also doubles.  For a leveraged book this compounds: a 2x fixed-notional position in SPY during the GFC experienced drawdowns close to 100%.

Vol targeting reframes the allocation problem.  Instead of asking "how many dollars do I hold?", the manager asks "how many dollars of *risk* do I hold?"  The answer is kept constant by construction.  Concretely, if realised vol is 30% and the target is 15%, the portfolio is run at 0.5x notional.  If vol falls to 10%, the portfolio runs at 1.5x.  The result is a risk budget that is stable through time rather than one that silently balloons in crises.

For a multi-manager fund or an allocation committee, this property is operationally important: a vol-targeted allocation consumes a predictable slice of the portfolio's risk budget regardless of the market regime.  A fixed-notional allocation does not.

## Honest Assessment of the Sharpe Improvement

The Sharpe ratio of the vol-targeted strategy is typically within ±0.1 of buy-and-hold over long samples.  This is not a bug; it is the expected result.  If realised vol were a perfect predictor of next-period vol, vol targeting would exactly equalise risk per unit of return, and the Sharpe should be preserved.  In practice, the noise in the vol signal, the one-day lag, the leverage cap, and transaction costs introduce small frictions that push the Sharpe slightly in either direction depending on the period.

The substantive benefit of vol targeting is not Sharpe improvement but two related properties:

1. **Drawdown compression.** By reducing exposure precisely when risk is highest, the strategy cuts peak-to-trough drawdowns substantially (typically 30–50% less severe than buy-and-hold for the GFC). This is a meaningful improvement for investors who face margin constraints, redemption risk, or loss limits.

2. **Risk budget predictability.** The distribution of portfolio volatility is compressed around the target, which matters for risk management, leverage allocation, and investor communication. A strategy that promises 15% volatility and delivers 13–17% most of the time is vastly preferable to one that delivers 8–80%.

Overclaiming a Sharpe improvement is a common mistake in strategy presentations.  The correct claim is that vol targeting achieves a more stable risk profile at approximately the same return per unit of *average* risk.

## Tradeoffs and Failure Modes

**Turnover cost.** The strategy rebalances daily, and in volatile regimes turnover can be high.  At 1.5 bps per unit of weight change, the annualised cost drag is typically 10–30 bps, small but non-trivial in a low-fee environment.  Weekly rebalancing roughly halves the cost at modest statistical cost.

**Performance drag in low-vol bull markets.** When realised vol is sustainably below target — as in 2013–2019 — the raw weight exceeds the 2x cap, and the leverage constraint is binding.  The strategy cannot fully participate in the rally because it is already at its notional ceiling.  This is a feature from a risk-management perspective (the cap prevents catastrophic exposure if vol snaps back), but it registers as underperformance versus an unconstrained buy-and-hold.

**Lookback sensitivity.** The two most common window choices — 20-day and 60-day rolling std — produce meaningfully different weights in transition periods.  The 60-day estimator carries more exposure into the early stages of a vol spike; the 20-day estimator deleverages faster but also re-leverages faster during the recovery.  There is no universally optimal window: it depends on the autocorrelation structure of the specific vol regime, which varies over time.

**Volatility regime shifts: February 2018.** The canonical stress test for vol targeting is the February 5–9, 2018 "Volmageddon" episode.  The VIX doubled in a single session (from 17 to 37), and many volatility-selling strategies were liquidated.  Vol-targeting strategies were not immune: the one-day lag meant the weight on February 5 was calibrated to January's calm, and the position was sized above 1x going into the spike.  The strategy correctly deleveraged afterward but did not anticipate the jump.  This episode demonstrates that vol targeting controls *conditional* risk (given the information available at t-1) but does not hedge against vol jumps.  Jump-diffusion extensions (e.g. adding a VIX-based overlay) address this at additional complexity cost.

## Extension to Multi-Asset Portfolios

The single-asset formulation extends naturally to a portfolio of *n* assets.  Instead of dividing the target vol by a scalar realised vol, the weight vector is determined by:

    w_t = σ_target · Σ_{t-1}^{-1} · μ_target / sqrt(μ_target' · Σ_{t-1}^{-1} · μ_target)

where Σ_{t-1} is the estimated covariance matrix.  In the simplest case where the target direction is equal-weight, this reduces to *inverse-volatility weighting* scaled to hit the portfolio vol target.  The more general version with a full covariance matrix is the basis of risk-parity allocations (Bridgewater's All Weather, AQR's Risk Parity fund).

The key implementation challenge shifts from scalar vol estimation to covariance estimation.  Sample covariance matrices of dimension *n* estimated on *T* days are noisy when *T/n* is small (a common condition in practice: 60 days of data for 10 assets gives a rank-deficient matrix).  Common remedies include Ledoit-Wolf shrinkage, factor-model covariance (1-factor CAPM being the simplest), and exponential weighting.

The relationship to risk parity is close but not identical.  Pure risk parity targets equal *marginal* risk contribution from each asset; vol targeting targets a fixed *portfolio-level* volatility without constraining how individual assets contribute.  In practice, both approaches allocate heavily to low-vol assets (bonds, gold) relative to a market-cap benchmark, and both benefit from the same volatility-persistence property that makes single-asset vol targeting work.

---

## Repository Structure

```
volatility-targeting/
├── README.md
├── requirements.txt
├── notebook.ipynb          # narrative + charts + verification
└── src/
    ├── __init__.py
    ├── data.py             # download, cache, validate, log returns
    ├── volatility.py       # rolling std, EWMA, Garman-Klass, Yang-Zhang
    ├── sizing.py           # w_t = sigma_target / sigma_{t-1}, lag + cap
    ├── backtest.py         # simulation engine + transaction costs
    └── metrics.py          # Sharpe, max drawdown, Calmar, summary table
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name python3 --display-name "Python 3"
jupyter nbconvert --execute notebook.ipynb --to notebook --output notebook_executed.ipynb
```

The `ipykernel install` step registers the venv's Python as a Jupyter kernel before nbconvert executes the notebook. This is a one-time setup per environment.

## Design Notes

- **One-day lag is mandatory.** `compute_weights()` calls `.shift(1)` on the vol series. Any code that removes this lag introduces look-ahead bias that inflates the Sharpe ratio and would not survive in live trading.
- **ddof=1 throughout.** All rolling std computations use the sample (unbiased) estimator. For a 20-day window, ddof=0 overstates volatility by a factor of sqrt(20/19) ≈ 1.026 — small but compounding.
- **Parquet cache.** The cache key encodes the ticker and start date (`SPY_20050101.parquet`). Changing `START_DATE` or `TICKER` automatically triggers a fresh download.
- **Pure functions.** All `src/` functions are free of side effects. Data download is isolated to `load_ohlcv()`; everything downstream takes DataFrames and returns DataFrames.
