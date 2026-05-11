# Alpha Research Engine — Systematic Trading Lab (Project #5)

An academic-grade, strategy-driven research environment for **signal design**, **vectorized backtesting**, and **publication-quality diagnostics**.

The goal is not “a backtest that looks good,” but a workflow that can withstand quant interview scrutiny:
- explicit model assumptions,
- robust diagnostics (tail risk, drawdowns, rolling metrics),
- transaction costs and turnover,
- benchmark-relative evaluation.

## Strategies implemented

1) **Dual Class Arbitrage (Cointegration Pairs)**
- Learns a cointegrating relationship in log prices using:
  - Engle–Granger two-step test (`coint`)
  - Johansen trace test (`coint_johansen`)
  - ADF stationarity test on residuals (\(\varepsilon_t\))
- Trades mean reversion via a rolling Z-score with **entry/exit hysteresis**.

2) **Sector-Based Pairs Trading**
- Selects a candidate pair within a sector universe (tech/consumer/energy/banks) by **best cointegration evidence**, then applies the same cointegration workflow.

3) **Bollinger Bands Mean Reversion**
- Uses SMA ± 2σ bands; enters on band touches and exits near mean reversion.

4) **Reinforcement Learning (Experimental)**
- A minimal tabular Q-learning baseline on discretized z-score states.
- Included as a portfolio talking point (not marketed as production-grade).

## Backtester + metrics (what employers look for)

Backtesting is fully vectorized with:
- **Transaction costs** (bps) applied via **turnover**.
- **Benchmark-relative** series: tracking error and information ratio.

Metrics computed:
- **Sharpe**, **Information Ratio**, **Sortino**
- **Skewness**, **Kurtosis** (tail shape)
- **MAE / MFE** (trade-run adverse/favorable excursion)
- **Beta to benchmark**
- **Information Coefficient (IC)**: correlation of signal vs next-day returns

## Report (4-panel Plotly)

The generated HTML report includes:
- Cumulative returns vs benchmark
- Rolling volatility & rolling Sharpe
- Underwater drawdown (filled)
- Monthly returns heatmap

Plus a centered scientific description block and a metric card grid.

## Repository structure

```
alpha-research/
  alpha_research/          # library: data, strategies, backtester, report
  run_alpha_report.py      # CLI entrypoint: JSON config -> HTML report + JSON payload
  requirements.txt
  PROJECT_OVERVIEW.md
```

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python3 run_alpha_report.py \
  '{"strategy":"dual_class_arb","start":"2022-01-01","end":"2024-01-01","asset_a":"GOOG","asset_b":"GOOGL","benchmark":"SPY","lookback":60,"entry_z":2.0,"exit_z":0.5,"transaction_cost_bps":2.0}' \
  ./output/report.html
```

## Configuration reference (JSON)

Common fields:
- `strategy`: `dual_class_arb | sector_pairs | bollinger_bands | rl_mean_reversion`
- `start`, `end` (YYYY-MM-DD)
- `benchmark`: `SPY | QQQ`
- `lookback` (int), `transaction_cost_bps` (float)

Pairs-only:
- `entry_z`, `exit_z`, `asset_a`, `asset_b` or `sector`

Single-asset:
- `ticker`

Risk controls (optional):
- `stop_loss_pct` (e.g. `0.05`)
- `take_profit_pct` (e.g. `0.08`)


