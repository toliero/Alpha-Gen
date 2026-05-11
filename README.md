# Alpha Research Engine (Project #5)

Multi-strategy systematic trading research environment with vectorized backtesting and a Plotly report:

- Dual Class Arbitrage (cointegration-tested pairs)
- Sector-based pairs trading (auto-select by cointegration evidence)
- Bollinger Bands mean reversion
- Experimental RL mean reversion (tabular Q-learning demo)

## Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 run_alpha_report.py '{"strategy":"dual_class_arb","start":"2022-01-01","end":"2024-01-01","asset_a":"GOOG","asset_b":"GOOGL","benchmark":"SPY","lookback":60,"entry_z":2.0,"exit_z":0.5,"transaction_cost_bps":2.0}' ./output/report.html
```

