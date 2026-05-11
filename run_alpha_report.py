#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import pandas as pd

from alpha_research.backtester import run_backtest
from alpha_research.data import fetch_close
from alpha_research.report import build_report_payload, performance_report
from alpha_research.strategies import run_strategy
from alpha_research.types import BacktestConfig


def _parse(argv: list[str]) -> tuple[BacktestConfig, str]:
    if len(argv) < 3:
        print("Usage: python3 run_alpha_report.py <config_json> <output_html>", file=sys.stderr)
        sys.exit(2)
    cfg_raw = json.loads(argv[1])
    out = argv[2]
    cfg = BacktestConfig(**cfg_raw)
    return cfg, out


def main() -> None:
    cfg, out_html = _parse(sys.argv)

    # Determine tickers required for the chosen strategy
    tickers: list[str] = []
    if cfg.strategy in ("dual_class_arb", "sector_pairs"):
        if cfg.asset_a:
            tickers.append(cfg.asset_a)
        if cfg.asset_b:
            tickers.append(cfg.asset_b)
    if cfg.strategy in ("bollinger_bands", "rl_mean_reversion"):
        if cfg.ticker:
            tickers.append(cfg.ticker)

    # For sector_pairs we need all candidate pairs in that sector; easiest is to pull a superset.
    if cfg.strategy == "sector_pairs":
        # matches the universe defined in strategies.py
        sector_universe = {
            "tech": ["MSFT", "AAPL", "GOOG", "META", "NVDA", "AMD"],
            "consumer": ["PEP", "KO", "MCD", "SBUX"],
            "energy": ["XOM", "CVX"],
            "banks": ["JPM", "BAC", "GS", "MS"],
        }
        tickers = sector_universe.get((cfg.sector or "tech").strip().lower(), tickers)

    bench = cfg.benchmark
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t]))
    if not tickers:
        raise ValueError("No tickers resolved from config.")

    md = fetch_close(tickers + [bench], start=cfg.start, end=cfg.end)
    close = md.close[tickers]
    bench_close = md.close[bench]

    strat = run_strategy(close=md.close, cfg=cfg)
    outputs = run_backtest(
        close=close if strat.positions.shape[1] > 1 else close[[strat.positions.columns[0]]],
        positions=strat.positions,
        signal=strat.signal,
        benchmark_close=bench_close,
        cfg=cfg,
        meta={k: float(v) for k, v in strat.meta.items() if isinstance(v, (int, float))},
    )

    title = f"Project #5 — Strategy Backtest ({cfg.strategy})"
    html = performance_report(outputs, title=title)

    parent = os.path.dirname(out_html)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    payload = build_report_payload(outputs, title=title)
    payload["output_html"] = out_html
    payload["strategy_meta"] = strat.meta
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

