from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import pandas as pd


StrategyId = Literal["dual_class_arb", "bollinger_bands", "sector_pairs", "rl_mean_reversion"]
BenchmarkId = Literal["SPY", "QQQ"]


@dataclass(frozen=True)
class BacktestConfig:
    strategy: StrategyId
    start: str
    end: str

    # Universe
    ticker: Optional[str] = None
    asset_a: Optional[str] = None
    asset_b: Optional[str] = None
    sector: Optional[str] = None
    benchmark: BenchmarkId = "SPY"

    # Model / signal knobs
    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5

    # Execution knobs
    transaction_cost_bps: float = 2.0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None


@dataclass(frozen=True)
class BacktestOutputs:
    equity: pd.Series
    returns: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    drawdown: pd.Series
    signals: pd.Series
    positions: pd.DataFrame
    meta: Dict[str, float]

