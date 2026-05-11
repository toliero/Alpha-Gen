from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .types import BacktestConfig, BacktestOutputs


def _max_drawdown(equity: pd.Series) -> pd.Series:
    rm = equity.cummax()
    return equity / rm - 1.0


def _apply_transaction_costs(positions: pd.DataFrame, returns: pd.DataFrame, bps: float) -> pd.Series:
    # Vectorized: turnover is sum(abs(delta weights))
    w = positions.fillna(0.0)
    dw = w.diff().abs().sum(axis=1).fillna(0.0)
    gross_ret = (w.shift(1) * returns).sum(axis=1).fillna(0.0)
    cost = (bps / 10_000.0) * dw
    return gross_ret - cost


def _trade_runs(pos: pd.Series) -> pd.DataFrame:
    # Identify contiguous non-zero position segments
    p = pos.fillna(0.0)
    is_open = p != 0
    change = is_open.ne(is_open.shift(1, fill_value=False))
    group = change.cumsum()
    trades = []
    for g, sub in p[is_open].groupby(group[is_open]):
        start = sub.index[0]
        end = sub.index[-1]
        direction = float(np.sign(sub.iloc[0]))
        trades.append({"start": start, "end": end, "direction": direction})
    return pd.DataFrame(trades)


def compute_mae_mfe(pos: pd.Series, pnl: pd.Series) -> Tuple[float, float]:
    """
    MAE/MFE over trades (in %). Uses trade-level cumulative PnL path.
    """
    trades = _trade_runs(pos)
    if trades.empty:
        return float("nan"), float("nan")

    maes, mfes = [], []
    for _, tr in trades.iterrows():
        seg = pnl.loc[tr["start"] : tr["end"]].cumsum()
        # adverse/favorable relative to 0 baseline
        mae = float(seg.min())
        mfe = float(seg.max())
        maes.append(mae)
        mfes.append(mfe)
    return float(np.nanmean(maes)), float(np.nanmean(mfes))


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    signal: pd.Series,
    pos_for_mae: pd.Series,
) -> Dict[str, float]:
    r = returns.dropna()
    b = benchmark_returns.reindex(r.index).fillna(0.0)
    active = (r - b).dropna()

    ann = np.sqrt(252.0)
    vol = float(r.std() * ann)
    sharpe = float((r.mean() / r.std()) * ann) if r.std() != 0 else float("nan")
    te = float(active.std() * ann)
    ir = float((active.mean() / active.std()) * ann) if active.std() != 0 else float("nan")

    downside = r[r < 0]
    sortino = float((r.mean() / downside.std()) * ann) if downside.std() != 0 else float("nan")

    skew = float(r.skew())
    kurt = float(r.kurtosis())

    # Beta to benchmark via OLS slope
    if b.var() == 0:
        beta = float("nan")
    else:
        beta = float(np.cov(r.values, b.values)[0, 1] / np.var(b.values))

    # IC: correlation between signal and next-day returns (aligned)
    sig = signal.reindex(r.index).astype(float)
    ic = float(sig.corr(r.shift(-1))) if len(sig) > 10 else float("nan")

    mae, mfe = compute_mae_mfe(pos_for_mae, r)

    return {
        "ann_vol": vol,
        "sharpe": sharpe,
        "information_ratio": ir,
        "tracking_error": te,
        "sortino": sortino,
        "skew": skew,
        "kurtosis": kurt,
        "beta_to_benchmark": beta,
        "information_coefficient": ic,
        "mae": mae,
        "mfe": mfe,
    }


def run_backtest(
    close: pd.DataFrame,
    positions: pd.DataFrame,
    signal: pd.Series,
    benchmark_close: pd.Series,
    cfg: BacktestConfig,
    meta: Dict[str, float],
) -> BacktestOutputs:
    # Daily returns
    rets = close.pct_change().dropna()
    pos = positions.reindex(rets.index).fillna(0.0)
    bret = benchmark_close.pct_change().reindex(rets.index).fillna(0.0)

    port_ret = _apply_transaction_costs(pos, rets, bps=cfg.transaction_cost_bps)

    equity = (1 + port_ret).cumprod()
    bench_eq = (1 + bret).cumprod()
    dd = _max_drawdown(equity)

    # If single-asset strategy: use that position for MAE/MFE; else use spread sign
    if pos.shape[1] == 1:
        pos_for_mae = pos.iloc[:, 0]
    else:
        # Use signed spread position if available (approx via first leg)
        pos_for_mae = np.sign(pos.iloc[:, 0])

    metrics = compute_metrics(port_ret, bret, signal.reindex(rets.index), pos_for_mae)

    merged_meta = {**{k: float(v) for k, v in meta.items() if isinstance(v, (int, float))}, **metrics}
    merged_meta.update({"transaction_cost_bps": float(cfg.transaction_cost_bps)})

    return BacktestOutputs(
        equity=equity,
        returns=port_ret,
        benchmark_equity=bench_eq,
        benchmark_returns=bret,
        drawdown=dd,
        signals=signal.reindex(rets.index),
        positions=pos,
        meta=merged_meta,
    )

