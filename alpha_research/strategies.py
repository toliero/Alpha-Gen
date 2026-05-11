from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .types import BacktestConfig


def rolling_z(x: pd.Series, lookback: int) -> pd.Series:
    mu = x.rolling(lookback, min_periods=max(lookback // 2, 10)).mean()
    sig = x.rolling(lookback, min_periods=max(lookback // 2, 10)).std()
    return (x - mu) / sig.replace(0, np.nan)


@dataclass(frozen=True)
class StrategyResult:
    # signals should be a continuous score (e.g., z-score); positions derived from it.
    signal: pd.Series
    # positions columns are tickers with dollar weights (can be fractional; sum abs ≈ 1)
    positions: pd.DataFrame
    meta: dict


def _estimate_cointegration(log_a: pd.Series, log_b: pd.Series) -> Tuple[float, float, pd.Series, dict]:
    df = pd.DataFrame({"la": log_a, "lb": log_b}).dropna()
    if len(df) < 120:
        raise ValueError("Insufficient observations for cointegration inference (need >= 120).")

    X = sm.add_constant(df["lb"])
    ols = sm.OLS(df["la"], X).fit()
    beta = float(ols.params.iloc[1])
    alpha = float(ols.params.iloc[0])
    eps = pd.Series(ols.resid, index=df.index)

    adf = adfuller(eps.values, autolag="AIC")
    eg_stat, eg_p, _ = coint(df["la"].values, df["lb"].values, trend="c", autolag="AIC")
    joh = coint_johansen(np.column_stack([df["la"].values, df["lb"].values]), det_order=0, k_ar_diff=1)
    trace_r0 = float(joh.lr1[0])
    cv95 = float(joh.cvt[0, 1])

    meta = {
        "hedge_beta": beta,
        "hedge_alpha": alpha,
        "adf_eps_stat": float(adf[0]),
        "adf_eps_p": float(adf[1]),
        "eg_stat": float(eg_stat),
        "eg_p": float(eg_p),
        "joh_trace_r0": trace_r0,
        "joh_cv95_r0": cv95,
        "joh_reject_r0_5pct": float(trace_r0 > cv95),
        "eg_evidence_coint_5pct": float(eg_p < 0.05),
    }
    return alpha, beta, eps, meta


def dual_class_arbitrage(close: pd.DataFrame, cfg: BacktestConfig) -> StrategyResult:
    if not cfg.asset_a or not cfg.asset_b:
        raise ValueError("dual_class_arb requires asset_a and asset_b.")
    a, b = cfg.asset_a.upper(), cfg.asset_b.upper()
    px = close[[a, b]].dropna()

    log_a, log_b = np.log(px[a]), np.log(px[b])
    alpha, beta, eps, meta = _estimate_cointegration(log_a, log_b)

    z = rolling_z(eps, cfg.lookback)
    z = z.reindex(px.index)

    # Entry/exit logic (hysteresis)
    pos = pd.Series(0.0, index=px.index)
    long_mask = z < -cfg.entry_z
    short_mask = z > cfg.entry_z
    exit_mask = z.abs() < cfg.exit_z

    state = 0.0
    for t in px.index:
        if state == 0.0:
            if long_mask.loc[t]:
                state = 1.0
            elif short_mask.loc[t]:
                state = -1.0
        else:
            if exit_mask.loc[t]:
                state = 0.0
        pos.loc[t] = state

    # Dollar weights: long residual => long A, short beta*B
    weights = pd.DataFrame(index=px.index, columns=[a, b], data=0.0)
    weights[a] = pos
    weights[b] = -beta * pos

    # Normalize gross exposure to 1 (avoid leverage drift)
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(gross, axis=0).fillna(0.0)

    meta = {**meta, "asset_a": a, "asset_b": b, "strategy": "dual_class_arb"}
    return StrategyResult(signal=z, positions=weights, meta=meta)


def sector_pairs(close: pd.DataFrame, cfg: BacktestConfig) -> StrategyResult:
    # Minimal curated sector universe (extendable)
    universe = {
        "tech": [("MSFT", "AAPL"), ("GOOG", "META"), ("NVDA", "AMD")],
        "consumer": [("PEP", "KO"), ("MCD", "SBUX")],
        "energy": [("XOM", "CVX")],
        "banks": [("JPM", "BAC"), ("GS", "MS")],
    }
    sector = (cfg.sector or "tech").strip().lower()
    pairs = universe.get(sector)
    if not pairs:
        raise ValueError(f"Unknown sector '{sector}'. Available: {sorted(universe.keys())}")

    # Choose pair with best Engle–Granger p-value
    best = None
    best_p = 1.0
    best_meta = None
    best_eps = None
    best_ab = None
    for a, b in pairs:
        if a not in close.columns or b not in close.columns:
            continue
        px = close[[a, b]].dropna()
        if len(px) < 150:
            continue
        log_a, log_b = np.log(px[a]), np.log(px[b])
        try:
            alpha, beta, eps, meta = _estimate_cointegration(log_a, log_b)
        except Exception:
            continue
        p = float(meta["eg_p"])
        if p < best_p:
            best_p = p
            best = (alpha, beta)
            best_meta = meta
            best_eps = eps
            best_ab = (a, b)

    if not best or not best_ab or best_eps is None or best_meta is None:
        raise ValueError(f"No viable pair found for sector '{sector}' in the available window.")

    cfg2 = BacktestConfig(
        strategy=cfg.strategy,
        start=cfg.start,
        end=cfg.end,
        asset_a=best_ab[0],
        asset_b=best_ab[1],
        benchmark=cfg.benchmark,
        lookback=cfg.lookback,
        entry_z=cfg.entry_z,
        exit_z=cfg.exit_z,
        transaction_cost_bps=cfg.transaction_cost_bps,
        stop_loss_pct=cfg.stop_loss_pct,
        take_profit_pct=cfg.take_profit_pct,
    )
    out = dual_class_arbitrage(close, cfg2)
    out.meta.update({"sector": sector, "strategy": "sector_pairs"})
    return out


def bollinger_bands(close: pd.DataFrame, cfg: BacktestConfig) -> StrategyResult:
    if not cfg.ticker:
        raise ValueError("bollinger_bands requires ticker.")
    t = cfg.ticker.upper()
    px = close[[t]].dropna()
    s = px[t]
    ma = s.rolling(cfg.lookback, min_periods=max(cfg.lookback // 2, 10)).mean()
    sd = s.rolling(cfg.lookback, min_periods=max(cfg.lookback // 2, 10)).std()
    upper = ma + 2.0 * sd
    lower = ma - 2.0 * sd

    # Signal: normalized distance to band center
    z = (s - ma) / sd.replace(0, np.nan)

    pos = pd.Series(0.0, index=px.index)
    long_mask = s < lower
    short_mask = s > upper
    exit_mask = z.abs() < 0.25

    state = 0.0
    for dt in px.index:
        if state == 0.0:
            if long_mask.loc[dt]:
                state = 1.0
            elif short_mask.loc[dt]:
                state = -1.0
        else:
            if exit_mask.loc[dt]:
                state = 0.0
        pos.loc[dt] = state

    weights = pd.DataFrame(index=px.index, columns=[t], data=0.0)
    weights[t] = pos
    meta = {"ticker": t, "strategy": "bollinger_bands"}
    return StrategyResult(signal=z, positions=weights, meta=meta)


def rl_mean_reversion(close: pd.DataFrame, cfg: BacktestConfig) -> StrategyResult:
    """
    Experimental toy RL: tabular Q-learning on discretized z-score state.
    Action space: {-1, 0, 1} on a single ticker.
    This is intentionally lightweight (no external RL libs) and meant as a portfolio demo.
    """
    if not cfg.ticker:
        raise ValueError("rl_mean_reversion requires ticker.")
    t = cfg.ticker.upper()
    px = close[[t]].dropna()
    s = px[t]
    r = s.pct_change().fillna(0.0)

    ma = s.rolling(cfg.lookback, min_periods=max(cfg.lookback // 2, 10)).mean()
    sd = s.rolling(cfg.lookback, min_periods=max(cfg.lookback // 2, 10)).std()
    z = ((s - ma) / sd.replace(0, np.nan)).fillna(0.0)

    bins = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
    state = np.digitize(z.values, bins=bins)  # 0..len(bins)

    actions = np.array([-1.0, 0.0, 1.0])
    nS = len(bins) + 1
    nA = len(actions)
    Q = np.zeros((nS, nA), dtype=float)

    # Train on first 60%
    n = len(px)
    split = max(int(0.6 * n), 50)
    alpha = 0.08
    gamma = 0.10  # short-horizon
    eps = 0.15

    for i in range(1, split - 1):
        s0, s1 = state[i], state[i + 1]
        if np.random.rand() < eps:
            a_idx = np.random.randint(0, nA)
        else:
            a_idx = int(np.argmax(Q[s0]))
        a = actions[a_idx]
        reward = float(a * r.iloc[i + 1])
        Q[s0, a_idx] = (1 - alpha) * Q[s0, a_idx] + alpha * (reward + gamma * np.max(Q[s1]))

    # Greedy policy on full sample
    a_idx = np.argmax(Q[state], axis=1)
    pos = pd.Series(actions[a_idx], index=px.index)

    # Normalize to {-1,0,1} already; weights are just the position.
    weights = pd.DataFrame(index=px.index, columns=[t], data=0.0)
    weights[t] = pos
    meta = {"ticker": t, "strategy": "rl_mean_reversion", "train_split": float(split / n)}
    return StrategyResult(signal=z, positions=weights, meta=meta)


def run_strategy(close: pd.DataFrame, cfg: BacktestConfig) -> StrategyResult:
    if cfg.strategy == "dual_class_arb":
        return dual_class_arbitrage(close, cfg)
    if cfg.strategy == "sector_pairs":
        return sector_pairs(close, cfg)
    if cfg.strategy == "bollinger_bands":
        return bollinger_bands(close, cfg)
    if cfg.strategy == "rl_mean_reversion":
        return rl_mean_reversion(close, cfg)
    raise ValueError(f"Unknown strategy '{cfg.strategy}'")

