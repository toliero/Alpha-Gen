"""
Multi-factor alpha research environment: cointegration-tested pairs trading,
momentum factors, and publication-grade diagnostics (Plotly).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import yfinance as yf


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    drawdown: pd.Series


def _rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    mu = returns.rolling(window, min_periods=max(10, window // 4)).mean()
    sig = returns.rolling(window, min_periods=max(10, window // 4)).std()
    ann = np.sqrt(252.0)
    return (mu / sig.replace(0, np.nan)) * ann


def save_matplotlib_backtest_png(result: BacktestResult, path: str) -> None:
    """
    Matplotlib / seaborn-compatible static figure for Next.js: equity, underwater drawdown,
    rolling Sharpe — saved to PNG (Agg backend).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    r = result.returns
    rs = _rolling_sharpe(r, window=60)
    dd_pct = result.drawdown * 100.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), dpi=120, sharex=True)
    axes[0].plot(result.equity.index, result.equity.values, color="#0f172a", lw=1.3)
    axes[0].set_ylabel("Equity ($1)")
    axes[0].grid(True, alpha=0.28)
    axes[0].set_title("Cointegration-filtered pairs — Matplotlib diagnostics")

    axes[1].fill_between(dd_pct.index, dd_pct.values, 0.0, color="#1e3a5f", alpha=0.42, linewidth=0)
    axes[1].plot(dd_pct.index, dd_pct.values, color="#0f172a", lw=0.9)
    axes[1].axhline(0.0, color="#94a3b8", lw=0.9)
    axes[1].set_ylabel("Drawdown %")

    axes[2].plot(rs.index, rs.values, color="#2563eb", lw=1.15)
    axes[2].axhline(0.0, color="#cbd5e1", ls=":", lw=0.9)
    axes[2].set_ylabel("Rolling Sharpe (ann.)")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.22)

    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


class PairsTrader:
    """
    Cointegration-aware pairs engine.

    Identification: (i) Engle–Granger two-step `coint` on log prices;
    (ii) Johansen trace test for rank r=0 vs r≥1 on the bivariate log-price VAR;
    (iii) ADF on OLS residuals ε_t from log p^A_t = α + β log p^B_t + ε_t.

    Trading signal: rolling Z-score on ε_t (mean-reversion on the estimated
    cointegrating residual), with dollar-neutral exposure in return space via
    r^p_t ≈ w_{t-1} (r^A_t − β r^B_t).
    """

    def __init__(
        self,
        asset_a: str,
        asset_b: str,
        window: int = 20,
        z_score_threshold: float = 2.0,
    ):
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.window = window
        self.threshold = z_score_threshold
        self.data = pd.DataFrame()
        self.diagnostics: Dict[str, Any] = {}
        self._hedge_beta: float = 1.0

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        df_a = yf.download(self.asset_a, start=start_date, end=end_date, progress=False)[
            "Close"
        ]
        df_b = yf.download(self.asset_b, start=start_date, end=end_date, progress=False)[
            "Close"
        ]
        if isinstance(df_a, pd.DataFrame):
            df_a = df_a.iloc[:, 0]
        if isinstance(df_b, pd.DataFrame):
            df_b = df_b.iloc[:, 0]
        self.data = pd.DataFrame({self.asset_a: df_a, self.asset_b: df_b}).dropna()
        return self.data

    def generate_signals(self) -> pd.DataFrame:
        if self.data.empty:
            raise ValueError("Data not loaded. Call fetch_data() first.")

        log_a = np.log(self.data[self.asset_a].astype(float))
        log_b = np.log(self.data[self.asset_b].astype(float))
        df = pd.DataFrame({"la": log_a, "lb": log_b}).dropna()
        if len(df) < 80:
            raise ValueError("Insufficient observations for cointegration inference.")

        X = sm.add_constant(df["lb"])
        ols = sm.OLS(df["la"], X).fit()
        beta = float(ols.params.iloc[1])
        alpha = float(ols.params.iloc[0])
        epsilon_fit = pd.Series(ols.resid, index=df.index)

        adf_eps = adfuller(epsilon_fit.values, autolag="AIC")
        eg_stat, eg_p, _crit = coint(df["la"].values, df["lb"].values, trend="c", autolag="AIC")

        endog = np.column_stack([df["la"].values, df["lb"].values])
        joh = coint_johansen(endog, det_order=0, k_ar_diff=1)
        trace_r0 = float(joh.lr1[0])
        crit_r0_95 = float(joh.cvt[0, 1])

        self._hedge_beta = beta
        self.diagnostics = {
            "hedge_ratio_log_ols": beta,
            "intercept_log_ols": alpha,
            "engle_granger_stat": float(eg_stat),
            "engle_granger_pvalue": float(eg_p),
            "adf_residual_stat": float(adf_eps[0]),
            "adf_residual_pvalue": float(adf_eps[1]),
            "adf_residual_used_lag": int(adf_eps[2]),
            "johansen_trace_r0": trace_r0,
            "johansen_cv95_r0": crit_r0_95,
            "johansen_reject_rank0_at_5pct": trace_r0 > crit_r0_95,
            "engle_granger_evidence_coint_5pct": eg_p < 0.05,
        }

        roll_mu = epsilon_fit.rolling(window=self.window).mean()
        roll_sig = epsilon_fit.rolling(window=self.window).std()
        z_fit = (epsilon_fit - roll_mu) / roll_sig.replace(0, np.nan)
        epsilon = epsilon_fit.reindex(self.data.index)
        z = z_fit.reindex(self.data.index)

        spread_pos = pd.Series(0.0, index=self.data.index)
        spread_pos.loc[z > self.threshold] = -1.0
        spread_pos.loc[z < -self.threshold] = 1.0

        out = self.data.copy()
        out["Cointegrating_Residual"] = epsilon
        out["Z_Score"] = z
        out["Hedge_Beta"] = self._hedge_beta
        out["Spread_Position"] = spread_pos
        out["Position_A"] = spread_pos
        out["Position_B"] = -self._hedge_beta * spread_pos
        return out


class MomentumFactor:
    """Time-series momentum (ROC) on a single asset."""

    def __init__(self, ticker: str, lookback: int = 21):
        self.ticker = ticker
        self.lookback = lookback

    def fetch_and_score(self, start_date: str, end_date: str) -> pd.DataFrame:
        px = yf.download(self.ticker, start=start_date, end=end_date, progress=False)[
            "Close"
        ]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        df = pd.DataFrame({self.ticker: px}).dropna()
        df["Momentum"] = df[self.ticker].pct_change(self.lookback)
        df["Position"] = np.where(df["Momentum"] > 0, 1.0, 0.0)
        return df


def cointegrated_pairs_backtest(signals: pd.DataFrame, asset_a: str, asset_b: str) -> BacktestResult:
    """Dollar-neutral spread returns: r_{p,t} ≈ s_{t-1} (r_{A,t} − β r_{B,t})."""
    ra = signals[asset_a].pct_change()
    rb = signals[asset_b].pct_change()
    beta = float(signals["Hedge_Beta"].iloc[-1])
    spread_ret = ra - beta * rb
    pos = signals["Spread_Position"]
    port_ret = (pos.shift(1) * spread_ret).fillna(0.0)
    equity = (1 + port_ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return BacktestResult(equity=equity, returns=port_ret, drawdown=drawdown)


class BacktestVisualizer:
    """Equity, underwater drawdown, and rolling Sharpe (annualized)."""

    def __init__(self, result: BacktestResult, z_series: Optional[pd.Series] = None):
        self.result = result
        self.z_series = z_series

    def figure(self) -> go.Figure:
        r = self.result.returns
        rs = _rolling_sharpe(r, window=60)
        dd_pct = self.result.drawdown * 100.0

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.34, 0.33, 0.33],
            subplot_titles=(
                "Cumulative equity (growth of $1)",
                "Drawdown (underwater)",
                "Rolling Sharpe (60d, ann.)",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=self.result.equity.index,
                y=self.result.equity.values,
                name="Equity",
                line=dict(color="#0f172a", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=dd_pct.index,
                y=dd_pct.values,
                name="Drawdown %",
                line=dict(color="#1e3a5f", width=1.2),
                fill="tozeroy",
                fillcolor="rgba(30,58,95,0.35)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="#94a3b8", row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=rs.index,
                y=rs.values,
                name="Rolling Sharpe",
                line=dict(color="#2563eb", width=1.6),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#cbd5e1", row=3, col=1)

        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="%", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe", row=3, col=1)
        fig.update_layout(
            template="plotly_white",
            height=900,
            margin=dict(l=52, r=28, t=56, b=44),
            showlegend=False,
            font=dict(family="Georgia, serif", size=11, color="#0f172a"),
        )
        return fig

    def write_html(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.figure().write_html(path, include_plotlyjs="cdn", full_html=True)


def run_demo(
    mode: Literal["pairs", "momentum"] = "pairs",
) -> Tuple[BacktestResult, Optional[pd.Series], Optional[Dict[str, Any]]]:
    start, end = "2023-01-01", "2024-01-01"
    if mode == "pairs":
        engine = PairsTrader("GOOG", "GOOGL", window=20, z_score_threshold=2.0)
        engine.fetch_data(start, end)
        sig = engine.generate_signals()
        bt = cointegrated_pairs_backtest(sig, engine.asset_a, engine.asset_b)
        return bt, sig["Z_Score"], engine.diagnostics
    mom = MomentumFactor("SPY", lookback=21)
    df = mom.fetch_and_score(start, end)
    rets = df[mom.ticker].pct_change()
    port_ret = df["Position"].shift(1) * rets
    port_ret = port_ret.fillna(0.0)
    equity = (1 + port_ret).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return BacktestResult(equity=equity, returns=port_ret, drawdown=dd), None, None


if __name__ == "__main__":
    result, z, diag = run_demo("pairs")
    if diag:
        print("--- Cointegration diagnostics ---")
        for k, v in diag.items():
            print(f"  {k}: {v}")
    viz = BacktestVisualizer(result, z_series=z)
    out_path = os.path.join(os.path.dirname(__file__), "output", "backtest_report.html")
    viz.write_html(out_path)
    print(f"Wrote {out_path}")
    print("Backtest demo completed successfully.")
