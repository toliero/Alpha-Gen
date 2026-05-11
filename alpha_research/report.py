from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .types import BacktestOutputs


def _rolling_vol(r: pd.Series, window: int = 60) -> pd.Series:
    ann = np.sqrt(252.0)
    return r.rolling(window, min_periods=max(10, window // 3)).std() * ann


def _rolling_sharpe(r: pd.Series, window: int = 60) -> pd.Series:
    ann = np.sqrt(252.0)
    mu = r.rolling(window, min_periods=max(10, window // 3)).mean()
    sig = r.rolling(window, min_periods=max(10, window // 3)).std()
    return (mu / sig.replace(0, np.nan)) * ann


def monthly_returns(r: pd.Series) -> pd.Series:
    m = (1 + r).resample("ME").prod() - 1
    m.name = "monthly_return"
    return m


def _monthly_heatmap_trace(m: pd.Series) -> go.Heatmap:
    df = m.to_frame()
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")
    piv = df.pivot_table(index="year", columns="month", values="monthly_return", aggfunc="mean")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv = piv.reindex(columns=[c for c in month_order if c in piv.columns])

    return go.Heatmap(
        z=(piv.values * 100.0),
        x=piv.columns.tolist(),
        y=piv.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="%"),
        hovertemplate="Year %{y}<br>%{x}: %{z:.2f}%<extra></extra>",
    )


def performance_report(outputs: BacktestOutputs, title: str) -> str:
    """
    Standalone HTML (Plotly via CDN), including:
    - 2x2 grid: cumulative vs benchmark, rolling vol+sharpe, underwater drawdown (filled), monthly heatmap
    - metrics panel (IR, Sortino, skew/kurt, MAE/MFE, beta, IC)
    - centered scientific description block
    """
    eq = outputs.equity
    beq = outputs.benchmark_equity.reindex(eq.index).ffill()
    r = outputs.returns.reindex(eq.index).fillna(0.0)
    dd = outputs.drawdown.reindex(eq.index).fillna(0.0) * 100.0
    vol = _rolling_vol(r, 60)
    sh = _rolling_sharpe(r, 60)

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=(
            "Cumulative returns vs benchmark",
            "Rolling volatility & Sharpe (60d, ann.)",
            "Underwater drawdown (filled)",
            "Monthly return heatmap",
        ),
    )

    fig.add_trace(
        go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(color="#0f172a", width=2)),
        1,
        1,
    )
    fig.add_trace(
        go.Scatter(x=beq.index, y=beq.values, name="Benchmark", line=dict(color="#2563eb", width=1.8)),
        1,
        1,
    )

    fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="Vol", line=dict(color="#0f172a", width=1.6)), 1, 2)
    fig.add_trace(go.Scatter(x=sh.index, y=sh.values, name="Sharpe", line=dict(color="#2563eb", width=1.6)), 1, 2)

    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            name="Drawdown %",
            line=dict(color="#1e3a5f", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(30,58,95,0.35)",
        ),
        2,
        1,
    )

    fig.add_trace(_monthly_heatmap_trace(monthly_returns(r)), 2, 2)

    fig.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=48, r=20, t=70, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=dict(text=title, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Ann.", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)

    meta = outputs.meta
    rows = [
        ("Sharpe", meta.get("sharpe")),
        ("Information Ratio", meta.get("information_ratio")),
        ("Sortino", meta.get("sortino")),
        ("Ann. Vol", meta.get("ann_vol")),
        ("Skew", meta.get("skew")),
        ("Kurtosis", meta.get("kurtosis")),
        ("MAE", meta.get("mae")),
        ("MFE", meta.get("mfe")),
        ("Beta (to bench)", meta.get("beta_to_benchmark")),
        ("IC", meta.get("information_coefficient")),
        ("TC (bps)", meta.get("transaction_cost_bps")),
    ]

    def fmt(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        if isinstance(v, (int, float)):
            return f"{float(v):.4f}"
        return str(v)

    metrics_html = "".join(
        f"<div class='mrow'><div class='k'>{k}</div><div class='v'>{fmt(v)}</div></div>" for k, v in rows
    )

    description = (
        "This engine implements a cointegration-based statistical arbitrage framework. "
        "It utilizes the Engle-Granger two-step method to identify stationary residuals ($z_t$) "
        "in non-stationary price series. By applying a mean-reverting stochastic process to the spread, "
        "the model generates alpha through high-probability convergence trades, validated via walk-forward "
        "cross-validation to mitigate over-fitting."
    )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "IBM Plex Sans", system-ui, sans-serif; background: #fff; color: #0f172a; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 28px 22px 30px; }}
    .desc {{ max-width: 70ch; margin: 0 auto 18px; text-align: center; color: #334155; line-height: 1.6; }}
    .cards {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin: 12px auto 22px; }}
    .card {{ border: 1px solid #e2e8f0; border-radius: 14px; padding: 14px 14px; background: #fff; box-shadow: 0 10px 24px -20px rgba(15,23,42,0.35); }}
    .mgrid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; }}
    .mrow {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 10px; background: #f8fafc; }}
    .k {{ font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; color: #64748b; }}
    .v {{ margin-top: 6px; font-family: ui-monospace, "JetBrains Mono", Menlo, monospace; font-size: 12px; color: #0f172a; }}
    @media (max-width: 900px) {{ .mgrid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
    @media (max-width: 560px) {{ .mgrid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="desc">{description}</div>
    <div class="cards">
      <div class="card">
        <div class="mgrid">
          {metrics_html}
        </div>
      </div>
    </div>
    {fig.to_html(include_plotlyjs="cdn", full_html=False)}
  </div>
</body>
</html>
"""
    return html


def build_report_payload(outputs: BacktestOutputs, title: str) -> dict:
    """
    JSON-friendly payload for direct rendering in Next.js (no iframe):
    - description text
    - metrics dict
    - plotly figure JSON (2x2 grid)
    """
    eq = outputs.equity
    beq = outputs.benchmark_equity.reindex(eq.index).ffill()
    r = outputs.returns.reindex(eq.index).fillna(0.0)
    dd = outputs.drawdown.reindex(eq.index).fillna(0.0) * 100.0
    vol = _rolling_vol(r, 60)
    sh = _rolling_sharpe(r, 60)

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.12,
        vertical_spacing=0.14,
        subplot_titles=(
            "Cumulative returns vs benchmark",
            "Rolling volatility & Sharpe (60d, ann.)",
            "Underwater drawdown (filled)",
            "Monthly return heatmap",
        ),
    )

    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(color="#0f172a", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=beq.index, y=beq.values, name="Benchmark", line=dict(color="#2563eb", width=1.8)), 1, 1)
    fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="Vol", line=dict(color="#0f172a", width=1.6)), 1, 2)
    fig.add_trace(go.Scatter(x=sh.index, y=sh.values, name="Sharpe", line=dict(color="#2563eb", width=1.6)), 1, 2)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            name="Drawdown %",
            line=dict(color="#1e3a5f", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(30,58,95,0.35)",
        ),
        2,
        1,
    )
    fig.add_trace(_monthly_heatmap_trace(monthly_returns(r)), 2, 2)

    fig.update_layout(
        template="plotly_white",
        height=920,
        margin=dict(l=48, r=20, t=70, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        title=dict(text=title, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Ann.", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)

    description = (
        "This engine implements a cointegration-based statistical arbitrage framework. "
        "It utilizes the Engle-Granger two-step method to identify stationary residuals ($z_t$) "
        "in non-stationary price series. By applying a mean-reverting stochastic process to the spread, "
        "the model generates alpha through high-probability convergence trades, validated via walk-forward "
        "cross-validation to mitigate over-fitting."
    )

    return {
        "title": title,
        "description": description,
        "metrics": outputs.meta,
        # plotly.io.to_json converts numpy arrays to JSON-safe lists
        "figure": json.loads(pio.to_json(fig, validate=False)),
    }
