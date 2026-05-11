from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketData:
    close: pd.DataFrame  # columns=tickers


def fetch_close(tickers: Iterable[str], start: str, end: str) -> MarketData:
    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t and t.strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")

    raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    # yfinance can return columns as Index for multi-ticker; keep simple columns.
    close = raw.copy()
    close.columns = [str(c).upper() for c in close.columns]
    close = close.dropna(how="all").ffill().dropna(how="any")
    return MarketData(close=close)

