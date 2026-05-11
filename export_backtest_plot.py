#!/usr/bin/env python3
"""CLI: write Matplotlib backtest PNG to argv[1] for Next.js /public/plots bridge."""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from alpha_engine import run_demo, save_matplotlib_backtest_png  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 export_backtest_plot.py <output.png>", file=sys.stderr)
        sys.exit(2)
    out = sys.argv[1]
    result, _z, diag = run_demo("pairs")
    if diag:
        for k, v in diag.items():
            print(f"{k}: {v}")
    save_matplotlib_backtest_png(result, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
