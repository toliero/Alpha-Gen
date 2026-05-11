# Multi-Factor Alpha Research Environment

## Objective
This repository (`/alpha-research`) serves as the primary sandbox for developing, backtesting, and validating quantitative trading strategies. The focus bridges standard statistical arbitrage with advanced market microstructure signals.

## Core Models

### 1. Statistical Arbitrage
Leveraging co-integration between correlated assets. The current baseline model utilizes rolling Z-scores of the spread to trigger mean-reverting positions.

### 2. Mean Reversion Dynamics
Baseline implementation utilizing standard Bollinger Bands. 

## Future Integrations: Non-Markovian Microstructure
The next phase of this project will integrate findings from our `/lab-simulators` project, specifically:
- **Hurst Exponent ($H$) Analysis**: Modulating strategy aggression based on the current Rough Volatility regime. If $H < 0.5$ (anti-persistent), mean-reversion strategies will be scaled up.
- **Algorithmic Coupling ($\rho$)**: Incorporating signals that detect high systemic coupling or overlap in order book flow to predict localized liquidity crunches or flash crashes, allowing the strategy to hedge out of standard statistical arbitrage positions before correlations break down.
