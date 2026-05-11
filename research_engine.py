import pandas as pd
import numpy as np
from typing import List, Dict, Union

class AlphaModel:
    """Base class for all quantitative alpha models."""
    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement generate_signals")

class StatisticalArbitrage(AlphaModel):
    """
    Statistical Arbitrage via Co-integration.
    Uses rolling Z-scores of the price spread between two co-integrated assets.
    """
    def __init__(self, asset_a: str, asset_b: str, window: int = 20, z_score_threshold: float = 2.0):
        super().__init__("Statistical Arbitrage")
        self.asset_a = asset_a
        self.asset_b = asset_b
        self.window = window
        self.threshold = z_score_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Expects data with columns for asset_a and asset_b prices.
        Returns a DataFrame of target positions (-1, 0, 1) for both assets.
        """
        if self.asset_a not in data.columns or self.asset_b not in data.columns:
            raise ValueError(f"Data must contain columns for {self.asset_a} and {self.asset_b}")

        # Simple spread (assuming beta = 1 for simplicity in this baseline)
        spread = data[self.asset_a] - data[self.asset_b]
        
        # Rolling statistics
        rolling_mean = spread.rolling(window=self.window).mean()
        rolling_std = spread.rolling(window=self.window).std()
        
        # Z-score of the spread
        z_score = (spread - rolling_mean) / rolling_std
        
        # Generate target positions
        positions = pd.DataFrame(index=data.index)
        positions[self.asset_a] = 0.0
        positions[self.asset_b] = 0.0

        # When Z-score is > threshold, spread is too high -> Short A, Long B
        positions.loc[z_score > self.threshold, self.asset_a] = -1.0
        positions.loc[z_score > self.threshold, self.asset_b] = 1.0

        # When Z-score is < -threshold, spread is too low -> Long A, Short B
        positions.loc[z_score < -self.threshold, self.asset_a] = 1.0
        positions.loc[z_score < -self.threshold, self.asset_b] = -1.0

        return positions

class BollingerMeanReversion(AlphaModel):
    """
    Mean Reversion strategy using Bollinger Bands.
    Triggers when price deviates significantly from its moving average.
    """
    def __init__(self, target_asset: str, window: int = 20, num_std: float = 2.0):
        super().__init__("Bollinger Mean Reversion")
        self.target_asset = target_asset
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Expects data with a column for target_asset price.
        Returns a Series of target positions (-1, 0, 1).
        """
        if self.target_asset not in data.columns:
            raise ValueError(f"Data must contain column for {self.target_asset}")

        prices = data[self.target_asset]
        
        rolling_mean = prices.rolling(window=self.window).mean()
        rolling_std = prices.rolling(window=self.window).std()
        
        upper_band = rolling_mean + (rolling_std * self.num_std)
        lower_band = rolling_mean - (rolling_std * self.num_std)
        
        signals = pd.Series(0.0, index=data.index)
        
        # Price > Upper Band -> Overbought -> Short
        signals.loc[prices > upper_band] = -1.0
        
        # Price < Lower Band -> Oversold -> Long
        signals.loc[prices < lower_band] = 1.0
        
        return signals

if __name__ == "__main__":
    # Simple test to ensure engine loads correctly
    print("Multi-Factor Alpha Research Engine Initialized.")
    print(f"Available Models:")
    print(f"- {StatisticalArbitrage.__name__}")
    print(f"- {BollingerMeanReversion.__name__}")