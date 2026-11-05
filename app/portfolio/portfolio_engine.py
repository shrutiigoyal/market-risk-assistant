"""Portfolio construction: loading, validation, returns, covariance.

R_p = sum_i w_i R_i          sigma_p^2 = w' Sigma w
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRADING_DAYS = 252


def load_portfolio(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "sample_portfolio.csv")
    missing = {"asset", "asset_class", "market_value"} - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {sorted(missing)}")
    if df["asset"].duplicated().any():
        raise ValueError("Duplicate assets in portfolio")
    if (df["market_value"] < 0).any():
        raise ValueError("Negative market values are not supported (long-only)")
    if df["market_value"].sum() <= 0:
        raise ValueError("Portfolio value must be positive")
    return df


def load_asset_returns(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "sample_asset_returns.csv", index_col="date", parse_dates=True)
    return df.sort_index()


def weights_from_portfolio(portfolio: pd.DataFrame) -> dict[str, float]:
    """Weights are always derived from market values so the two cannot disagree."""
    total = portfolio["market_value"].sum()
    return {r.asset: float(r.market_value / total) for r in portfolio.itertuples()}


def calculate_portfolio_returns(asset_returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Daily weights-times-returns series (constant weights, i.e. daily rebalanced)."""
    assets = list(weights)
    if not set(assets).issubset(asset_returns.columns):
        raise ValueError("Missing asset return columns")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Portfolio weights must sum to 1")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Negative weights are not supported")
    returns = asset_returns[assets].dropna()
    if returns.empty:
        raise ValueError("No return observations")
    return returns.dot(pd.Series(weights))


def covariance_matrix(asset_returns: pd.DataFrame, assets: list[str] | None = None,
                      annualize: bool = False) -> pd.DataFrame:
    r = asset_returns[assets] if assets else asset_returns
    cov = r.dropna().cov()
    return cov * TRADING_DAYS if annualize else cov


def portfolio_volatility(weights: dict, cov: pd.DataFrame, annualize: bool = False) -> float:
    assets = list(weights)
    w = np.array([weights[a] for a in assets])
    var = float(w @ cov.loc[assets, assets].values @ w)
    vol = float(np.sqrt(var))
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol
