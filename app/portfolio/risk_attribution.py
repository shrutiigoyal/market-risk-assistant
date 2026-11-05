"""Euler risk decomposition of portfolio volatility.

marginal_i = (Sigma w)_i / sigma_p ;  component_i = w_i * marginal_i ;  sum_i component_i = sigma_p
"""
from __future__ import annotations

import numpy as np


def calculate_risk_contribution(weights: dict, covariance) -> dict:
    assets = list(weights)
    w = np.array([weights[a] for a in assets], dtype=float)
    sigma = covariance.loc[assets, assets].values
    port_vol = float(np.sqrt(w @ sigma @ w))
    if port_vol <= 0:
        raise ValueError("Portfolio volatility must be positive")
    marginal = sigma @ w / port_vol
    component = w * marginal
    pct = component / port_vol
    return {a: {"weight": float(w[i]), "marginal_contribution": float(marginal[i]),
                "risk_contribution": float(component[i]), "percentage": float(pct[i] * 100)}
            for i, a in enumerate(assets)}


def concentration_metrics(weights: dict, covariance) -> dict:
    """Diversification ratio = weighted avg stand-alone vol / portfolio vol (>= 1 for long-only);
    HHI of weights (1/n = equal weight, 1 = single asset)."""
    assets = list(weights)
    w = np.array([weights[a] for a in assets])
    sigma = covariance.loc[assets, assets].values
    vols = np.sqrt(np.diag(sigma))
    port_vol = float(np.sqrt(w @ sigma @ w))
    return {"diversification_ratio": float((w @ vols) / port_vol),
            "weight_hhi": float((w ** 2).sum())}
