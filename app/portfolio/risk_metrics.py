"""One-day VaR and Expected Shortfall.

Convention: losses are positive numbers, VaR(c) is the c-quantile of the loss
distribution, ES(c) is the mean loss at or beyond VaR. Long-only portfolio.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def _validate(returns, confidence: float) -> np.ndarray:
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level")
    r = np.asarray(returns, dtype=float)
    if len(r) < 30:
        raise ValueError("Insufficient observations")
    if not np.isfinite(r).all():
        raise ValueError("Returns contain invalid values")
    return r


def historical_var_es(portfolio_returns, portfolio_value: float, confidence: float = 0.99) -> dict:
    """Historical simulation VaR/ES (empirical quantile, linear interpolation)."""
    r = _validate(portfolio_returns, confidence)
    losses = -r
    var = float(np.quantile(losses, confidence))
    es = float(losses[losses >= var].mean())
    return {"method": "historical", "confidence": confidence, "observations": len(r),
            "var_pct": var, "es_pct": es,
            "var": var * portfolio_value, "es": es * portfolio_value}


def parametric_var_es(portfolio_returns, portfolio_value: float, confidence: float = 0.99,
                      use_mean: bool = False) -> dict:
    """Variance-covariance VaR/ES assuming normal returns.

    VaR = z_c * sigma - mu ;  ES = sigma * phi(z_c) / (1 - c) - mu
    mu is ignored by default (zero-mean, the conservative convention).
    """
    r = _validate(portfolio_returns, confidence)
    mu = float(r.mean()) if use_mean else 0.0
    sigma = float(r.std(ddof=1))
    z = float(norm.ppf(confidence))
    var = z * sigma - mu
    es = sigma * float(norm.pdf(z)) / (1 - confidence) - mu
    return {"method": "parametric", "confidence": confidence, "observations": len(r),
            "var_pct": var, "es_pct": es,
            "var": var * portfolio_value, "es": es * portfolio_value}


def scale_horizon(one_day_amount: float, horizon_days: int) -> float:
    """Square-root-of-time scaling. An APPROXIMATION: assumes i.i.d. returns,
    so it understates risk when volatility is clustered or returns autocorrelate."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    return one_day_amount * math.sqrt(horizon_days)
