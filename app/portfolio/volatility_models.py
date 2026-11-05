"""EWMA and GARCH(1,1) volatility. GARCH is fitted by Gaussian quasi-maximum
likelihood with scipy (no extra dependency); tests cross-check against the
`arch` package when it is installed.

sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252
EWMA_INIT_OBS = 30


def ewma_variance_series(returns, lam: float = 0.94, init_obs: int = EWMA_INIT_OBS) -> np.ndarray:
    """sigma2[t] is the variance forecast for day t made with returns up to t-1.
    Length n+1: the last element is the one-step-ahead forecast.
    Initialised with the variance of the first `init_obs` returns (no look-ahead)."""
    r = np.asarray(returns, dtype=float)
    if not 0 < lam < 1:
        raise ValueError("Lambda must be between 0 and 1")
    if len(r) < max(init_obs, 2):
        raise ValueError("Insufficient observations")
    if not np.isfinite(r).all():
        raise ValueError("Returns contain invalid values")
    s2 = np.empty(len(r) + 1)
    s2[0] = np.var(r[:init_obs], ddof=1)
    for t in range(len(r)):
        s2[t + 1] = lam * s2[t] + (1 - lam) * r[t] ** 2
    return s2


def ewma_volatility(returns, lam: float = 0.94) -> dict:
    s2 = ewma_variance_series(returns, lam)
    daily = float(np.sqrt(s2[-1]))
    return {"model": "EWMA", "lambda": lam, "daily_vol_forecast": daily,
            "annualized_vol_forecast": daily * np.sqrt(TRADING_DAYS),
            "initialisation": f"variance of first {EWMA_INIT_OBS} observations"}


def _garch_variance(params, eps):
    omega, alpha, beta = params
    s2 = np.empty(len(eps) + 1)
    s2[0] = np.var(eps)
    for t in range(len(eps)):
        s2[t + 1] = omega + alpha * eps[t] ** 2 + beta * s2[t]
    return s2


def fit_garch11(returns) -> dict:
    """Fit GARCH(1,1) with constant mean. Returns parameters, diagnostics and forecasts.
    Returns are scaled by 100 internally for numerical stability and scaled back."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 250:
        raise ValueError("GARCH needs at least 250 observations")
    if not np.isfinite(r).all():
        raise ValueError("Returns contain invalid values")
    scale = 100.0
    y = r * scale
    mu = y.mean()
    eps = y - mu
    v = eps.var()

    def nll(p):
        s2 = _garch_variance(p, eps)[:-1]
        if np.any(s2 <= 0):
            return 1e10
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps ** 2 / s2)

    res = minimize(nll, x0=[0.05 * v, 0.05, 0.90], method="SLSQP",
                   bounds=[(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)],
                   constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p[1] - p[2]}],
                   options={"maxiter": 500, "ftol": 1e-10})
    omega, alpha, beta = res.x
    s2 = _garch_variance(res.x, eps)
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    return {
        "model": "GARCH(1,1)",
        "omega": float(omega / scale ** 2), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence),
        "long_run_daily_vol": float(np.sqrt(long_run) / scale),
        "log_likelihood": float(-res.fun),
        "converged": bool(res.success), "optimizer_message": str(res.message),
        "conditional_volatility": np.sqrt(s2[:-1]) / scale,    # in-sample, per day
        "_next_variance": float(s2[-1]), "_long_run_variance": float(long_run), "_scale": scale,
    }


def garch_forecast(fit: dict, horizon_days: int = 10) -> dict:
    """h-step variance: V + (a+b)^(h-1) * (s2_{T+1} - V); total horizon vol = sqrt(sum of daily variances)."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    v, s1, p = fit["_long_run_variance"], fit["_next_variance"], fit["persistence"]
    daily_var = np.array([v + p ** h * (s1 - v) for h in range(horizon_days)])
    scale = fit["_scale"]
    return {"next_day_vol": float(np.sqrt(daily_var[0]) / scale),
            "next_day_annualized_vol": float(np.sqrt(daily_var[0]) / scale * np.sqrt(TRADING_DAYS)),
            "horizon_days": horizon_days,
            "horizon_vol": float(np.sqrt(daily_var.sum()) / scale),
            "daily_vol_path": [float(x) for x in np.sqrt(daily_var) / scale]}
