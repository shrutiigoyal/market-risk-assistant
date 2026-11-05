"""Out-of-sample VaR backtesting: Kupiec POF, Christoffersen independence,
conditional coverage and the Basel traffic light.

VaR for day t uses ONLY returns strictly before t (rolling window), so there is
no look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, norm

from .volatility_models import ewma_variance_series

METHODS = ("historical", "parametric", "ewma")


def rolling_var_forecasts(returns: pd.Series, confidence: float = 0.99, window: int = 250,
                          method: str = "historical", lam: float = 0.94) -> pd.Series:
    """VaR forecast (positive loss fraction) for each day t >= window, made from data before t."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level")
    r = pd.Series(returns).dropna()
    if len(r) <= window:
        raise ValueError("Need more observations than the window")
    vals = r.values
    z = norm.ppf(confidence)
    if method == "ewma":
        # EWMA variance forecast for day t uses returns up to t-1 (see ewma_variance_series)
        s2 = ewma_variance_series(vals, lam)
        out = z * np.sqrt(s2[window:len(vals)])
    else:
        out = np.empty(len(vals) - window)
        for i, t in enumerate(range(window, len(vals))):
            past = vals[t - window:t]
            out[i] = (np.quantile(-past, confidence) if method == "historical"
                      else z * past.std(ddof=1))
    return pd.Series(out, index=r.index[window:], name=f"{method}_var")


def kupiec_pof(n_obs: int, n_exceptions: int, confidence: float) -> dict:
    """Unconditional coverage test. H0: exception probability = 1 - confidence."""
    if n_obs <= 0 or not 0 <= n_exceptions <= n_obs:
        raise ValueError("Invalid counts")
    p, x, n = 1 - confidence, n_exceptions, n_obs
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = float(max(-2 * (ll0 - ll1), 0.0))
    return {"statistic": lr, "p_value": float(chi2.sf(lr, 1))}


def christoffersen_independence(hits) -> dict:
    """First-order Markov independence test on the 0/1 exception sequence."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    if n01 + n11 == 0 or n10 + n11 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n00": n00, "n01": n01, "n10": n10, "n11": n11,
                "note": "No exceptions (or none followed by another day): independence cannot be assessed."}
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = float(max(-2 * (ll0 - ll1), 0.0))
    return {"statistic": lr, "p_value": float(chi2.sf(lr, 1)),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def basel_traffic_light(exceptions_250d: int) -> str:
    """Basel zones for 99% 1-day VaR over 250 trading days."""
    return "green" if exceptions_250d <= 4 else "yellow" if exceptions_250d <= 9 else "red"


def backtest_var(returns: pd.Series, confidence: float = 0.99, window: int = 250,
                 method: str = "historical", significance: float = 0.05) -> dict:
    r = pd.Series(returns).dropna()
    var = rolling_var_forecasts(r, confidence, window, method)
    realised_loss = -r.loc[var.index]
    exc = realised_loss > var
    n, x = len(var), int(exc.sum())
    pof = kupiec_pof(n, x, confidence)
    ind = christoffersen_independence(exc.values.astype(int))
    cc_stat = pof["statistic"] + ind["statistic"]
    cc_p = float(chi2.sf(cc_stat, 2))
    passed = pof["p_value"] > significance and ind["p_value"] > significance
    out = {
        "method": method, "confidence": confidence, "window": window, "test_observations": n,
        "exceptions": x, "expected_exceptions": n * (1 - confidence), "exception_rate": x / n,
        "exception_dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                            for d in var.index[exc.values]],
        "kupiec": pof, "christoffersen_independence": ind,
        "conditional_coverage": {"statistic": cc_stat, "p_value": cc_p},
        "significance": significance, "passed": bool(passed),
    }
    if abs(confidence - 0.99) < 1e-9 and n >= 250:
        out["basel_traffic_light_last_250d"] = basel_traffic_light(int(exc.values[-250:].sum()))
    return out
