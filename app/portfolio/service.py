"""Agent-facing tools. Plain functions with docstrings (ADK builds the tool schema
from signature + docstring). Every number comes from the deterministic engine in
this package; the LLM only chooses tools and explains results.

Data: bundled synthetic CSVs, or set PORTFOLIO_CSV / ASSET_RETURNS_CSV to your own.
"""
from __future__ import annotations

import os
import re

import numpy as np

from . import backtesting, volatility_models as vm
from .portfolio_engine import (calculate_portfolio_returns, covariance_matrix, load_asset_returns,
                               load_portfolio, portfolio_volatility, weights_from_portfolio)
from .risk_attribution import calculate_risk_contribution, concentration_metrics
from .risk_metrics import historical_var_es, parametric_var_es, scale_horizon
from .stress_testing import run_stress_test

CURRENCY = os.getenv("BASE_CURRENCY", "INR")
RECENT_DAYS = 60


def _load():
    pf = load_portfolio(os.getenv("PORTFOLIO_CSV") or None)
    ar = load_asset_returns(os.getenv("ASSET_RETURNS_CSV") or None)
    w = weights_from_portfolio(pf)
    return pf, ar, w, calculate_portfolio_returns(ar, w), float(pf["market_value"].sum())


def _err(msg: str) -> dict:
    return {"status": "error", "error_message": msg}


def _conf(c) -> float:
    c = float(c)
    if not 0.90 <= c <= 0.999:
        raise ValueError("confidence_level must be between 0.90 and 0.999.")
    return c


def analyze_portfolio() -> dict:
    """Describe the portfolio: allocation, annualised volatility, correlation matrix,
    and each asset's contribution to portfolio risk.

    Returns:
        dict with status, portfolio_value, allocation, annualized_volatility,
        correlation, risk_contribution, concentration and assumptions.
    """
    try:
        pf, ar, w, pr, value = _load()
        cov = covariance_matrix(ar, list(w))
        rc = calculate_risk_contribution(w, cov)
        return {
            "status": "success", "currency": CURRENCY, "portfolio_value": round(value, 2),
            "allocation": [{"asset": r.asset, "asset_class": r.asset_class,
                            "market_value": float(r.market_value), "weight": round(w[r.asset], 4)}
                           for r in pf.itertuples()],
            "annualized_volatility": round(portfolio_volatility(w, cov, annualize=True), 4),
            "correlation": ar[list(w)].corr().round(3).to_dict(),
            "risk_contribution": {a: {"weight": round(v["weight"], 4), "percentage": round(v["percentage"], 2)}
                                  for a, v in rc.items()},
            "largest_risk_contributor": max(rc, key=lambda a: rc[a]["percentage"]),
            "concentration": {k: round(v, 4) for k, v in concentration_metrics(w, cov).items()},
            "assumptions": [f"{len(pr)} daily observations", "constant weights (daily rebalanced)",
                            "sample covariance, long-only", "synthetic data unless PORTFOLIO_CSV is set"],
        }
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def get_risk_contribution() -> dict:
    """Which assets contribute most to portfolio volatility, and how diversified is the portfolio?

    Returns:
        dict with per-asset weight, marginal and component risk contribution (percentage of total
        volatility) plus diversification ratio.
    """
    try:
        pf, ar, w, _, _ = _load()
        cov = covariance_matrix(ar, list(w))
        rc = calculate_risk_contribution(w, cov)
        return {"status": "success",
                "risk_contribution": {a: {k: round(v, 6) for k, v in d.items()} for a, d in rc.items()},
                "concentration": {k: round(v, 4) for k, v in concentration_metrics(w, cov).items()},
                "note": "Percentages sum to 100. Assets with low weight but high percentage are risk-dominant."}
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def calculate_var(confidence_level: float = 0.99, horizon_days: int = 1, method: str = "both") -> dict:
    """Portfolio Value at Risk and Expected Shortfall.

    Args:
        confidence_level: 0.90 to 0.999 (default 0.99).
        horizon_days: 1 to 30. Multi-day figures use square-root-of-time scaling (approximation).
        method: "historical", "parametric" (normal) or "both" (default).

    Returns:
        dict with VaR and ES in currency and as a percent of portfolio value, plus assumptions.
    """
    try:
        c, h, method = _conf(confidence_level), int(horizon_days), (method or "").lower()
        if not 1 <= h <= 30:
            return _err("horizon_days must be between 1 and 30.")
        if method not in {"historical", "parametric", "both"}:
            return _err("method must be historical, parametric or both.")
        _, _, _, pr, value = _load()
        fns = {"historical": historical_var_es, "parametric": parametric_var_es}
        results = {}
        for m in (fns if method == "both" else [method]):
            r = fns[m](pr.values, value, c)
            results[m] = {"var": round(scale_horizon(r["var"], h), 2),
                          "expected_shortfall": round(scale_horizon(r["es"], h), 2),
                          "var_pct_of_portfolio": round(scale_horizon(r["var_pct"], h) * 100, 3),
                          "es_pct_of_portfolio": round(scale_horizon(r["es_pct"], h) * 100, 3)}
        return {"status": "success", "currency": CURRENCY, "portfolio_value": round(value, 2),
                "confidence_level": c, "horizon_days": h, "results": results,
                "assumptions": [f"{len(pr)} daily portfolio returns", "long-only, constant weights",
                                "parametric method assumes normal returns (zero mean) and understates fat tails",
                                "square-root-of-time scaling for horizons > 1 day (approximation)",
                                "illustrative only, not a regulatory capital figure"]}
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def forecast_volatility(model: str = "both", horizon_days: int = 10) -> dict:
    """Forecast portfolio volatility with EWMA (lambda 0.94) and/or GARCH(1,1).

    Args:
        model: "ewma", "garch" or "both" (default).
        horizon_days: GARCH multi-day horizon, 1 to 60.

    Returns:
        dict with annualised volatility forecasts, GARCH parameters and convergence status,
        and the sample volatility for comparison.
    """
    try:
        model, h = (model or "").lower(), int(horizon_days)
        if model not in {"ewma", "garch", "both"} or not 1 <= h <= 60:
            return _err("model must be ewma, garch or both; horizon_days 1 to 60.")
        _, _, _, pr, _ = _load()
        out = {"status": "success",
               "sample_annualized_vol": round(float(pr.std(ddof=1) * np.sqrt(vm.TRADING_DAYS)), 4)}
        if model in {"ewma", "both"}:
            e = vm.ewma_volatility(pr.values)
            out["ewma"] = {"annualized_vol_forecast": round(e["annualized_vol_forecast"], 4),
                           "lambda": e["lambda"], "initialisation": e["initialisation"]}
        if model in {"garch", "both"}:
            g = vm.fit_garch11(pr.values)
            f = vm.garch_forecast(g, h)
            out["garch"] = {"omega": g["omega"], "alpha": round(g["alpha"], 4), "beta": round(g["beta"], 4),
                            "persistence": round(g["persistence"], 4),
                            "long_run_annualized_vol": round(g["long_run_daily_vol"] * np.sqrt(vm.TRADING_DAYS), 4),
                            "next_day_annualized_vol": round(f["next_day_annualized_vol"], 4),
                            f"{h}_day_horizon_vol": round(f["horizon_vol"], 4),
                            "converged": g["converged"]}
        out["note"] = "Forecasts above the sample/long-run volatility signal a currently elevated-risk regime."
        return out
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def backtest_var(confidence_level: float = 0.99, window: int = 250, method: str = "historical") -> dict:
    """Out-of-sample backtest of a VaR model (Kupiec and Christoffersen tests).

    Args:
        confidence_level: 0.90 to 0.999 (default 0.99).
        window: rolling estimation window in days, 60 to 500 (default 250).
        method: "historical", "parametric" or "ewma".

    Returns:
        dict with exception count/rate/dates, Kupiec and Christoffersen statistics and p-values,
        pass/fail at 5% significance and (for 99%) the Basel traffic-light zone.
    """
    try:
        c, w_ = _conf(confidence_level), int(window)
        if not 60 <= w_ <= 500:
            return _err("window must be between 60 and 500.")
        _, _, _, pr, _ = _load()
        r = backtesting.backtest_var(pr, c, w_, (method or "").lower())
        r["exception_dates"] = r["exception_dates"][:25]
        r["status"] = "success"
        return r
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def run_stress_tests() -> dict:
    """Apply hypothetical market shocks (equity crash, inflation shock, rally, risk-off) to the portfolio.

    Returns:
        dict with scenario P&L in currency and percent, and the worst scenario.
    """
    try:
        pf, *_ = _load()
        res = run_stress_test(dict(zip(pf["asset"], pf["market_value"])),
                              dict(zip(pf["asset"], pf["asset_class"])))
        rows = [{"scenario": r["scenario"], "pnl": round(r["pnl"], 2), "pnl_pct": round(r["pnl_pct"], 2),
                 "pnl_by_asset": {a: round(v, 2) for a, v in r["pnl_by_asset"].items()}} for r in res]
        return {"status": "success", "currency": CURRENCY, "scenarios": rows,
                "worst_scenario": min(rows, key=lambda r: r["pnl"])["scenario"],
                "note": "Hypothetical instantaneous shocks to position value, not forecasts."}
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


def generate_risk_report() -> dict:
    """One structured snapshot of all risk metrics for explanation: allocation, VaR/ES, volatility
    forecasts, why risk is moving (recent vs full-history), stress tests and VaR model validation.

    Returns:
        dict combining the other tools' outputs plus risk-driver diagnostics.
    """
    try:
        _, _, w, pr, value = _load()
        rep = {"status": "success",
               "portfolio": analyze_portfolio(),
               "var_99_1d": calculate_var(0.99, 1, "both"),
               "volatility": forecast_volatility("both", 10),
               "stress": run_stress_tests(),
               "validation": {m: {k: v for k, v in backtest_var(0.99, 250, m).items()
                                  if k in {"exceptions", "expected_exceptions", "passed", "status",
                                           "basel_traffic_light_last_250d"}}
                              for m in ("historical", "ewma")}}
        rec = pr.iloc[-RECENT_DAYS:]
        full = historical_var_es(pr.values, value, 0.99)["var"]
        recent = historical_var_es(rec.values, value, 0.99)["var"]
        rep["risk_drivers"] = {
            "recent_window_days": RECENT_DAYS,
            "recent_hist_var_99": round(recent, 2), "full_sample_hist_var_99": round(full, 2),
            "recent_vs_full_ratio": round(recent / full, 3),
            "recent_annualized_vol": round(float(rec.std(ddof=1) * np.sqrt(252)), 4),
            "full_annualized_vol": round(float(pr.std(ddof=1) * np.sqrt(252)), 4)}
        return rep
    except Exception as exc:
        return _err(f"{type(exc).__name__}: {exc}")


_KB = [
    {"topic": "Value at Risk", "keywords": {"var", "value", "risk"},
     "text": "VaR is the loss not expected to be exceeded over a horizon at a confidence level (e.g. 1-day 99%). "
             "Methods: historical simulation, variance-covariance (parametric), Monte Carlo. It says nothing "
             "about how bad losses are beyond the threshold."},
    {"topic": "Expected Shortfall", "keywords": {"es", "shortfall", "tail", "cvar"},
     "text": "Expected Shortfall is the average loss on days when loss exceeds VaR. It is always >= VaR and "
             "captures tail severity; it is also sub-additive, unlike VaR."},
    {"topic": "Basel FRTB", "keywords": {"frtb", "basel", "capital", "trading", "book"},
     "text": "Under the Basel Fundamental Review of the Trading Book (FRTB), 97.5% Expected Shortfall replaces "
             "99% VaR as the main internal-model measure for market-risk capital, with liquidity-horizon scaling."},
    {"topic": "VaR backtesting", "keywords": {"backtest", "backtesting", "exceptions", "kupiec", "christoffersen", "traffic"},
     "text": "Backtesting counts days where loss exceeded VaR. Kupiec POF tests whether the exception rate equals "
             "1 - confidence; Christoffersen tests whether exceptions cluster. Basel traffic light for 99% VaR over "
             "250 days: 0-4 green, 5-9 yellow, 10+ red."},
    {"topic": "EWMA and GARCH", "keywords": {"ewma", "garch", "volatility", "clustering", "forecast"},
     "text": "EWMA (RiskMetrics, lambda 0.94) weights recent squared returns more. GARCH(1,1) estimates "
             "sigma2 = omega + alpha*eps2 + beta*sigma2_prev by maximum likelihood and mean-reverts to a long-run "
             "variance omega/(1-alpha-beta); alpha+beta near 1 means persistent volatility shocks."},
    {"topic": "Risk contribution", "keywords": {"contribution", "attribution", "diversification", "concentration", "marginal"},
     "text": "Euler risk contribution splits portfolio volatility into per-asset parts that sum to the total. "
             "An asset can be 40% of capital but 80% of risk if it is volatile and correlated with the rest."},
    {"topic": "Stress testing", "keywords": {"stress", "scenario", "shock"},
     "text": "Stress tests apply hypothetical shocks to positions. They complement VaR because they do not rely on "
             "historical distributions or correlations holding."},
]


def lookup_reference(query: str) -> dict:
    """Look up short reference notes on risk concepts and regulation (VaR, Expected Shortfall, FRTB/Basel,
    backtesting, EWMA/GARCH, risk contribution, stress testing).

    Args:
        query: The concept to look up.

    Returns:
        dict with status and matching notes. Educational only, not advice.
    """
    words = set(re.findall(r"[a-z0-9\-]+", (query or "").lower()))
    scored = sorted(((len(words & e["keywords"]), e) for e in _KB), key=lambda t: -t[0])
    hits = [{"topic": e["topic"], "note": e["text"]} for s, e in scored if s > 0][:3]
    if not hits:
        return {"status": "not_found", "message": "No matching reference note. Do not guess; say so."}
    return {"status": "success", "notes": hits, "disclaimer": "Educational reference, not advice."}
