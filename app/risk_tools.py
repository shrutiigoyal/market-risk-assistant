"""Tools the specialist agents can call. Plain Python functions with docstrings
(ADK builds the tool schema from the signature + docstring).

All numbers come from deterministic code, never from the LLM.
Data is synthetic and for demonstration only.
"""
from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from statistics import NormalDist, mean, stdev

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VALID_METHODS = {"historical", "parametric"}


# ---------------------------------------------------------------- VaR maths
def load_returns(path: Path | None = None) -> list[float]:
    with open(path or DATA_DIR / "sample_returns.csv", newline="") as f:
        return [float(r["daily_return"]) for r in csv.DictReader(f)]


def historical_var_pct(returns: list[float], confidence: float) -> float:
    """1-day historical-simulation VaR as a positive loss fraction."""
    losses = sorted(-r for r in returns)
    idx = min(len(losses) - 1, max(0, math.ceil(confidence * len(losses)) - 1))
    return max(losses[idx], 0.0)


def expected_shortfall_pct(returns: list[float], confidence: float) -> float:
    """Average loss in the tail at or beyond the VaR quantile."""
    var = historical_var_pct(returns, confidence)
    tail = [-r for r in returns if -r >= var]
    return mean(tail) if tail else var


def parametric_var_pct(returns: list[float], confidence: float) -> float:
    """1-day variance-covariance (normal) VaR as a positive loss fraction."""
    z = NormalDist().inv_cdf(confidence)
    return max(z * stdev(returns) - mean(returns), 0.0)


# ------------------------------------------------------------ agent tools
def compute_var(portfolio_value: float, confidence_level: float = 0.99,
                horizon_days: int = 1, method: str = "historical") -> dict:
    """Compute Value at Risk (VaR) and Expected Shortfall for a portfolio using
    500 days of bundled sample returns.

    Args:
        portfolio_value: Portfolio market value in USD (must be > 0).
        confidence_level: Confidence level between 0.90 and 0.999 (e.g. 0.99).
        horizon_days: Holding period in days, 1 to 30. Scaled with square-root-of-time.
        method: "historical" (historical simulation) or "parametric" (normal).

    Returns:
        dict with status, var_usd, expected_shortfall_usd and the assumptions used.
    """
    try:
        portfolio_value = float(portfolio_value)
        confidence_level = float(confidence_level)
        horizon_days = int(horizon_days)
    except (TypeError, ValueError):
        return {"status": "error", "error_message": "Inputs must be numeric."}
    method = (method or "").lower()
    if portfolio_value <= 0:
        return {"status": "error", "error_message": "portfolio_value must be positive."}
    if not 0.90 <= confidence_level <= 0.999:
        return {"status": "error", "error_message": "confidence_level must be between 0.90 and 0.999."}
    if not 1 <= horizon_days <= 30:
        return {"status": "error", "error_message": "horizon_days must be between 1 and 30."}
    if method not in VALID_METHODS:
        return {"status": "error", "error_message": f"method must be one of {sorted(VALID_METHODS)}."}

    returns = load_returns()
    var1 = (historical_var_pct(returns, confidence_level) if method == "historical"
            else parametric_var_pct(returns, confidence_level))
    es1 = expected_shortfall_pct(returns, confidence_level)
    scale = math.sqrt(horizon_days)
    return {
        "status": "success",
        "method": method,
        "confidence_level": confidence_level,
        "horizon_days": horizon_days,
        "var_usd": round(var1 * scale * portfolio_value, 2),
        "expected_shortfall_usd": round(es1 * scale * portfolio_value, 2),
        "assumptions": [
            f"{len(returns)} days of synthetic daily returns",
            "square-root-of-time scaling for multi-day horizons",
            "expected shortfall computed historically at the same confidence level",
            "illustrative only, not a regulatory capital figure",
        ],
    }


def _positions_from_csv(desk: str) -> list[dict]:
    with open(DATA_DIR / "sample_positions.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["desk"].lower() == desk.lower()]


def _positions_from_bigquery(table: str, desk: str) -> list[dict]:
    if not re.fullmatch(r"[\w.-]+", table):          # table name comes from env, still validate
        raise ValueError("Invalid BQ_POSITIONS_TABLE")
    from google.cloud import bigquery                 # lazy import: not needed for local/CSV mode
    client = bigquery.Client()
    job = client.query(
        f"SELECT desk, instrument, quantity, market_value_usd, pnl_usd FROM `{table}` "
        "WHERE LOWER(desk) = LOWER(@desk) LIMIT 100",
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("desk", "STRING", desk)]),
    )
    return [dict(r) for r in job.result()]


def lookup_positions(desk: str) -> dict:
    """Look up positions and P&L for a trading desk (rates, credit, equities, fx).

    Args:
        desk: Desk name, one of rates, credit, equities, fx.

    Returns:
        dict with status, the positions, total market value and total P&L in USD.
    """
    desk = (desk or "").strip()
    if not re.fullmatch(r"[A-Za-z ]{2,30}", desk):
        return {"status": "error", "error_message": "Desk must be a short plain name, e.g. 'rates'."}
    table = os.getenv("BQ_POSITIONS_TABLE")
    try:
        rows = _positions_from_bigquery(table, desk) if table else _positions_from_csv(desk)
    except Exception as exc:                          # surfaced to the agent, logged by caller
        return {"status": "error", "error_message": f"Data source error: {type(exc).__name__}"}
    if not rows:
        return {"status": "error", "error_message": f"No positions found for desk '{desk}'."}
    mv = sum(float(r["market_value_usd"]) for r in rows)
    pnl = sum(float(r["pnl_usd"]) for r in rows)
    return {"status": "success", "desk": desk.lower(), "source": "bigquery" if table else "sample_csv",
            "position_count": len(rows), "total_market_value_usd": round(mv, 2),
            "total_pnl_usd": round(pnl, 2), "positions": rows[:10]}


_KB = [
    {"topic": "Volcker Rule", "keywords": {"volcker", "proprietary", "619", "dodd-frank", "dodd"},
     "text": "The Volcker Rule (Section 619 of the Dodd-Frank Act) generally restricts US banking entities "
             "from proprietary trading and from owning or sponsoring certain covered funds. Permitted "
             "activities include underwriting, market-making-related activities, and risk-mitigating hedging."},
    {"topic": "Volcker metrics reporting", "keywords": {"metrics", "reporting", "report", "quarterly"},
     "text": "Banking entities with significant trading activity must compute and report quantitative "
             "metrics for trading desks to regulators on a periodic basis (quarterly for the largest firms). "
             "Metrics include risk and position limits and usage, VaR and stress VaR, P&L attribution, "
             "inventory turnover and aging, and customer-facing trade ratios."},
    {"topic": "Value at Risk", "keywords": {"var", "value", "risk"},
     "text": "VaR estimates the loss that should not be exceeded over a given horizon at a given confidence "
             "level (for example 1-day 99%). Common methods: historical simulation, variance-covariance "
             "(parametric) and Monte Carlo. VaR says nothing about the size of losses beyond the threshold."},
    {"topic": "Expected Shortfall and FRTB", "keywords": {"es", "shortfall", "frtb", "basel", "tail"},
     "text": "Expected Shortfall (ES) is the average loss beyond the VaR threshold. Under the Basel "
             "Fundamental Review of the Trading Book (FRTB), ES at 97.5% replaces 99% VaR as the main "
             "internal-model measure of market risk capital."},
    {"topic": "VaR backtesting", "keywords": {"backtest", "backtesting", "exceptions", "traffic"},
     "text": "VaR backtesting compares daily P&L with the VaR estimate and counts exceptions (days the loss "
             "exceeded VaR). Under the Basel traffic-light approach, for 99% 1-day VaR over 250 days, "
             "0-4 exceptions is the green zone, 5-9 yellow, 10 or more red."},
]


def regulatory_lookup(query: str) -> dict:
    """Look up short reference notes on market-risk regulation and risk concepts
    (Volcker Rule, metrics reporting, VaR, Expected Shortfall / FRTB, backtesting).

    Args:
        query: The regulatory or risk concept to look up.

    Returns:
        dict with status and matching notes. Informational only, not legal advice.
    """
    words = set(re.findall(r"[a-z0-9\-]+", (query or "").lower()))
    scored = sorted(((len(words & e["keywords"]), e) for e in _KB), key=lambda t: -t[0])
    hits = [{"topic": e["topic"], "note": e["text"]} for s, e in scored if s > 0][:3]
    if not hits:
        return {"status": "not_found", "message": "No matching reference note. Do not guess; say so."}
    return {"status": "success", "notes": hits,
            "disclaimer": "General educational reference, not legal or compliance advice."}
