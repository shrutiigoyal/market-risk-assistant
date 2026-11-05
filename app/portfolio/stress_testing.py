"""Deterministic scenario P&L. Shocks are percentage moves in position VALUE
(so for FX a positive shock is a gain on the position)."""
from __future__ import annotations

# shocks by asset class
DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "Equity crash":    {"Equity": -0.20, "Bonds": 0.03,  "Commodity": 0.08,  "FX": 0.02},
    "Inflation shock": {"Equity": -0.10, "Bonds": -0.08, "Commodity": 0.12,  "FX": 0.04},
    "Market rally":    {"Equity": 0.15,  "Bonds": -0.02, "Commodity": -0.03, "FX": -0.01},
    "Global risk-off": {"Equity": -0.15, "Bonds": 0.05,  "Commodity": 0.10,  "FX": 0.05},
}


def run_stress_test(market_values: dict, asset_classes: dict, scenarios: dict | None = None) -> list[dict]:
    """market_values: asset -> value; asset_classes: asset -> class; scenarios: name -> {class: shock}."""
    scenarios = scenarios or DEFAULT_SCENARIOS
    total = sum(market_values.values())
    if total <= 0:
        raise ValueError("Portfolio value must be positive")
    results = []
    for name, shocks in scenarios.items():
        by_asset = {a: v * shocks.get(asset_classes[a], 0.0) for a, v in market_values.items()}
        pnl = sum(by_asset.values())
        results.append({"scenario": name, "pnl": pnl, "pnl_pct": pnl / total * 100,
                        "pnl_by_asset": by_asset})
    return results
