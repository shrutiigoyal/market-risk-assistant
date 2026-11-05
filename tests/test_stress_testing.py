import pytest

from app.portfolio.stress_testing import DEFAULT_SCENARIOS, run_stress_test

MV = {"NIFTY50": 400_000, "GOVT_BOND": 300_000, "GOLD": 200_000, "USD_INR": 100_000}
CL = {"NIFTY50": "Equity", "GOVT_BOND": "Bonds", "GOLD": "Commodity", "USD_INR": "FX"}


def test_equity_crash_pnl_by_hand():
    r = {x["scenario"]: x for x in run_stress_test(MV, CL)}["Equity crash"]
    # -80,000 + 9,000 + 16,000 + 2,000
    assert r["pnl"] == pytest.approx(-53_000)
    assert r["pnl_pct"] == pytest.approx(-5.3)


def test_rally_is_positive_and_missing_class_gets_zero():
    r = run_stress_test(MV, CL, {"x": {"Equity": 0.1}})[0]
    assert r["pnl"] == pytest.approx(40_000)


def test_pnl_by_asset_sums_to_total():
    for r in run_stress_test(MV, CL):
        assert sum(r["pnl_by_asset"].values()) == pytest.approx(r["pnl"])


def test_empty_portfolio_rejected():
    with pytest.raises(ValueError):
        run_stress_test({"A": 0}, {"A": "Equity"})


def test_default_scenarios_present():
    assert {"Equity crash", "Inflation shock", "Market rally", "Global risk-off"} == set(DEFAULT_SCENARIOS)
