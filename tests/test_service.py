from app.portfolio import service as s


def test_analyze_portfolio():
    r = s.analyze_portfolio()
    assert r["status"] == "success" and r["portfolio_value"] == 1_000_000
    assert abs(sum(v["percentage"] for v in r["risk_contribution"].values()) - 100) < 0.1


def test_calculate_var_both_methods_and_horizon():
    d1 = s.calculate_var(0.99, 1, "both")
    d10 = s.calculate_var(0.99, 10, "historical")
    assert d1["results"]["historical"]["expected_shortfall"] >= d1["results"]["historical"]["var"]
    assert abs(d10["results"]["historical"]["var"] / d1["results"]["historical"]["var"] - 10 ** 0.5) < 0.01


def test_tool_input_validation():
    assert s.calculate_var(0.5)["status"] == "error"
    assert s.calculate_var(0.99, 99)["status"] == "error"
    assert s.calculate_var(0.99, 1, "magic")["status"] == "error"
    assert s.forecast_volatility("magic")["status"] == "error"
    assert s.backtest_var(0.99, 10)["status"] == "error"


def test_forecast_and_backtest_and_stress_and_report():
    f = s.forecast_volatility("both", 10)
    assert f["status"] == "success" and f["garch"]["converged"]
    b = s.backtest_var(0.99, 250, "ewma")
    assert b["status"] == "success" and "kupiec" in b
    assert s.run_stress_tests()["worst_scenario"]
    rep = s.generate_risk_report()
    assert rep["status"] == "success" and rep["risk_drivers"]["recent_vs_full_ratio"] > 0


def test_lookup_reference():
    assert s.lookup_reference("what is kupiec backtesting")["status"] == "success"
    assert s.lookup_reference("pasta recipe")["status"] == "not_found"


def test_bad_data_path_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_CSV", "/nonexistent.csv")
    assert s.analyze_portfolio()["status"] == "error"
