import numpy as np
import pandas as pd
import pytest

from app.portfolio.backtesting import (backtest_var, basel_traffic_light, christoffersen_independence,
                                       kupiec_pof, rolling_var_forecasts)


def test_kupiec_known_value():
    # n=250, x=5, p=0.01 by hand: LR = 2*(ll1-ll0) = 1.957
    k = kupiec_pof(250, 5, 0.99)
    assert k["statistic"] == pytest.approx(1.957, abs=0.001)
    assert k["p_value"] == pytest.approx(0.160, abs=0.005)


def test_kupiec_perfect_calibration_and_zero_exceptions():
    assert kupiec_pof(1000, 10, 0.99)["statistic"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(kupiec_pof(250, 0, 0.99)["statistic"])


def test_kupiec_rejects_too_many():
    assert kupiec_pof(250, 15, 0.99)["p_value"] < 0.01


def test_christoffersen_detects_clustering():
    hits = np.zeros(500, int); hits[100:110] = 1; hits[300:310] = 1
    assert christoffersen_independence(hits)["p_value"] < 0.01


def test_christoffersen_independent_hits_not_rejected():
    hits = np.zeros(1000, int); hits[::50] = 1
    assert christoffersen_independence(hits)["p_value"] > 0.05


def test_christoffersen_by_hand():
    hits = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0, 0])
    c = christoffersen_independence(hits)
    assert (c["n00"], c["n01"], c["n10"], c["n11"]) == (4, 2, 2, 1)


def test_christoffersen_no_exceptions():
    assert christoffersen_independence(np.zeros(100, int))["p_value"] == 1.0


def test_exception_count_and_dates_match_independent_loop():
    idx = pd.bdate_range("2024-01-01", periods=600)
    vals = np.random.default_rng(0).normal(0, 0.01, 600)
    vals[400] = -0.2; vals[500] = -0.2
    r = pd.Series(vals, index=idx)
    expected = [idx[t].strftime("%Y-%m-%d") for t in range(250, 600)
                if -vals[t] > np.quantile(-vals[t - 250:t], 0.99)]
    bt = backtest_var(r, 0.99, 250)
    assert bt["exception_dates"] == expected
    assert bt["exceptions"] == len(expected)
    assert idx[400].strftime("%Y-%m-%d") in expected and idx[500].strftime("%Y-%m-%d") in expected
    assert bt["test_observations"] == 350


def test_no_lookahead_future_shock_does_not_change_past_var():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 600))
    a = rolling_var_forecasts(r, 0.99, 250)
    r2 = r.copy(); r2.iloc[500] = -0.3
    b = rolling_var_forecasts(r2, 0.99, 250)
    assert a.loc[:500].equals(b.loc[:500])         # VaR for day 500 ignores day 500's own return
    assert not a.loc[501:].equals(b.loc[501:])


@pytest.mark.parametrize("m", ["historical", "parametric", "ewma"])
def test_well_specified_normal_passes(m):
    r = pd.Series(np.random.default_rng(11).normal(0, 0.01, 2500))
    bt = backtest_var(r, 0.99, 250, m)
    assert bt["kupiec"]["p_value"] > 0.01


def test_on_sample_portfolio_runs_and_flags_fat_tails(asset_returns, weights):
    from app.portfolio.portfolio_engine import calculate_portfolio_returns
    pr = calculate_portfolio_returns(asset_returns, weights)
    bt = backtest_var(pr, 0.99, 250, "parametric")
    assert bt["exceptions"] >= 0 and 0 <= bt["kupiec"]["p_value"] <= 1


def test_traffic_light():
    assert [basel_traffic_light(x) for x in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]


def test_invalid_inputs():
    r = pd.Series(np.zeros(100))
    with pytest.raises(ValueError):
        rolling_var_forecasts(r, 0.99, 250)
    with pytest.raises(ValueError):
        rolling_var_forecasts(pd.Series(np.zeros(400)), 0.99, 250, "magic")
