import numpy as np
import pytest

from app.portfolio.volatility_models import ewma_variance_series, ewma_volatility, fit_garch11, garch_forecast


def simulate_garch(n, omega, alpha, beta, seed=0):
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        out[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * out[t] ** 2 + beta * s2
    return out


def test_ewma_recursion_by_hand():
    r = np.r_[np.full(30, 0.01) * np.tile([1, -1], 15), 0.03]
    s2 = ewma_variance_series(r, 0.94)
    assert s2[-1] == pytest.approx(0.94 * s2[-2] + 0.06 * 0.03 ** 2)


def test_ewma_uses_only_past_data():
    base = np.random.default_rng(1).normal(0, 0.01, 200)
    changed = base.copy(); changed[-1] = 0.5
    a, b = ewma_variance_series(base), ewma_variance_series(changed)
    assert a[-2] == b[-2]            # forecast for the last day is unaffected by that day's return
    assert b[-1] > a[-1]


def test_ewma_reacts_to_shock():
    r = np.random.default_rng(2).normal(0, 0.01, 300)
    before = ewma_volatility(r)["annualized_vol_forecast"]
    after = ewma_volatility(np.r_[r, 0.08])["annualized_vol_forecast"]
    assert after > before


def test_ewma_bad_lambda():
    with pytest.raises(ValueError):
        ewma_variance_series(np.zeros(100), 1.0)


def test_garch_recovers_parameters():
    r = simulate_garch(8000, 2e-6, 0.08, 0.90, seed=5)
    g = fit_garch11(r)
    assert g["converged"]
    assert g["alpha"] == pytest.approx(0.08, abs=0.03)
    assert g["beta"] == pytest.approx(0.90, abs=0.04)
    assert g["persistence"] < 1
    assert g["long_run_daily_vol"] == pytest.approx(np.sqrt(2e-6 / 0.02), rel=0.25)


def test_garch_forecast_mean_reverts():
    r = simulate_garch(3000, 2e-6, 0.08, 0.90, seed=6)
    g = fit_garch11(r)
    path = np.array(garch_forecast(g, 250)["daily_vol_path"])
    assert abs(path[-1] - g["long_run_daily_vol"]) < abs(path[0] - g["long_run_daily_vol"]) + 1e-12


def test_garch_matches_arch_package_if_installed():
    arch = pytest.importorskip("arch")
    r = simulate_garch(3000, 2e-6, 0.08, 0.90, seed=7)
    ours = fit_garch11(r)
    res = arch.arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
    assert ours["alpha"] == pytest.approx(res.params["alpha[1]"], abs=0.01)
    assert ours["beta"] == pytest.approx(res.params["beta[1]"], abs=0.02)


def test_garch_needs_data():
    with pytest.raises(ValueError):
        fit_garch11(np.zeros(100))
