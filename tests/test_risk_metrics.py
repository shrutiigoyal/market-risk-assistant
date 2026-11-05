import numpy as np
import pytest
from scipy.stats import norm

from app.portfolio.risk_metrics import historical_var_es, parametric_var_es, scale_horizon


def test_historical_var_positive_and_es_ge_var(normal_returns):
    r = historical_var_es(normal_returns, 1_000_000, 0.99)
    assert r["var"] > 0 and r["es"] >= r["var"]


def test_historical_known_example():
    returns = np.linspace(-0.05, 0.05, 101)         # losses -0.05 .. 0.05 in steps of 0.001
    r = historical_var_es(returns, 100.0, 0.95)
    assert r["var_pct"] == pytest.approx(0.045)      # quantile at index 0.95 * 100 = 95
    assert r["es_pct"] == pytest.approx(np.mean([0.045, 0.046, 0.047, 0.048, 0.049, 0.05]))
    assert r["var"] == pytest.approx(4.5)


def test_parametric_matches_normal_formula():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 5000)
    r = parametric_var_es(x, 1.0, 0.99)
    s = x.std(ddof=1)
    assert r["var_pct"] == pytest.approx(norm.ppf(0.99) * s)
    assert r["es_pct"] == pytest.approx(s * norm.pdf(norm.ppf(0.99)) / 0.01)
    assert r["es_pct"] > r["var_pct"]


def test_parametric_close_to_historical_for_normal_data(normal_returns):
    h = historical_var_es(normal_returns, 1.0, 0.99)["var_pct"]
    p = parametric_var_es(normal_returns, 1.0, 0.99, use_mean=True)["var_pct"]
    assert h == pytest.approx(p, rel=0.15)


def test_parametric_understates_fat_tails():
    x = np.random.default_rng(4).standard_t(3, 20000) * 0.005
    assert historical_var_es(x, 1, 0.999)["es_pct"] > parametric_var_es(x, 1, 0.999)["es_pct"]


@pytest.mark.parametrize("c", [0, 1, -0.1, 1.5])
def test_invalid_confidence_rejected(normal_returns, c):
    with pytest.raises(ValueError, match="confidence"):
        historical_var_es(normal_returns, 1.0, c)
    with pytest.raises(ValueError, match="confidence"):
        parametric_var_es(normal_returns, 1.0, c)


def test_insufficient_and_invalid_data_rejected():
    with pytest.raises(ValueError, match="Insufficient"):
        historical_var_es([0.01] * 10, 1.0)
    with pytest.raises(ValueError, match="invalid"):
        historical_var_es([0.01] * 29 + [float("nan")], 1.0)


def test_sqrt_time():
    assert scale_horizon(100, 10) == pytest.approx(100 * 10 ** 0.5)
