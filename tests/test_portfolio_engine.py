import pandas as pd
import pytest

from app.portfolio.portfolio_engine import (calculate_portfolio_returns, covariance_matrix,
                                            load_portfolio, portfolio_volatility)
from app.portfolio.risk_attribution import calculate_risk_contribution, concentration_metrics


def test_sample_data_size(asset_returns):
    assert len(asset_returns) >= 500


def test_valid_weights_accepted(asset_returns):
    r = calculate_portfolio_returns(asset_returns, {"NIFTY50": 0.5, "GOLD": 0.5})
    assert r.iloc[0] == pytest.approx(0.5 * asset_returns["NIFTY50"].iloc[0] + 0.5 * asset_returns["GOLD"].iloc[0])


def test_weights_not_summing_to_one_rejected(asset_returns):
    with pytest.raises(ValueError, match="sum to 1"):
        calculate_portfolio_returns(asset_returns, {"NIFTY50": 0.6, "GOLD": 0.6})


def test_negative_weight_rejected(asset_returns):
    with pytest.raises(ValueError, match="Negative"):
        calculate_portfolio_returns(asset_returns, {"NIFTY50": 1.2, "GOLD": -0.2})


def test_missing_column_rejected(asset_returns):
    with pytest.raises(ValueError, match="Missing"):
        calculate_portfolio_returns(asset_returns, {"NIFTY50": 0.5, "BITCOIN": 0.5})


def test_empty_returns_rejected():
    with pytest.raises(ValueError):
        calculate_portfolio_returns(pd.DataFrame({"A": []}), {"A": 1.0})


def test_portfolio_csv_validation(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("asset,asset_class,market_value\nA,Equity,100\nA,Bonds,50\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_portfolio(p)


def test_risk_contributions_sum_to_portfolio_vol(asset_returns, weights):
    cov = covariance_matrix(asset_returns, list(weights))
    rc = calculate_risk_contribution(weights, cov)
    assert sum(v["risk_contribution"] for v in rc.values()) == pytest.approx(portfolio_volatility(weights, cov))
    assert sum(v["percentage"] for v in rc.values()) == pytest.approx(100.0)


def test_portfolio_vol_matches_series_std(asset_returns, weights):
    cov = covariance_matrix(asset_returns, list(weights))
    pr = calculate_portfolio_returns(asset_returns, weights)
    assert portfolio_volatility(weights, cov) == pytest.approx(pr.std(ddof=1))


def test_equity_dominates_risk_and_diversification_above_one(asset_returns, weights):
    cov = covariance_matrix(asset_returns, list(weights))
    rc = calculate_risk_contribution(weights, cov)
    assert rc["NIFTY50"]["percentage"] > rc["NIFTY50"]["weight"] * 100   # risk share > capital share
    assert concentration_metrics(weights, cov)["diversification_ratio"] > 1
