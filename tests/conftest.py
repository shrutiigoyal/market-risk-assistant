import numpy as np
import pandas as pd
import pytest

from app.portfolio.portfolio_engine import load_asset_returns, load_portfolio, weights_from_portfolio


@pytest.fixture(scope="session")
def asset_returns():
    return load_asset_returns()


@pytest.fixture(scope="session")
def portfolio():
    return load_portfolio()


@pytest.fixture(scope="session")
def weights(portfolio):
    return weights_from_portfolio(portfolio)


@pytest.fixture
def normal_returns():
    return pd.Series(np.random.default_rng(1).normal(0.0004, 0.01, 2000))
