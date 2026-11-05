"""Generate deterministic synthetic multi-asset data (no real market data).

Equity follows a GARCH(1,1) process with Student-t shocks plus a short
high-volatility regime, so volatility clusters and VaR exceptions can cluster.
Other assets are correlated through a Cholesky factor.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
N = 756  # ~3 business years
rng = np.random.default_rng(7)

assets = ["NIFTY50", "GOVT_BOND", "GOLD", "USD_INR"]
corr = np.array([
    [1.00, -0.15, 0.05, 0.20],
    [-0.15, 1.00, 0.10, -0.05],
    [0.05, 0.10, 1.00, 0.15],
    [0.20, -0.05, 0.15, 1.00],
])
chol = np.linalg.cholesky(corr)
nu = 5
z = rng.standard_t(nu, size=(N, 4)) / np.sqrt(nu / (nu - 2))  # unit variance
z = z @ chol.T

mu = np.array([0.0005, 0.0002, 0.0003, 0.0000])
vol = np.array([0.0, 0.0025, 0.0085, 0.0030])

# equity: GARCH(1,1), unconditional daily vol about 0.9%
omega, alpha, beta = 1.2e-6, 0.09, 0.89
sigma2 = np.empty(N)
eq = np.empty(N)
sigma2[0] = omega / (1 - alpha - beta)
for t in range(N):
    if t > 0:
        sigma2[t] = omega + alpha * eq[t - 1] ** 2 + beta * sigma2[t - 1]
    s = np.sqrt(sigma2[t]) * (2.2 if 520 <= t < 535 else 1.0)   # stress regime
    eq[t] = mu[0] + s * z[t, 0]

ret = np.column_stack([eq] + [mu[i] + vol[i] * z[:, i] for i in range(1, 4)])
dates = pd.bdate_range("2023-01-02", periods=N)
df = pd.DataFrame(np.round(ret, 6), index=dates, columns=assets)
df.index.name = "date"
df.to_csv(OUT / "sample_asset_returns.csv")

pd.DataFrame({
    "asset": assets,
    "asset_class": ["Equity", "Bonds", "Commodity", "FX"],
    "weight": [0.40, 0.30, 0.20, 0.10],
    "market_value": [400000, 300000, 200000, 100000],
}).to_csv(OUT / "sample_portfolio.csv", index=False)
print(f"wrote {N} rows x {len(assets)} assets")
