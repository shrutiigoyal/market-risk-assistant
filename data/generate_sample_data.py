"""Generate deterministic synthetic data (no real market or bank data)."""
import csv, random
from pathlib import Path

random.seed(42)
out = Path(__file__).parent

with open(out / "sample_returns.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["day", "daily_return"])
    for d in range(1, 501):
        r = random.gauss(0.0003, 0.010)
        if random.random() < 0.02:          # occasional shock -> fatter tails
            r += random.gauss(0, 0.03)
        w.writerow([d, round(r, 6)])

desks = {"rates": ["UST 10Y", "UST 2Y", "IRS 5Y"], "credit": ["IG CDX", "HY CDX", "Corp Bond A"],
         "equities": ["SPX Future", "AAPL", "MSFT"], "fx": ["EURUSD", "USDINR", "USDJPY"]}
with open(out / "sample_positions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["desk", "instrument", "quantity", "market_value_usd", "pnl_usd"])
    for desk, insts in desks.items():
        for i in insts:
            q = random.randint(100, 5000)
            mv = round(q * random.uniform(50, 400), 2)
            pnl = round(random.gauss(0, 0.01) * mv, 2)
            w.writerow([desk, i, q, mv, pnl])
