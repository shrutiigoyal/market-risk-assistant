import math
import unittest

from app import risk_tools as rt


class VarMaths(unittest.TestCase):
    def test_historical_var_known_values(self):
        # 100 returns: losses 1%..100%/100 -> 99% quantile is the 99th smallest loss
        returns = [-(i / 100) for i in range(1, 101)]
        self.assertAlmostEqual(rt.historical_var_pct(returns, 0.99), 0.99)
        self.assertAlmostEqual(rt.expected_shortfall_pct(returns, 0.99), 0.995)

    def test_var_never_negative(self):
        self.assertEqual(rt.historical_var_pct([0.01, 0.02, 0.03], 0.99), 0.0)

    def test_es_at_least_var_on_sample_data(self):
        r = rt.load_returns()
        for c in (0.95, 0.99):
            self.assertGreaterEqual(rt.expected_shortfall_pct(r, c), rt.historical_var_pct(r, c))


class ComputeVarTool(unittest.TestCase):
    def test_success_and_sqrt_time_scaling(self):
        one = rt.compute_var(1_000_000, 0.99, 1)
        ten = rt.compute_var(1_000_000, 0.99, 10)
        self.assertEqual(one["status"], "success")
        self.assertAlmostEqual(ten["var_usd"] / one["var_usd"], math.sqrt(10), places=2)

    def test_higher_confidence_means_higher_var(self):
        self.assertGreater(rt.compute_var(1e6, 0.99)["var_usd"], rt.compute_var(1e6, 0.95)["var_usd"])

    def test_parametric_method(self):
        self.assertEqual(rt.compute_var(1e6, 0.99, 1, "parametric")["method"], "parametric")

    def test_validation_errors(self):
        for kwargs in [dict(portfolio_value=-5), dict(portfolio_value=1e6, confidence_level=0.5),
                       dict(portfolio_value=1e6, horizon_days=0), dict(portfolio_value=1e6, method="magic"),
                       dict(portfolio_value="abc")]:
            self.assertEqual(rt.compute_var(**kwargs)["status"], "error", kwargs)


class LookupTools(unittest.TestCase):
    def test_positions_for_known_desk(self):
        r = rt.lookup_positions("Rates")
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["position_count"], 3)

    def test_unknown_and_malicious_desk(self):
        self.assertEqual(rt.lookup_positions("commodities")["status"], "error")
        self.assertEqual(rt.lookup_positions("rates'; DROP TABLE x;--")["status"], "error")

    def test_regulatory_lookup(self):
        self.assertEqual(rt.regulatory_lookup("What is the Volcker Rule?")["notes"][0]["topic"], "Volcker Rule")
        self.assertEqual(rt.regulatory_lookup("zebra migration")["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
