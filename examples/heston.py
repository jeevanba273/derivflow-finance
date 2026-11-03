"""Heston model: analytic price, Monte Carlo check and a calibration round trip.

Run with:  python examples/heston.py
"""

from derivflow import HestonModel, black_scholes_price, implied_volatility

S, r, q, T = 100.0, 0.03, 0.0, 1.0
model = HestonModel()
# v0 and theta are variances: 0.04 is a 20% volatility.
model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.35, rho=-0.7)

print("Heston smile (1 year): price and Black-Scholes implied volatility")
for K in (80.0, 90.0, 100.0, 110.0, 120.0):
    price = model.analytic_price(S, K, T, r, "call", q)
    print(
        f"  K={K:5.1f}  call {price:8.4f}  implied vol {implied_volatility(price, S, K, T, r):.4f}"
    )

mc = model.monte_carlo_price(S, 100.0, T, r, "call", num_sims=200_000, num_steps=100, random_seed=1)
analytic = model.analytic_price(S, 100.0, T, r, "call")
lo, hi = mc.confidence_interval
print(
    f"\nATM call: analytic {analytic:.4f}, QE Monte Carlo {mc.price:.4f} (95% CI {lo:.4f}-{hi:.4f})"
)
print(f"Black-Scholes at 20% vol: {black_scholes_price(S, 100.0, T, r, 0.2):.4f}")

# Calibrate to quotes priced by the model itself (a self-consistency check).
quotes = [
    {"K": K, "T": t, "price": model.analytic_price(S, K, t, r, "call")}
    for t in (0.25, 1.0)
    for K in (85.0, 100.0, 115.0)
]
result = HestonModel().calibrate_to_market(quotes, S=S, r=r)
params = ", ".join(f"{k}={v:.4f}" for k, v in result["parameters"].items())
print(f"\ncalibrated: {params}; rmse {result['rmse']:.1e}")
