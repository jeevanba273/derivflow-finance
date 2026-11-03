"""Volatility surface: SVI fit per expiry, interpolation and an arbitrage check.

Run with:  python examples/volatility_surface.py
"""

import numpy as np

from derivflow import VolatilitySurface

# Quotes from a skewed smile at three expiries (illustrative numbers).
surface = VolatilitySurface(method="svi", reference_price=100.0)
for T, atm in ((0.25, 0.22), (0.5, 0.21), (1.0, 0.20)):
    for K in np.linspace(70, 130, 13):
        k = np.log(K / 100.0)
        surface.add_volatility_point(K, T, atm - 0.10 * k + 0.25 * k * k)
surface.build_surface()

for T, params in surface.smile_params.items():
    shown = ", ".join(f"{n}={params[n]:.4f}" for n in ("a", "b", "rho", "m", "sigma"))
    print(f"T={T:.2f}: {shown}")

print("\ninterpolated volatilities at T=0.75 (between fitted expiries):")
for K in (80.0, 100.0, 120.0):
    print(f"  K={K:5.1f}: {surface.interpolate(K, 0.75):.4f}")

strikes = np.linspace(70, 130, 25)
smile = np.array([surface.interpolate(K, 0.5) for K in strikes])
check = surface.check_arbitrage(strikes, smile, expiry=0.5, spot=100.0, r=0.03)
print(f"\nT=0.5 smile arbitrage free: {check['arbitrage_free']}")

spiked = smile.copy()
spiked[12] += 0.08
print(
    "same smile with a +8 vol point spike at K=100:",
    surface.check_arbitrage(strikes, spiked, expiry=0.5, spot=100.0, r=0.03)["arbitrage_free"],
)
