"""Published reference values (no third-party library needed).

Hull: Options, Futures, and Other Derivatives, Black-Scholes-Merton worked
example (S=42, K=40, r=10%, sigma=20%, T=0.5: call 4.76, put 0.81).

Haug: The Complete Guide to Option Pricing Formulas, 2nd ed. (2007), worked
examples for the generalised Black-Scholes-Merton model in chapters 1 and 2.
Haug writes the model with a cost of carry b; with a dividend yield q,
b = r - q (b = 0 is the Black-76 futures case, i.e. q = r). Haug quotes theta
per year and rho per unit rate; here they are converted from this package's
per-day and per-point units.
"""

import itertools

import numpy as np
import pytest

from derivflow_finance import GreeksCalculator, black_scholes_price, price_european_option

greeks = GreeksCalculator().analytical_greeks


def test_hull_worked_example_to_two_decimals():
    assert round(price_european_option(42, 40, 0.5, 0.10, 0.20, "call"), 2) == 4.76
    assert round(price_european_option(42, 40, 0.5, 0.10, 0.20, "put"), 2) == 0.81


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        # Black-Scholes call: S=60, K=65, T=0.25, r=8%, sigma=30%
        ((60, 65, 0.25, 0.08, 0.30, "call", 0.0), 2.1334),
        # Merton put with dividend yield: S=100, K=95, T=0.5, r=10%, q=5%, sigma=20%
        ((100, 95, 0.5, 0.10, 0.20, "put", 0.05), 2.4648),
    ],
)
def test_haug_prices(args, expected):
    assert round(black_scholes_price(*args), 4) == expected


def test_haug_greeks():
    # Delta, b = 0: S=105, K=100, T=0.5, r=10%, sigma=36%
    assert round(greeks(105, 100, 0.5, 0.10, 0.36, "call", 0.10).delta, 4) == 0.5946
    assert round(greeks(105, 100, 0.5, 0.10, 0.36, "put", 0.10).delta, 4) == -0.3566
    # Gamma: S=55, K=60, T=0.75, r=b=10%, sigma=30%
    assert round(greeks(55, 60, 0.75, 0.10, 0.30, "call", 0.0).gamma, 4) == 0.0278
    # Put theta per year: S=430, K=405, T=1/12, r=7%, b=2% (q=5%), sigma=20%
    assert round(greeks(430, 405, 1 / 12, 0.07, 0.20, "put", 0.05).theta * 365, 4) == -31.1924
    # Call rho per unit rate: S=72, K=75, T=1, r=b=9%, sigma=19%
    assert round(greeks(72, 75, 1.0, 0.09, 0.19, "call", 0.0).rho * 100, 4) == 38.7325


GRID = list(
    itertools.product(
        [50.0, 100.0, 200.0],
        [60.0, 100.0, 150.0],
        [1 / 365, 0.5, 5.0],
        [-0.01, 0.05],
        [0.0, 0.04],
        [0.05, 0.3, 1.2],
    )
)


@pytest.mark.parametrize(("S", "K", "T", "r", "q", "sigma"), GRID)
def test_put_call_parity_to_1e10(S, K, T, r, q, sigma):
    c = black_scholes_price(S, K, T, r, sigma, "call", q)
    p = black_scholes_price(S, K, T, r, sigma, "put", q)
    assert abs((c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-10
