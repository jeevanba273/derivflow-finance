"""Cross-checks against QuantLib (skipped when the QuantLib package is absent).

QuantLib is an independent C++ implementation. Maturities are whole days on
an Actual/365 Fixed basis, so T = days / 365 matches exactly. Unit
conversions: QuantLib's vega and rho are per unit of volatility and rate and
its theta is per year; this package reports per point and per calendar day.
"""

import itertools
import warnings

import numpy as np
import pytest

ql = pytest.importorskip("QuantLib")

from derivflow_finance import (  # noqa: E402
    AsianOptions,
    BarrierOptions,
    BinomialTree,
    GreeksCalculator,
    HestonModel,
    MonteCarloEngine,
    black_scholes_price,
    implied_volatility,
)

TODAY = ql.Date(15, 1, 2025)
DC = ql.Actual365Fixed()


@pytest.fixture(autouse=True)
def _evaluation_date():
    ql.Settings.instance().evaluationDate = TODAY
    yield


def _curve(rate):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, rate, DC))


def _bsm(S, r, q, sigma):
    vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), sigma, DC))
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(S)), _curve(q), _curve(r), vol
    )


def _payoff(option_type, K):
    return ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, K)


EUROPEAN = list(
    itertools.product(
        [80.0, 100.0, 125.0],
        [30, 365, 1095],
        [0.02, 0.07],
        [0.0, 0.03],
        [0.15, 0.5],
        ["call", "put"],
    )
)


@pytest.mark.parametrize(("S", "days", "r", "q", "sigma", "ot"), EUROPEAN)
def test_european_price_and_greeks(S, days, r, q, sigma, ot):
    K, T = 100.0, days / 365
    opt = ql.VanillaOption(_payoff(ot, K), ql.EuropeanExercise(TODAY + days))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(_bsm(S, r, q, sigma)))
    g = GreeksCalculator().analytical_greeks(S, K, T, r, sigma, ot, q)
    assert black_scholes_price(S, K, T, r, sigma, ot, q) == pytest.approx(opt.NPV(), abs=1e-10)
    assert g.delta == pytest.approx(opt.delta(), abs=1e-10)
    assert g.gamma == pytest.approx(opt.gamma(), abs=1e-10)
    assert g.vega * 100 == pytest.approx(opt.vega(), abs=1e-8)
    assert g.rho * 100 == pytest.approx(opt.rho(), abs=1e-8)
    assert g.theta * 365 == pytest.approx(opt.theta(), abs=1e-8)


AMERICAN = [
    (36.0, 40.0, 365, 0.06, 0.0, 0.2, "put"),
    (100.0, 100.0, 182, 0.05, 0.0, 0.3, "put"),
    (100.0, 110.0, 730, 0.03, 0.0, 0.25, "put"),
    (100.0, 90.0, 365, 0.03, 0.08, 0.2, "call"),
    (80.0, 100.0, 90, 0.04, 0.02, 0.4, "put"),
]


@pytest.mark.parametrize(("S", "K", "days", "r", "q", "sigma", "ot"), AMERICAN)
def test_american_tree_against_quantlib_finite_differences(S, K, days, r, q, sigma, ot):
    """Two different discretisations (2000-step tree, 2000x2000 FD grid): both
    carry O(1e-4) error, hence the 2e-3 tolerance."""
    opt = ql.VanillaOption(_payoff(ot, K), ql.AmericanExercise(TODAY, TODAY + days))
    opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(_bsm(S, r, q, sigma), 2000, 2000))
    ours = BinomialTree(2000).price(S, K, days / 365, r, sigma, ot, american=True, q=q)
    assert ours == pytest.approx(opt.NPV(), abs=2e-3)


BARRIER_TYPES = {
    "down_and_in": ql.Barrier.DownIn,
    "down_and_out": ql.Barrier.DownOut,
    "up_and_in": ql.Barrier.UpIn,
    "up_and_out": ql.Barrier.UpOut,
}
BARRIER_CASES = [
    (bt, H, K, ot)
    for bt in BARRIER_TYPES
    for H in ((90.0, 95.0) if bt.startswith("down") else (105.0, 110.0))
    for K in (90.0, 100.0, 110.0)
    for ot in ("call", "put")
]


@pytest.mark.parametrize(("bt", "H", "K", "ot"), BARRIER_CASES)
def test_barrier_closed_form(bt, H, K, ot):
    S, r, sigma, days = 100.0, 0.05, 0.25, 182
    opt = ql.BarrierOption(
        BARRIER_TYPES[bt], H, 0.0, _payoff(ot, K), ql.EuropeanExercise(TODAY + days)
    )
    opt.setPricingEngine(ql.AnalyticBarrierEngine(_bsm(S, r, 0.0, sigma)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = BarrierOptions().analytical_price(S, K, H, days / 365, r, sigma, bt, ot).price
    assert ours == pytest.approx(opt.NPV(), abs=1e-10)


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_geometric_asian_closed_forms(K, ot):
    S, r, sigma, days, n = 100.0, 0.05, 0.25, 360, 12
    T = days / 365
    proc = _bsm(S, r, 0.0, sigma)
    asian = AsianOptions()
    cont = ql.ContinuousAveragingAsianOption(
        ql.Average.Geometric, _payoff(ot, K), ql.EuropeanExercise(TODAY + days)
    )
    cont.setPricingEngine(ql.AnalyticContinuousGeometricAveragePriceAsianEngine(proc))
    ours_cont = asian.price_geometric_asian_analytical(S, K, T, r, sigma, ot, "continuous").price
    assert ours_cont == pytest.approx(cont.NPV(), abs=1e-10)

    fixings = [TODAY + 30 * (i + 1) for i in range(n)]
    disc = ql.DiscreteAveragingAsianOption(
        ql.Average.Geometric, 1.0, 0, fixings, _payoff(ot, K), ql.EuropeanExercise(TODAY + days)
    )
    disc.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
    ours_disc = asian.price_geometric_asian_analytical(
        S, K, T, r, sigma, ot, "discrete", num_fixings=n
    ).price
    assert ours_disc == pytest.approx(disc.NPV(), abs=1e-10)


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_arithmetic_asian_monte_carlo_against_quantlib_monte_carlo(K, ot):
    S, r, sigma, days, n = 100.0, 0.05, 0.25, 360, 12
    fixings = [TODAY + 30 * (i + 1) for i in range(n)]
    opt = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        0.0,
        0,
        fixings,
        _payoff(ot, K),
        ql.EuropeanExercise(TODAY + days),
    )
    engine = ql.MCDiscreteArithmeticAPEngine(
        _bsm(S, r, 0.0, sigma),
        "pseudorandom",
        controlVariate=True,
        requiredSamples=200_000,
        seed=42,
    )
    opt.setPricingEngine(engine)
    ours = AsianOptions(num_sims=200_000, random_seed=42).price_arithmetic_asian_mc(
        S, K, days / 365, r, sigma, ot, num_steps=n
    )
    combined = np.hypot(ours.std_error, opt.errorEstimate())
    assert abs(ours.price - opt.NPV()) < 4 * combined


HESTON = list(
    itertools.product(
        [
            (0.04, 2.0, 0.04, 0.3, -0.7),
            (0.09, 1.0, 0.04, 0.9, -0.5),  # Feller condition violated
            (0.02, 4.0, 0.06, 0.5, 0.3),
        ],
        [30, 365, 1825],
        [70.0, 100.0, 140.0],
    )
)


@pytest.mark.parametrize(("params", "days", "K"), HESTON)
def test_heston_analytic(params, days, K):
    S, r, q = 100.0, 0.03, 0.01
    v0, kappa, theta, sigma, rho = params
    process = ql.HestonProcess(_curve(r), _curve(q), ql.QuoteHandle(ql.SimpleQuote(S)), *params)
    opt = ql.VanillaOption(_payoff("call", K), ql.EuropeanExercise(TODAY + days))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 1e-12, 100_000))
    model = HestonModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.set_parameters(v0, kappa, theta, sigma, rho)
    assert model.analytic_price(S, K, days / 365, r, "call", q) == pytest.approx(
        opt.NPV(), abs=1e-8
    )


@pytest.mark.parametrize(
    ("K", "days", "sigma", "ot"),
    list(itertools.product([70.0, 100.0, 130.0], [7, 90, 730], [0.1, 0.4, 1.0], ["call", "put"])),
)
def test_implied_volatility_at_least_as_accurate_as_quantlib(K, days, sigma, ot):
    """The true volatility is known, so both solvers are scored against it.

    QuantLib's solver works on the quoted price; on low-vega quotes (deep in or
    out of the money, short expiry) it can stop far from the truth or at its
    lower bound. Quotes whose time value is lost in rounding have no
    identifiable volatility and must raise.
    """
    S, r, q = 100.0, 0.03, 0.01
    T = days / 365
    price = black_scholes_price(S, K, T, r, sigma, ot, q)
    try:
        ours = implied_volatility(price, S, K, T, r, ot, q)
    except ValueError as exc:
        assert "identifiable" in str(exc) or "intrinsic" in str(exc)
        return
    opt = ql.VanillaOption(_payoff(ot, K), ql.EuropeanExercise(TODAY + days))
    try:
        ref = opt.impliedVolatility(price, _bsm(S, r, q, 0.2), 1e-12, 10_000, 1e-7, 4.0)
    except RuntimeError:
        ref = float("nan")
    our_error = abs(ours - sigma)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(T)
    if vega > 1e-6 * S:  # elsewhere the price barely determines the volatility
        assert our_error < 1e-8
    if np.isfinite(ref):
        assert our_error <= abs(ref - sigma) + 1e-9


def test_monte_carlo_engine_against_quantlib_analytic():
    S, K, r, q, sigma, days = 100.0, 105.0, 0.04, 0.01, 0.3, 365
    opt = ql.VanillaOption(_payoff("call", K), ql.EuropeanExercise(TODAY + days))
    opt.setPricingEngine(ql.AnalyticEuropeanEngine(_bsm(S, r, q, sigma)))
    res = MonteCarloEngine(400_000, random_seed=8).price(
        S, K, days / 365, r, sigma, "call", q=q, control_variate=True
    )
    assert abs(res["price"] - opt.NPV()) < 4 * res["std_error"]
