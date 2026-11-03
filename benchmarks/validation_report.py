"""
Validation tables: derivflow against published values, QuantLib and py_vollib.

Run with:

    pip install -e ".[validation]"
    python benchmarks/validation_report.py                       # Markdown to stdout
    python benchmarks/validation_report.py --output benchmarks/validation.md

The same comparisons are asserted, with tolerances, in tests/test_references.py,
tests/test_crosscheck_quantlib.py and tests/test_crosscheck_vollib.py. This
script reports the measured differences.
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np

import derivflow
from derivflow import (
    AsianOptions,
    BarrierOptions,
    BinomialTree,
    GreeksCalculator,
    HestonModel,
    MonteCarloEngine,
    black_scholes_price,
    implied_volatility,
)

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover
    ql = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from py_vollib.black_scholes_merton.implied_volatility import implied_volatility as pv_iv
except ImportError:  # pragma: no cover
    pv_iv = None

COMMAND = "python benchmarks/validation_report.py"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def published() -> list[list[str]]:
    g = GreeksCalculator().analytical_greeks
    cases = [
        (
            "Hull: call, S=42 K=40 T=0.5 r=10% sigma=20%",
            4.76,
            black_scholes_price(42, 40, 0.5, 0.1, 0.2),
            2,
        ),
        ("Hull: put, same inputs", 0.81, black_scholes_price(42, 40, 0.5, 0.1, 0.2, "put"), 2),
        (
            "Hull: American put, 5-step tree, S=K=50 r=10% sigma=40% T=5/12",
            4.49,
            BinomialTree(5).price(50, 50, 5 / 12, 0.1, 0.4, "put", american=True),
            2,
        ),
        (
            "Haug: call, S=60 K=65 T=0.25 r=8% sigma=30%",
            2.1334,
            black_scholes_price(60, 65, 0.25, 0.08, 0.3),
            4,
        ),
        (
            "Haug: put with q, S=100 K=95 T=0.5 r=10% q=5% sigma=20%",
            2.4648,
            black_scholes_price(100, 95, 0.5, 0.1, 0.2, "put", 0.05),
            4,
        ),
        (
            "Haug: call delta, b=0, S=105 K=100 T=0.5 r=10% sigma=36%",
            0.5946,
            g(105, 100, 0.5, 0.1, 0.36, "call", 0.1).delta,
            4,
        ),
        ("Haug: put delta, same inputs", -0.3566, g(105, 100, 0.5, 0.1, 0.36, "put", 0.1).delta, 4),
        (
            "Haug: gamma, S=55 K=60 T=0.75 r=b=10% sigma=30%",
            0.0278,
            g(55, 60, 0.75, 0.1, 0.3).gamma,
            4,
        ),
        (
            "Haug: put theta per year, S=430 K=405 T=1/12 r=7% q=5% sigma=20%",
            -31.1924,
            g(430, 405, 1 / 12, 0.07, 0.2, "put", 0.05).theta * 365,
            4,
        ),
        (
            "Haug: call rho per unit rate, S=72 K=75 T=1 r=b=9% sigma=19%",
            38.7325,
            g(72, 75, 1, 0.09, 0.19).rho * 100,
            4,
        ),
    ]
    rows = [
        [name, f"{ref:.{dp}f}", f"{val:.6f}", "yes" if round(val, dp) == ref else "NO"]
        for name, ref, val, dp in cases
    ]
    model = HestonModel()
    model.set_parameters(0.04, 4.0, 0.25, 1.0, -0.5)
    lewis = {
        80: 26.774758743998854,
        90: 20.933349000596710,
        100: 16.070154917028834,
        110: 12.132211516709845,
        120: 9.024913483457836,
    }
    worst = max(
        abs(model.analytic_price(100, k, 1.0, 0.01, "call", 0.02) - v) for k, v in lewis.items()
    )
    rows.append(
        [
            "Lewis: Heston calls K=80..120 (5 strikes), max abs difference",
            "15+ digits",
            f"{worst:.1e}",
            "yes" if worst < 1e-8 else "NO",
        ]
    )
    return rows


def parity() -> str:
    worst = 0.0
    n = 0
    for S, K, T, r, q, s in itertools.product(
        [50, 100, 200],
        [60, 100, 150],
        [1 / 365, 0.5, 5.0],
        [-0.01, 0.05],
        [0.0, 0.04],
        [0.05, 0.3, 1.2],
    ):
        c, p = (
            black_scholes_price(S, K, T, r, s, "call", q),
            black_scholes_price(S, K, T, r, s, "put", q),
        )
        worst = max(worst, abs((c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))))
        n += 1
    return (
        "Put-call parity C - P = S e^(-qT) - K e^(-rT): "
        f"max abs error {worst:.1e} over {n} parameter sets."
    )


def quantlib_rows() -> list[list[str]]:
    if ql is None:
        return [["QuantLib not installed", "-", "-", "-"]]
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    curve = lambda x: ql.YieldTermStructureHandle(ql.FlatForward(today, x, dc))  # noqa: E731

    def bsm(S, r, q, sig):
        vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sig, dc))
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)), curve(q), curve(r), vol
        )

    def payoff(ot, K):
        return ql.PlainVanillaPayoff(ql.Option.Call if ot == "call" else ql.Option.Put, K)

    rows = []
    # European prices and Greeks
    diffs = {k: 0.0 for k in ("price", "delta", "gamma", "vega", "theta", "rho")}
    n = 0
    for S, days, r, q, sig, ot in itertools.product(
        [80.0, 100.0, 125.0],
        [30, 365, 1095],
        [0.02, 0.07],
        [0.0, 0.03],
        [0.15, 0.5],
        ["call", "put"],
    ):
        opt = ql.VanillaOption(payoff(ot, 100.0), ql.EuropeanExercise(today + days))
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(bsm(S, r, q, sig)))
        T = days / 365
        g = GreeksCalculator().analytical_greeks(S, 100.0, T, r, sig, ot, q)
        diffs["price"] = max(
            diffs["price"], abs(black_scholes_price(S, 100.0, T, r, sig, ot, q) - opt.NPV())
        )
        diffs["delta"] = max(diffs["delta"], abs(g.delta - opt.delta()))
        diffs["gamma"] = max(diffs["gamma"], abs(g.gamma - opt.gamma()))
        diffs["vega"] = max(diffs["vega"], abs(g.vega * 100 - opt.vega()))
        diffs["theta"] = max(diffs["theta"], abs(g.theta * 365 - opt.theta()))
        diffs["rho"] = max(diffs["rho"], abs(g.rho * 100 - opt.rho()))
        n += 1
    rows.append(["European price", "AnalyticEuropeanEngine", n, f"{diffs['price']:.1e}"])
    for k in ("delta", "gamma", "vega", "theta", "rho"):
        rows.append(
            [f"European {k} (QuantLib units)", "AnalyticEuropeanEngine", n, f"{diffs[k]:.1e}"]
        )

    # American
    worst, n = 0.0, 0
    for S, K, days, r, q, sig, ot in [
        (36.0, 40.0, 365, 0.06, 0.0, 0.2, "put"),
        (100.0, 100.0, 182, 0.05, 0.0, 0.3, "put"),
        (100.0, 110.0, 730, 0.03, 0.0, 0.25, "put"),
        (100.0, 90.0, 365, 0.03, 0.08, 0.2, "call"),
        (80.0, 100.0, 90, 0.04, 0.02, 0.4, "put"),
    ]:
        opt = ql.VanillaOption(payoff(ot, K), ql.AmericanExercise(today, today + days))
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(bsm(S, r, q, sig), 2000, 2000))
        worst = max(
            worst,
            abs(
                BinomialTree(2000).price(S, K, days / 365, r, sig, ot, american=True, q=q)
                - opt.NPV()
            ),
        )
        n += 1
    rows.append(
        [
            "American option, 2000-step tree",
            "FdBlackScholesVanillaEngine 2000x2000",
            n,
            f"{worst:.1e}",
        ]
    )

    # Barrier
    worst, n = 0.0, 0
    types = {
        "down_and_in": ql.Barrier.DownIn,
        "down_and_out": ql.Barrier.DownOut,
        "up_and_in": ql.Barrier.UpIn,
        "up_and_out": ql.Barrier.UpOut,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for bt in types:
            for H in (90.0, 95.0) if bt.startswith("down") else (105.0, 110.0):
                for K in (90.0, 100.0, 110.0):
                    for ot in ("call", "put"):
                        opt = ql.BarrierOption(
                            types[bt], H, 0.0, payoff(ot, K), ql.EuropeanExercise(today + 182)
                        )
                        opt.setPricingEngine(ql.AnalyticBarrierEngine(bsm(100.0, 0.05, 0.0, 0.25)))
                        ours = (
                            BarrierOptions()
                            .analytical_price(100.0, K, H, 182 / 365, 0.05, 0.25, bt, ot)
                            .price
                        )
                        worst = max(worst, abs(ours - opt.NPV()))
                        n += 1
    rows.append(["Barrier, closed form", "AnalyticBarrierEngine", n, f"{worst:.1e}"])

    # Asian geometric
    worst_c = worst_d = 0.0
    n = 0
    fixings = [today + 30 * (i + 1) for i in range(12)]
    for K in (90.0, 100.0, 110.0):
        for ot in ("call", "put"):
            proc = bsm(100.0, 0.05, 0.0, 0.25)
            c = ql.ContinuousAveragingAsianOption(
                ql.Average.Geometric, payoff(ot, K), ql.EuropeanExercise(today + 360)
            )
            c.setPricingEngine(ql.AnalyticContinuousGeometricAveragePriceAsianEngine(proc))
            d = ql.DiscreteAveragingAsianOption(
                ql.Average.Geometric,
                1.0,
                0,
                fixings,
                payoff(ot, K),
                ql.EuropeanExercise(today + 360),
            )
            d.setPricingEngine(ql.AnalyticDiscreteGeometricAveragePriceAsianEngine(proc))
            a = AsianOptions()
            worst_c = max(
                worst_c,
                abs(
                    a.price_geometric_asian_analytical(
                        100, K, 360 / 365, 0.05, 0.25, ot, "continuous"
                    ).price
                    - c.NPV()
                ),
            )
            worst_d = max(
                worst_d,
                abs(
                    a.price_geometric_asian_analytical(
                        100, K, 360 / 365, 0.05, 0.25, ot, "discrete", 12
                    ).price
                    - d.NPV()
                ),
            )
            n += 1
    rows.append(
        [
            "Geometric Asian, continuous",
            "AnalyticContinuousGeometricAveragePriceAsianEngine",
            n,
            f"{worst_c:.1e}",
        ]
    )
    rows.append(
        [
            "Geometric Asian, 12 fixings",
            "AnalyticDiscreteGeometricAveragePriceAsianEngine",
            n,
            f"{worst_d:.1e}",
        ]
    )

    # Arithmetic Asian MC vs QuantLib MC
    opt = ql.DiscreteAveragingAsianOption(
        ql.Average.Arithmetic,
        0.0,
        0,
        fixings,
        payoff("call", 100.0),
        ql.EuropeanExercise(today + 360),
    )
    opt.setPricingEngine(
        ql.MCDiscreteArithmeticAPEngine(
            bsm(100.0, 0.05, 0.0, 0.25),
            "pseudorandom",
            controlVariate=True,
            requiredSamples=200_000,
            seed=42,
        )
    )
    ours = AsianOptions(num_sims=200_000, random_seed=42).price_arithmetic_asian_mc(
        100, 100, 360 / 365, 0.05, 0.25, "call", num_steps=12
    )
    z = (ours.price - opt.NPV()) / np.hypot(ours.std_error, opt.errorEstimate())
    rows.append(
        [
            "Arithmetic Asian, 12 fixings, Monte Carlo (200k paths each)",
            "MCDiscreteArithmeticAPEngine",
            1,
            f"{abs(ours.price - opt.NPV()):.1e} ({abs(z):.1f} combined s.e.)",
        ]
    )

    # Heston
    worst, n = 0.0, 0
    for params, days, K in itertools.product(
        [(0.04, 2.0, 0.04, 0.3, -0.7), (0.09, 1.0, 0.04, 0.9, -0.5), (0.02, 4.0, 0.06, 0.5, 0.3)],
        [30, 365, 1825],
        [70.0, 100.0, 140.0],
    ):
        hp = ql.HestonProcess(
            curve(0.03), curve(0.01), ql.QuoteHandle(ql.SimpleQuote(100.0)), *params
        )
        opt = ql.VanillaOption(payoff("call", K), ql.EuropeanExercise(today + days))
        opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp), 1e-12, 100_000))
        m = HestonModel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.set_parameters(*params)
        worst = max(
            worst, abs(m.analytic_price(100.0, K, days / 365, 0.03, "call", 0.01) - opt.NPV())
        )
        n += 1
    rows.append(
        [
            "Heston, analytic (3 parameter sets incl. one violating Feller)",
            "AnalyticHestonEngine",
            n,
            f"{worst:.1e}",
        ]
    )
    return rows


def vollib_row() -> list[list[str]]:
    if pv_iv is None:
        return [["py_vollib not installed", "-", "-"]]
    worst, n = 0.0, 0
    for T, K, s, ot in itertools.product(
        [7 / 365, 30 / 365, 0.25, 1.0, 3.0],
        [70.0, 90.0, 100.0, 110.0, 140.0],
        [0.1, 0.25, 0.6],
        ["call", "put"],
    ):
        price = black_scholes_price(100.0, K, T, 0.03, s, ot, 0.01)
        F = 100.0 * np.exp(0.02 * T)
        d1 = (np.log(F / K) + 0.5 * s * s * T) / (s * np.sqrt(T))
        vega = 100.0 * np.exp(-0.01 * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(T)
        if vega < 1e-6 * 100.0:
            continue
        worst = max(
            worst,
            abs(
                implied_volatility(price, 100.0, K, T, 0.03, ot, 0.01)
                - pv_iv(price, 100.0, K, T, 0.03, 0.01, ot[0])
            ),
        )
        n += 1
    return [
        ["Implied volatility vs Let's Be Rational (quotes with vega > 1e-6 S)", n, f"{worst:.1e}"]
    ]


def mc_coverage(seeds: int = 400, paths: int = 20_000) -> list[list[str]]:
    """How often the 95% confidence interval contains the closed-form price."""
    truth = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2)
    rows = []
    for label, kw in (
        ("plain", {}),
        ("antithetic", {"antithetic": True}),
        ("control variate", {"control_variate": True}),
    ):
        hits, z = 0, []
        for seed in range(seeds):
            res = MonteCarloEngine(paths, random_seed=seed).price(
                100.0, 100.0, 1.0, 0.05, 0.2, **kw
            )
            lo, hi = res["confidence_interval"]
            hits += lo < truth < hi
            z.append((res["price"] - truth) / res["std_error"])
        rows.append([label, f"{hits / seeds:.1%}", f"{np.mean(z):+.3f}", f"{np.std(z):.3f}"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", help="also write the Markdown report to this file")
    args = parser.parse_args()
    ql_version = ql.__version__ if ql else "not installed"
    versions = f"derivflow {derivflow.__version__}, QuantLib {ql_version}, numpy {np.__version__}"
    report = "\n".join(
        [
            "# derivflow validation results",
            "",
            f"Command: `{COMMAND}` ({versions})",
            "",
            "## Published reference values",
            "",
            table(["case", "reference", "derivflow", "matches to the quoted digits"], published()),
            "",
            parity(),
            "",
            "## QuantLib",
            "",
            "Maximum absolute difference; maturities are whole days on Actual/365 Fixed.",
            "",
            table(["quantity", "QuantLib engine", "cases", "max abs difference"], quantlib_rows()),
            "",
            "## Monte Carlo confidence intervals",
            "",
            "ATM European call (S=K=100, T=1, r=5%, sigma=20%), 20,000 draws, seeds 0 to 399. "
            "z = (estimate - closed form) / standard error should have mean 0 "
            "and standard deviation 1.",
            "",
            table(["method", "95% CI coverage", "mean z", "std of z"], mc_coverage()),
            "",
            "## py_vollib",
            "",
            table(["quantity", "cases", "max abs difference"], vollib_row()),
            "",
        ]
    )
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)


if __name__ == "__main__":
    main()
