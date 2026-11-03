"""
Benchmarks for derivflow, with QuantLib, py_vollib and plain NumPy/SciPy as baselines.

Run with:

    pip install -e ".[validation]"
    python benchmarks/run_benchmarks.py                    # Markdown to stdout
    python benchmarks/run_benchmarks.py --output benchmarks/results.md --json benchmarks/results.json
    python benchmarks/run_benchmarks.py --quick            # smaller sizes, for a smoke test

Every timing is repeated (``--reps``, default 7) after one warm-up run; tables
show the median and the [min, max] range. All libraries get identical inputs
on the same machine. Baselines that are not installed are reported as
"not installed". Comparisons that are not like-for-like are flagged in the
notes under each table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
from scipy.stats import norm

import derivflow
from derivflow import (
    BinomialTree,
    BlackScholesAnalytical,
    GreeksCalculator,
    HestonModel,
    MonteCarloEngine,
    black_scholes_price,
    implied_volatility,
    implied_volatility_vectorized,
)

try:
    import QuantLib as ql
except ImportError:  # pragma: no cover
    ql = None

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from py_vollib.black_scholes_merton import black_scholes_merton as pv_price
        from py_vollib.black_scholes_merton.implied_volatility import (
            implied_volatility as pv_iv,
        )
except ImportError:  # pragma: no cover
    pv_price = pv_iv = None

COMMAND = "python benchmarks/run_benchmarks.py"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**30
        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        pass
    return None


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment() -> dict[str, Any]:
    ram = _ram_gb()
    return {
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(ram, 1) if ram else None,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "derivflow": derivflow.__version__,
        "numpy": np.__version__,
        "scipy": _version("scipy"),
        "QuantLib": _version("QuantLib"),
        "py_vollib": _version("py_vollib"),
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def timed(fn: Callable[[], Any], reps: int) -> dict[str, float]:
    fn()  # warm-up
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def rate_cell(t: dict[str, float], n: int) -> str:
    med = n / t["median"]
    lo, hi = n / t["max"], n / t["min"]
    return f"{med:,.0f} [{lo:,.0f}, {hi:,.0f}]"


def time_cell(t: dict[str, float], scale: float = 1e6, unit: str = "us") -> str:
    return f"{t['median'] * scale:,.1f} [{t['min'] * scale:,.1f}, {t['max'] * scale:,.1f}] {unit}"


def err_cell(errors: np.ndarray, ref: np.ndarray | None = None) -> tuple[str, str]:
    errors = np.abs(np.asarray(errors, dtype=float))
    abs_max = f"{errors.max():.1e}"
    if ref is None:
        return abs_max, "-"
    rel = errors / np.maximum(np.abs(ref), 1e-300)
    return abs_max, f"{rel.max():.1e}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def random_inputs(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "S": rng.uniform(80, 120, n),
        "K": rng.uniform(80, 120, n),
        "T": rng.uniform(0.05, 2.0, n),
        "r": rng.uniform(0.0, 0.06, n),
        "q": rng.uniform(0.0, 0.03, n),
        "sigma": rng.uniform(0.1, 0.6, n),
    }


def numpy_scipy_bs(S, K, T, r, sigma, q):
    """Textbook NumPy/SciPy Black-Scholes-Merton call (the plain baseline)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


class QLEuropean:
    """One QuantLib option per strike, repriced by changing quote values.

    Spot, volatility and rates are SimpleQuotes, so a repricing is a quote
    update plus NPV(); object construction is excluded from the timing.
    Maturities are whole days (Actual/365 Fixed).
    """

    def __init__(self, days: int = 365):
        self.today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = self.today
        dc = ql.Actual365Fixed()
        self.spot, self.vol = ql.SimpleQuote(100.0), ql.SimpleQuote(0.2)
        self.r, self.q = ql.SimpleQuote(0.03), ql.SimpleQuote(0.01)
        ts = lambda x: ql.YieldTermStructureHandle(  # noqa: E731
            ql.FlatForward(self.today, ql.QuoteHandle(x), dc)
        )
        vol = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.today, ql.NullCalendar(), ql.QuoteHandle(self.vol), dc)
        )
        self.process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(self.spot), ts(self.q), ts(self.r), vol
        )
        self.engine = ql.AnalyticEuropeanEngine(self.process)
        self.days = days
        self.T = days / 365

    def option(self, K: float, option_type: str = "call") -> Any:
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, K
        )
        opt = ql.VanillaOption(payoff, ql.EuropeanExercise(self.today + self.days))
        opt.setPricingEngine(self.engine)
        return opt


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_pricing(reps: int, quick: bool) -> dict[str, Any]:
    n_vec = 100_000 if quick else 1_000_000
    n_scalar = 2_000 if quick else 20_000
    x = random_inputs(n_vec)
    rows, notes = [], []

    t = timed(
        lambda: black_scholes_price(x["S"], x["K"], x["T"], x["r"], x["sigma"], "call", x["q"]),
        reps,
    )
    rows.append(["derivflow `black_scholes_price` (vectorised)", f"{n_vec:,}", rate_cell(t, n_vec)])
    t = timed(lambda: numpy_scipy_bs(x["S"], x["K"], x["T"], x["r"], x["sigma"], x["q"]), reps)
    rows.append(["plain NumPy/SciPy formula (vectorised)", f"{n_vec:,}", rate_cell(t, n_vec)])

    # Scalar: fixed strike and maturity, spot and vol vary (QuantLib reprices via quotes).
    xs = {k: v[:n_scalar] for k, v in x.items()}
    K, T = 100.0, 1.0
    bs = BlackScholesAnalytical()
    t = timed(
        lambda: [
            bs.price(s, K, T, 0.03, v, "call", 0.01)
            for s, v in zip(xs["S"], xs["sigma"], strict=True)
        ],
        reps,
    )
    rows.append(
        [
            "derivflow `BlackScholesAnalytical.price` (scalar loop)",
            f"{n_scalar:,}",
            rate_cell(t, n_scalar),
        ]
    )
    if pv_price is not None:
        t = timed(
            lambda: [
                pv_price("c", s, K, T, 0.03, v, 0.01)
                for s, v in zip(xs["S"], xs["sigma"], strict=True)
            ],
            reps,
        )
        rows.append(
            [
                "py_vollib `black_scholes_merton` (scalar loop)",
                f"{n_scalar:,}",
                rate_cell(t, n_scalar),
            ]
        )
    if ql is not None:
        env = QLEuropean(365)
        opt = env.option(K)

        def ql_loop():
            for s, v in zip(xs["S"], xs["sigma"], strict=True):
                env.spot.setValue(float(s))
                env.vol.setValue(float(v))
                opt.NPV()

        t = timed(ql_loop, reps)
        rows.append(
            [
                "QuantLib `AnalyticEuropeanEngine` (scalar loop)",
                f"{n_scalar:,}",
                rate_cell(t, n_scalar),
            ]
        )
        notes.append(
            "QuantLib reprices a prebuilt option after updating two quotes, so each call "
            "includes its observer notifications; the Python loops include interpreter overhead."
        )

    # Accuracy against QuantLib on 2,000 options with varying strikes and maturities.
    acc_rows = []
    if ql is not None:
        n_acc = 2_000
        y = random_inputs(n_acc, seed=1)
        days = np.maximum(1, np.round(y["T"] * 365)).astype(int)
        ours, ref, ref_np = [], [], []
        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        for i in range(n_acc):
            Ti = days[i] / 365
            h = lambda rate: ql.YieldTermStructureHandle(ql.FlatForward(today, float(rate), dc))  # noqa: E731
            proc = ql.BlackScholesMertonProcess(
                ql.QuoteHandle(ql.SimpleQuote(float(y["S"][i]))),
                h(y["q"][i]),
                h(y["r"][i]),
                ql.BlackVolTermStructureHandle(
                    ql.BlackConstantVol(today, ql.NullCalendar(), float(y["sigma"][i]), dc)
                ),
            )
            opt = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, float(y["K"][i])),
                ql.EuropeanExercise(today + int(days[i])),
            )
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            ref.append(opt.NPV())
            ours.append(
                black_scholes_price(
                    y["S"][i], y["K"][i], Ti, y["r"][i], y["sigma"][i], "call", y["q"][i]
                )
            )
            ref_np.append(
                numpy_scipy_bs(y["S"][i], y["K"][i], Ti, y["r"][i], y["sigma"][i], y["q"][i])
            )
        ref_a = np.array(ref)
        a, rel = err_cell(np.array(ours) - ref_a, ref_a)
        acc_rows.append(["derivflow vs QuantLib", f"{n_acc:,}", a, rel])
        a, rel = err_cell(np.array(ref_np) - ref_a, ref_a)
        acc_rows.append(["NumPy/SciPy formula vs QuantLib", f"{n_acc:,}", a, rel])
    return {
        "rows": rows,
        "notes": notes,
        "accuracy": acc_rows,
        "n_vec": n_vec,
        "n_scalar": n_scalar,
    }


def iv_inputs(n: int, seed: int = 2) -> dict[str, np.ndarray]:
    """Quotes generated from known volatilities, moneyness 0.7-1.3, T 1 week to 2 years.
    Quotes whose time value is below 1e-12 of the price are dropped (no identifiable
    volatility; see derivflow.volatility.implied_vol)."""
    rng = np.random.default_rng(seed)
    S = np.full(n, 100.0)
    K = S * rng.uniform(0.7, 1.3, n)
    T = rng.uniform(7 / 365, 2.0, n)
    r, q = np.full(n, 0.03), np.full(n, 0.01)
    sigma = rng.uniform(0.08, 0.8, n)
    ot = np.where(rng.random(n) < 0.5, "call", "put")
    price = np.array(
        [black_scholes_price(*a, o, qq) for *a, o, qq in zip(S, K, T, r, sigma, ot, q, strict=True)]
    )
    F = S * np.exp((r - q) * T)
    otm = np.where(K >= F, "call", "put")
    tv = np.array(
        [
            black_scholes_price(s, k, t, rr, v, o, qq)
            for s, k, t, rr, v, o, qq in zip(S, K, T, r, sigma, otm, q, strict=True)
        ]
    )
    keep = tv > 1e-12 * price
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    out = {
        "S": S,
        "K": K,
        "T": T,
        "r": r,
        "q": q,
        "sigma": sigma,
        "ot": ot,
        "price": price,
        "vega": vega,
    }
    return {k: v[keep] for k, v in out.items()}


def bench_implied_vol(reps: int, quick: bool) -> dict[str, Any]:
    n_scalar = 500 if quick else 5_000
    n_vec = 10_000 if quick else 100_000
    x = iv_inputs(max(n_scalar, n_vec))
    xs = {k: v[:n_scalar] for k, v in x.items()}
    rows, notes = [], []

    def run_ours():
        return [
            implied_volatility(p, s, k, t, r, o, q)
            for p, s, k, t, r, o, q in zip(
                xs["price"], xs["S"], xs["K"], xs["T"], xs["r"], xs["ot"], xs["q"], strict=True
            )
        ]

    def accuracy(vols, data):
        err = np.abs(np.asarray(vols) - data["sigma"])
        good = data["vega"] > 1e-6 * data["S"]
        return f"{err.max():.1e}", f"{err[good].max():.1e}", f"{(err / data['sigma']).max():.1e}"

    t = timed(run_ours, reps)
    rows.append(
        [
            "derivflow `implied_volatility` (scalar loop)",
            f"{len(xs['price']):,}",
            time_cell({k: v / len(xs["price"]) for k, v in t.items()}),
            *accuracy(run_ours(), xs),
        ]
    )

    xv = {k: v[:n_vec] for k, v in x.items()}

    def run_vec():
        return implied_volatility_vectorized(
            xv["price"], xv["S"], xv["K"], xv["T"], xv["r"], xv["ot"], xv["q"]
        )

    t = timed(run_vec, reps)
    rows.append(
        [
            "derivflow `implied_volatility_vectorized`",
            f"{len(xv['price']):,}",
            time_cell({k: v / len(xv["price"]) for k, v in t.items()}),
            *accuracy(run_vec(), xv),
        ]
    )

    if pv_iv is not None:

        def run_pv():
            return [
                pv_iv(p, s, k, t, r, q, o[0])
                for p, s, k, t, r, o, q in zip(
                    xs["price"], xs["S"], xs["K"], xs["T"], xs["r"], xs["ot"], xs["q"], strict=True
                )
            ]

        t = timed(run_pv, reps)
        rows.append(
            [
                "py_vollib (Let's Be Rational, scalar loop)",
                f"{len(xs['price']):,}",
                time_cell({k: v / len(xs["price"]) for k, v in t.items()}),
                *accuracy(run_pv(), xs),
            ]
        )

    if ql is not None:
        n_ql = min(len(xs["price"]), 500 if quick else 2_000)
        xq = {k: v[:n_ql] for k, v in xs.items()}
        env = QLEuropean(365)
        opts = []
        today = env.today
        for i in range(n_ql):
            days = max(1, int(round(xq["T"][i] * 365)))
            xq["T"][i] = days / 365
            xq["price"][i] = black_scholes_price(
                xq["S"][i], xq["K"][i], xq["T"][i], 0.03, xq["sigma"][i], xq["ot"][i], 0.01
            )
            payoff = ql.PlainVanillaPayoff(
                ql.Option.Call if xq["ot"][i] == "call" else ql.Option.Put, float(xq["K"][i])
            )
            opt = ql.VanillaOption(payoff, ql.EuropeanExercise(today + days))
            opts.append(opt)

        def run_ql():
            out = []
            for i, opt in enumerate(opts):
                try:
                    out.append(
                        opt.impliedVolatility(
                            float(xq["price"][i]), env.process, 1e-12, 1000, 1e-7, 4.0
                        )
                    )
                except RuntimeError:
                    out.append(np.nan)
            return out

        t = timed(run_ql, max(1, reps // 2))
        vols = np.array(run_ql())
        failed = int(np.isnan(vols).sum())
        rows.append(
            [
                "QuantLib `impliedVolatility` (scalar loop)",
                f"{n_ql:,}",
                time_cell({k: v / n_ql for k, v in t.items()}),
                *accuracy(np.nan_to_num(vols, nan=np.inf) if failed else vols, xq),
            ]
        )
        notes.append(
            f"QuantLib's solver is given a price accuracy of 1e-12 and bounds [1e-7, 4]; "
            f"{failed} of {n_ql} quotes raised. Its maturities are rounded to whole days, so its "
            "quotes are regenerated at those maturities."
        )
    notes.append(
        "Inputs: S=100, K/S uniform in [0.7, 1.3], T uniform in [1 week, 2 years], sigma uniform "
        "in [8%, 80%], r=3%, q=1%, half calls and half puts, prices from known volatilities. "
        "Error columns: max |solved - true| over all quotes, over quotes with vega > 1e-6 S, "
        "and max relative error."
    )
    return {"rows": rows, "notes": notes}


def bench_greeks(reps: int, quick: bool) -> dict[str, Any]:
    n = 2_000 if quick else 20_000
    x = random_inputs(n, seed=3)
    calc = GreeksCalculator()
    rows, notes, acc = [], [], []
    K, T = 100.0, 1.0

    def ours():
        return [
            calc.analytical_greeks(s, K, T, 0.03, v, "call", 0.01)
            for s, v in zip(x["S"], x["sigma"], strict=True)
        ]

    t = timed(ours, reps)
    rows.append(
        ["derivflow `analytical_greeks` (10 Greeks, scalar loop)", f"{n:,}", rate_cell(t, n)]
    )
    if ql is not None:
        env = QLEuropean(365)
        opt = env.option(K)

        def ql_greeks():
            out = []
            for s, v in zip(x["S"], x["sigma"], strict=True):
                env.spot.setValue(float(s))
                env.vol.setValue(float(v))
                out.append((opt.delta(), opt.gamma(), opt.vega(), opt.theta(), opt.rho()))
            return out

        t = timed(ql_greeks, reps)
        rows.append(
            ["QuantLib `AnalyticEuropeanEngine` (5 Greeks, scalar loop)", f"{n:,}", rate_cell(t, n)]
        )
        notes.append(
            "derivflow computes 10 Greeks per call, QuantLib 5 here (delta, gamma, vega, theta, rho)."
        )
        ref = np.array(ql_greeks())
        mine = np.array(
            [(g.delta, g.gamma, g.vega * 100, g.theta * 365, g.rho * 100) for g in ours()]
        )
        for j, name in enumerate(["delta", "gamma", "vega", "theta", "rho"]):
            a, rel = err_cell(mine[:, j] - ref[:, j], ref[:, j])
            acc.append([name, f"{n:,}", a, rel])
    return {"rows": rows, "notes": notes, "accuracy": acc}


def bench_monte_carlo(reps: int, quick: bool) -> dict[str, Any]:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    truth = black_scholes_price(S, K, T, r, sigma)
    sizes = [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000]
    rows = []
    for n in sizes:
        for label, kw in (
            ("plain", {}),
            ("antithetic", {"antithetic": True}),
            ("control variate", {"control_variate": True}),
        ):
            res = MonteCarloEngine(n, random_seed=11).price(S, K, T, r, sigma, "call", **kw)
            t = timed(
                lambda n=n, kw=kw: MonteCarloEngine(n, random_seed=11).price(
                    S, K, T, r, sigma, "call", **kw
                ),
                reps,
            )
            rows.append(
                [
                    f"{n:,}",
                    label,
                    f"{res['price']:.4f}",
                    f"{res['std_error']:.4f}",
                    f"{abs(res['price'] - truth):.4f}",
                    time_cell(t, 1e3, "ms"),
                ]
            )
    return {"rows": rows, "truth": truth}


def bench_trees_heston(reps: int, quick: bool) -> dict[str, Any]:
    rows, notes = [], []
    S, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    ref = None
    if ql is not None:
        env_today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = env_today
        dc = ql.Actual365Fixed()
        h = lambda x: ql.YieldTermStructureHandle(ql.FlatForward(env_today, x, dc))  # noqa: E731
        proc = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            h(0.0),
            h(r),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(env_today, ql.NullCalendar(), sigma, dc)
            ),
        )
        opt = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, K), ql.AmericanExercise(env_today, env_today + 365)
        )
        opt.setPricingEngine(ql.FdBlackScholesVanillaEngine(proc, 4000, 4000))
        ref = opt.NPV()
        for steps in (100, 500, 2000):
            opt.setPricingEngine(ql.BinomialVanillaEngine(proc, "crr", steps))
            t = timed(opt.NPV, reps)
            rows.append(
                [
                    f"QuantLib CRR tree ({steps} steps)",
                    f"{opt.NPV():.6f}",
                    f"{abs(opt.NPV() - ref):.1e}",
                    time_cell(t, 1e3, "ms"),
                ]
            )
        notes.append(
            f"Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid ({ref:.6f}). "
            "QuantLib's 'crr' tree uses a drift-adjusted up-probability, so its tree values differ "
            "from the textbook CRR used by derivflow at the same step count."
        )
    for steps in (100, 500, 2000) if not quick else (100, 500):
        tree = BinomialTree(steps)
        val = tree.price(S, K, T, r, sigma, "put", american=True)
        t = timed(lambda tree=tree: tree.price(S, K, T, r, sigma, "put", american=True), reps)
        rows.append(
            [
                f"derivflow BinomialTree ({steps} steps)",
                f"{val:.6f}",
                f"{abs(val - ref):.1e}" if ref else "-",
                time_cell(t, 1e3, "ms"),
            ]
        )

    heston_rows = []
    params = (0.04, 2.0, 0.04, 0.5, -0.7)
    model = HestonModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.set_parameters(*params)
    strikes = np.linspace(70, 130, 13)
    t = timed(
        lambda: [model.analytic_price(100.0, k, 1.0, 0.03, "call", 0.01) for k in strikes], reps
    )
    ours = np.array([model.analytic_price(100.0, k, 1.0, 0.03, "call", 0.01) for k in strikes])
    if ql is not None:
        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        h = lambda x: ql.YieldTermStructureHandle(ql.FlatForward(today, x, dc))  # noqa: E731
        hp = ql.HestonProcess(h(0.03), h(0.01), ql.QuoteHandle(ql.SimpleQuote(100.0)), *params)
        engine = ql.AnalyticHestonEngine(ql.HestonModel(hp), 1e-12, 100_000)
        opts = []
        for k in strikes:
            o = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, float(k)), ql.EuropeanExercise(today + 365)
            )
            o.setPricingEngine(engine)
            opts.append(o)

        def ql_prices():
            for o in opts:
                o.recalculate()
            return [o.NPV() for o in opts]

        tq = timed(ql_prices, reps)
        ref_h = np.array(ql_prices())
        a, rel = err_cell(ours - ref_h, ref_h)
        heston_rows.append(
            [
                "derivflow `analytic_price` (scipy quad)",
                time_cell({k: v / len(strikes) for k, v in t.items()}, 1e3, "ms"),
                a,
                rel,
            ]
        )
        heston_rows.append(
            [
                "QuantLib `AnalyticHestonEngine` (Gauss-Laguerre, rel. tol 1e-12)",
                time_cell({k: v / len(strikes) for k, v in tq.items()}, 1e3, "ms"),
                "reference",
                "-",
            ]
        )
    else:
        heston_rows.append(
            [
                "derivflow `analytic_price` (scipy quad)",
                time_cell({k: v / len(strikes) for k, v in t.items()}, 1e3, "ms"),
                "-",
                "-",
            ]
        )
    return {"rows": rows, "notes": notes, "heston": heston_rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reps", type=int, default=7, help="timed repetitions per measurement")
    parser.add_argument("--quick", action="store_true", help="smaller problem sizes")
    parser.add_argument("--output", help="also write the Markdown report to this file")
    parser.add_argument("--json", help="also write raw results as JSON to this file")
    args = parser.parse_args()

    env = environment()
    pricing = bench_pricing(args.reps, args.quick)
    iv = bench_implied_vol(args.reps, args.quick)
    greeks = bench_greeks(args.reps, args.quick)
    mc = bench_monte_carlo(args.reps, args.quick)
    th = bench_trees_heston(args.reps, args.quick)

    cmd = (
        COMMAND
        + (" --quick" if args.quick else "")
        + (f" --reps {args.reps}" if args.reps != 7 else "")
    )
    lines = [
        "# derivflow benchmark results",
        "",
        f"Command: `{cmd}`",
        "",
        table(["item", "value"], [[k, str(v)] for k, v in env.items()]),
        "",
        f"Each timing: one warm-up run, then {args.reps} repetitions; median and [min, max].",
        "",
        "## European option pricing throughput (options per second)",
        "",
        table(["implementation", "options", "options/s median [min, max]"], pricing["rows"]),
        "",
        *[f"- {n}" for n in pricing["notes"]],
        "- Vectorised rows: random S, K in [80, 120], T in [0.05, 2], r in [0, 6%], q in [0, 3%], sigma in [10%, 60%].",
        "- Scalar rows: K=100, T=1, r=3%, q=1%, the same random S and sigma for every library.",
        "",
        "Accuracy against QuantLib (random inputs, maturities rounded to whole days):",
        "",
        table(["comparison", "options", "max abs error", "max rel error"], pricing["accuracy"])
        if pricing["accuracy"]
        else "QuantLib not installed.",
        "",
        "## Implied volatility (time per quote and error against the true volatility)",
        "",
        table(
            [
                "implementation",
                "quotes",
                "time per quote median [min, max]",
                "max abs error",
                "max abs error (vega > 1e-6 S)",
                "max rel error",
            ],
            iv["rows"],
        ),
        "",
        *[f"- {n}" for n in iv["notes"]],
        "",
        "## Greeks (options per second) and accuracy against QuantLib",
        "",
        table(["implementation", "options", "options/s median [min, max]"], greeks["rows"]),
        "",
        *[f"- {n}" for n in greeks["notes"]],
        "",
        table(
            ["Greek (QuantLib units)", "options", "max abs error", "max rel error"],
            greeks["accuracy"],
        )
        if greeks["accuracy"]
        else "",
        "",
        "## Monte Carlo: standard error against number of paths",
        "",
        f"ATM European call, S=K=100, T=1, r=5%, sigma=20%, seed 11; Black-Scholes value {mc['truth']:.4f}.",
        "",
        table(
            ["paths", "method", "price", "std error", "abs error", "time median [min, max]"],
            mc["rows"],
        ),
        "",
        "## American put (S=36, K=40, T=1, r=6%, sigma=20%) and Heston pricing",
        "",
        table(
            ["implementation", "price", "abs diff to FD reference", "time median [min, max]"],
            th["rows"],
        ),
        "",
        *[f"- {n}" for n in th["notes"]],
        "",
        "Heston analytic call prices, 13 strikes 70-130, T=1, r=3%, q=1%, v0=theta=0.04, kappa=2, sigma=0.5, rho=-0.7 (time per price):",
        "",
        table(
            ["implementation", "time per price median [min, max]", "max abs diff", "max rel diff"],
            th["heston"],
        ),
        "",
    ]
    report = "\n".join(lines)
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(
                {
                    "environment": env,
                    "pricing": pricing,
                    "implied_vol": iv,
                    "greeks": greeks,
                    "monte_carlo": mc,
                    "trees_heston": th,
                },
                f,
                indent=2,
                default=str,
            )
            f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
