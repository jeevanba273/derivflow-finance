"""
Benchmarks for DerivFlow Finance, with QuantLib, py_vollib and plain NumPy/SciPy as baselines.

Run with:

    pip install -e ".[validation]"
    python benchmarks/run_benchmarks.py                    # Markdown to stdout
    python benchmarks/run_benchmarks.py --output benchmarks/results.md
    python benchmarks/run_benchmarks.py --json benchmarks/results.json
    python benchmarks/run_benchmarks.py --quick            # smaller sizes, for a smoke test

Method. Implementations that are compared are timed in the same loop: every
repetition runs each of them once, in an order that rotates between
repetitions, after one warm-up call each. Background load therefore hits all
of them alike, and the "time / DerivFlow Finance" column is the per-repetition ratio
of their times, reported as median [min, max]. Tables also give the median
and [min, max] of the rates or times themselves. The total CPU load of the
machine during the run and the busiest other processes are recorded (needs
psutil). Everything runs single-threaded in one Python process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

import derivflow_finance
from derivflow_finance import (
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from py_vollib.black_scholes_merton import black_scholes_merton as pv_price
        from py_vollib.black_scholes_merton.implied_volatility import (
            implied_volatility as pv_iv,
        )
except ImportError:  # pragma: no cover
    pv_price = pv_iv = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

COMMAND = "python benchmarks/run_benchmarks.py"
DEFAULT_REPS = 15


# ---------------------------------------------------------------------------
# Environment and background load
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
    if psutil is not None:
        return psutil.virtual_memory().total / 2**30
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError):
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
        "derivflow-finance": derivflow_finance.__version__,
        "numpy": np.__version__,
        "scipy": _version("scipy"),
        "QuantLib": _version("QuantLib"),
        "py_vollib": _version("py_vollib"),
    }


class LoadMonitor:
    """Samples total CPU load once a second and the CPU time used by every
    other process over the run."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_times: dict[int, tuple[str, float]] = {}
        self.wall = 0.0

    @staticmethod
    def _process_times() -> dict[int, tuple[str, float]]:
        out = {}
        for p in psutil.process_iter(["name", "cpu_times"]):
            try:
                t = p.info["cpu_times"]
                if t is not None:
                    out[p.pid] = (p.info["name"] or "?", t.user + t.system)
            except (psutil.Error, AttributeError):
                continue
        return out

    def __enter__(self) -> LoadMonitor:
        self._t0 = time.perf_counter()
        if psutil is not None:
            self._start_times = self._process_times()
            psutil.cpu_percent(None)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=1.0))

    def __exit__(self, *exc: Any) -> None:
        self.wall = time.perf_counter() - self._t0
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            end = self._process_times()
            own = os.getpid()
            busy: dict[str, float] = {}
            for pid, (name, cpu) in end.items():
                if pid == own or pid not in self._start_times or name == "System Idle Process":
                    continue
                busy[name] = busy.get(name, 0.0) + max(cpu - self._start_times[pid][1], 0.0)
            self.busiest = sorted(busy.items(), key=lambda kv: -kv[1])[:4]

    def summary(self) -> dict[str, Any]:
        if psutil is None or not self.samples:
            return {"measured": False}
        return {
            "measured": True,
            "wall_seconds": round(self.wall, 1),
            "total_cpu_percent_median": statistics.median(self.samples),
            "total_cpu_percent_min": min(self.samples),
            "total_cpu_percent_max": max(self.samples),
            "samples": len(self.samples),
            "busiest_other_processes_cpu_seconds": [(n, round(s, 1)) for n, s in self.busiest],
        }


# ---------------------------------------------------------------------------
# Interleaved timing
# ---------------------------------------------------------------------------
def interleaved(fns: dict[str, Callable[[], Any]], reps: int) -> dict[str, list[float]]:
    """Time each function once per repetition, in a rotating order, after a
    warm-up call of each. Returns the per-repetition times by label."""
    labels = list(fns)
    for fn in fns.values():
        fn()
    samples: dict[str, list[float]] = {k: [] for k in labels}
    for rep in range(reps):
        shift = rep % len(labels)
        for label in labels[shift:] + labels[:shift]:
            t0 = time.perf_counter()
            fns[label]()
            samples[label].append(time.perf_counter() - t0)
    return samples


def _med_range(values: list[float], fmt: str) -> str:
    return f"{statistics.median(values):{fmt}} [{min(values):{fmt}}, {max(values):{fmt}}]"


def rate_cell(times: list[float], n: int) -> str:
    return _med_range([n / t for t in times], ",.0f")


def time_cell(times: list[float], n: int = 1, scale: float = 1e6, unit: str = "us") -> str:
    return _med_range([t / n * scale for t in times], ",.1f") + f" {unit}"


def ratio_cell(times: list[float], ref: list[float], n: int = 1, n_ref: int = 1) -> str:
    """Per-repetition (time per unit) / (DerivFlow Finance time per unit)."""
    ratios = [(t / n) / (tr / n_ref) for t, tr in zip(times, ref, strict=True)]
    return _med_range(ratios, ".2f")


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
# Inputs and baselines
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


def numpy_scipy_bs_ndtr(S, K, T, r, sigma, q):
    """Plain NumPy/SciPy Black-Scholes-Merton call with scipy.special.ndtr, the same
    normal CDF DerivFlow Finance uses (the like-for-like baseline)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def numpy_scipy_bs_textbook(S, K, T, r, sigma, q):
    """The same formula in its common textbook form, with scipy.stats.norm.cdf."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _ql_today():
    today = ql.Date(15, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    return today


def _ql_bsm(S, r, q, sigma, quotes=False):
    """QuantLib Black-Scholes-Merton process; with quotes=True spot and vol are
    SimpleQuotes so an option can be repriced by updating them."""
    today = _ql_today()
    dc = ql.Actual365Fixed()
    spot, vol = ql.SimpleQuote(S), ql.SimpleQuote(sigma)
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc))
    vts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(vol), dc)
    )
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), qts, rts, vts)
    return (process, spot, vol) if quotes else process


def _ql_payoff(option_type, K):
    return ql.PlainVanillaPayoff(ql.Option.Call if option_type == "call" else ql.Option.Put, K)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_pricing(reps: int, quick: bool) -> dict[str, Any]:
    n_vec = 100_000 if quick else 1_000_000
    n_scalar = 2_000 if quick else 20_000
    x = random_inputs(n_vec)
    args = (x["S"], x["K"], x["T"], x["r"], x["sigma"], x["q"])

    vec = interleaved(
        {
            "derivflow": lambda: black_scholes_price(*args[:5], "call", args[5]),
            "ndtr": lambda: numpy_scipy_bs_ndtr(*args),
            "textbook": lambda: numpy_scipy_bs_textbook(*args),
        },
        reps,
    )
    names = {
        "derivflow": "DerivFlow Finance `black_scholes_price` (vectorised)",
        "ndtr": "plain NumPy/SciPy formula, `scipy.special.ndtr` (vectorised)",
        "textbook": "textbook NumPy/SciPy formula, `scipy.stats.norm.cdf` (vectorised)",
    }
    rows = [
        [names[k], f"{n_vec:,}", rate_cell(v, n_vec), ratio_cell(v, vec["derivflow"])]
        for k, v in vec.items()
    ]

    # Scalar: fixed strike and maturity, spot and vol vary (QuantLib reprices via quotes).
    S_s, v_s = x["S"][:n_scalar], x["sigma"][:n_scalar]
    K, T = 100.0, 1.0
    bs = BlackScholesAnalytical()
    fns = {
        "derivflow": lambda: [
            bs.price(s, K, T, 0.03, v, "call", 0.01) for s, v in zip(S_s, v_s, strict=True)
        ]
    }
    names2 = {"derivflow": "DerivFlow Finance `BlackScholesAnalytical.price` (scalar loop)"}
    if pv_price is not None:
        fns["py_vollib"] = lambda: [
            pv_price("c", s, K, T, 0.03, v, 0.01) for s, v in zip(S_s, v_s, strict=True)
        ]
        names2["py_vollib"] = "py_vollib `black_scholes_merton` (scalar loop)"
    if ql is not None:
        process, spot, vol = _ql_bsm(100.0, 0.03, 0.01, 0.2, quotes=True)
        opt = ql.VanillaOption(_ql_payoff("call", K), ql.EuropeanExercise(_ql_today() + 365))
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        def ql_loop():
            for s, v in zip(S_s, v_s, strict=True):
                spot.setValue(float(s))
                vol.setValue(float(v))
                opt.NPV()

        fns["QuantLib"] = ql_loop
        names2["QuantLib"] = "QuantLib `AnalyticEuropeanEngine` (scalar loop)"
    sc = interleaved(fns, reps)
    rows += [
        [names2[k], f"{n_scalar:,}", rate_cell(v, n_scalar), ratio_cell(v, sc["derivflow"])]
        for k, v in sc.items()
    ]

    acc_rows = []
    if ql is not None:
        n_acc = 2_000
        y = random_inputs(n_acc, seed=1)
        days = np.maximum(1, np.round(y["T"] * 365)).astype(int)
        ours, ref, ref_ndtr, ref_textbook = [], [], [], []
        today = _ql_today()
        for i in range(n_acc):
            Ti = days[i] / 365
            process = _ql_bsm(
                float(y["S"][i]), float(y["r"][i]), float(y["q"][i]), float(y["sigma"][i])
            )
            opt = ql.VanillaOption(
                _ql_payoff("call", float(y["K"][i])), ql.EuropeanExercise(today + int(days[i]))
            )
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            ref.append(opt.NPV())
            a = (y["S"][i], y["K"][i], Ti, y["r"][i], y["sigma"][i], y["q"][i])
            ours.append(black_scholes_price(*a[:5], "call", a[5]))
            ref_ndtr.append(numpy_scipy_bs_ndtr(*a))
            ref_textbook.append(numpy_scipy_bs_textbook(*a))
        ref_a = np.array(ref)
        for label, vals in (
            ("DerivFlow Finance vs QuantLib", ours),
            ("NumPy/SciPy formula with `ndtr` vs QuantLib", ref_ndtr),
            ("NumPy/SciPy formula with `norm.cdf` vs QuantLib", ref_textbook),
        ):
            a, rel = err_cell(np.array(vals) - ref_a, ref_a)
            acc_rows.append([label, f"{n_acc:,}", a, rel])

    notes = [
        "The `ndtr` row is the like-for-like baseline: the same normal CDF as DerivFlow "
        "Finance, without its input checks, broadcasting of all six inputs, expiry handling "
        "and call/put branch. The `norm.cdf` row is the textbook form, which adds scipy.stats "
        "overhead.",
        "QuantLib reprices a prebuilt option after updating two quotes, so each call includes its "
        "observer notifications and the Python binding; the Python loops include interpreter "
        "overhead.",
        "Vectorised rows: random S, K in [80, 120], T in [0.05, 2], r in [0, 6%], q in [0, 3%], "
        "sigma in [10%, 60%].",
        "Scalar rows: K=100, T=1, r=3%, q=1%, the same random S and sigma for every library.",
    ]
    return {
        "rows": rows,
        "notes": notes,
        "accuracy": acc_rows,
        "samples": {"vectorised": vec, "scalar": sc},
    }


def iv_inputs(n: int, seed: int = 2) -> dict[str, np.ndarray]:
    """Quotes priced from known volatilities, moneyness 0.7-1.3, T 1 week to 2 years.
    Quotes whose time value is below 1e-12 of the price are dropped (no identifiable
    volatility; see derivflow_finance.volatility.implied_vol)."""
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
    xv = {k: v[:n_vec] for k, v in x.items()}
    notes = []

    def accuracy(vols, data):
        err = np.abs(np.asarray(vols, dtype=float) - data["sigma"])
        good = data["vega"] > 1e-6 * data["S"]
        return f"{err.max():.1e}", f"{err[good].max():.1e}", f"{(err / data['sigma']).max():.1e}"

    def scalar_args(d):
        return zip(d["price"], d["S"], d["K"], d["T"], d["r"], d["ot"], d["q"], strict=True)

    fns = {
        "derivflow": lambda: [
            implied_volatility(p, s, k, t, r, o, q) for p, s, k, t, r, o, q in scalar_args(xs)
        ],
        "derivflow_vec": lambda: implied_volatility_vectorized(
            xv["price"], xv["S"], xv["K"], xv["T"], xv["r"], xv["ot"], xv["q"]
        ),
    }
    sizes = {"derivflow": len(xs["price"]), "derivflow_vec": len(xv["price"])}
    names = {
        "derivflow": "DerivFlow Finance `implied_volatility` (scalar loop)",
        "derivflow_vec": "DerivFlow Finance `implied_volatility_vectorized`",
    }
    data = {"derivflow": xs, "derivflow_vec": xv}
    if pv_iv is not None:
        fns["py_vollib"] = lambda: [
            pv_iv(p, s, k, t, r, q, o[0]) for p, s, k, t, r, o, q in scalar_args(xs)
        ]
        sizes["py_vollib"] = len(xs["price"])
        names["py_vollib"] = "py_vollib (Let's Be Rational, scalar loop)"
        data["py_vollib"] = xs
    if ql is not None:
        n_ql = min(len(xs["price"]), 500 if quick else 2_000)
        xq = {k: v[:n_ql].copy() for k, v in xs.items()}
        process = _ql_bsm(100.0, 0.03, 0.01, 0.2)
        today = _ql_today()
        opts = []
        for i in range(n_ql):
            days = max(1, round(xq["T"][i] * 365))
            xq["T"][i] = days / 365
            xq["price"][i] = black_scholes_price(
                xq["S"][i], xq["K"][i], xq["T"][i], 0.03, xq["sigma"][i], xq["ot"][i], 0.01
            )
            opt = ql.VanillaOption(
                _ql_payoff(xq["ot"][i], float(xq["K"][i])), ql.EuropeanExercise(today + days)
            )
            opts.append(opt)

        def run_ql():
            out = []
            for i, opt in enumerate(opts):
                try:
                    out.append(
                        opt.impliedVolatility(
                            float(xq["price"][i]), process, 1e-12, 1000, 1e-7, 4.0
                        )
                    )
                except RuntimeError:
                    out.append(np.nan)
            return out

        fns["QuantLib"] = run_ql
        sizes["QuantLib"] = n_ql
        names["QuantLib"] = "QuantLib `impliedVolatility` (scalar loop)"
        data["QuantLib"] = xq
        failed = int(np.isnan(np.array(run_ql())).sum())
        notes.append(
            "QuantLib's solver is given a price accuracy of 1e-12 and bounds [1e-7, 4]; "
            f"{failed} of {n_ql} quotes raised. Its maturities are rounded to whole days, "
            "so its quotes are repriced at those maturities."
        )
    samples = interleaved(fns, reps)
    rows = []
    for k, v in samples.items():
        vols = np.asarray(fns[k](), dtype=float)
        rows.append(
            [
                names[k],
                f"{sizes[k]:,}",
                time_cell(v, sizes[k]),
                ratio_cell(v, samples["derivflow"], sizes[k], sizes["derivflow"]),
                *accuracy(np.nan_to_num(vols, nan=np.inf), data[k]),
            ]
        )
    notes.append(
        "Inputs: S=100, K/S uniform in [0.7, 1.3], T uniform in [1 week, 2 years], sigma uniform "
        "in [8%, 80%], r=3%, q=1%, half calls and half puts, prices from known volatilities. "
        "Error columns: max |solved - true| over all quotes, over quotes with vega > 1e-6 S, and "
        "max relative error. The ratio compares time per quote with the scalar DerivFlow "
        "Finance solver."
    )
    return {"rows": rows, "notes": notes, "samples": samples}


def bench_greeks(reps: int, quick: bool) -> dict[str, Any]:
    n = 2_000 if quick else 20_000
    x = random_inputs(n, seed=3)
    calc = GreeksCalculator()
    K, T = 100.0, 1.0
    fns = {
        "derivflow": lambda: [
            calc.analytical_greeks(s, K, T, 0.03, v, "call", 0.01)
            for s, v in zip(x["S"], x["sigma"], strict=True)
        ]
    }
    names = {"derivflow": "DerivFlow Finance `analytical_greeks` (10 Greeks, scalar loop)"}
    acc, notes = [], []
    if ql is not None:
        process, spot, vol = _ql_bsm(100.0, 0.03, 0.01, 0.2, quotes=True)
        opt = ql.VanillaOption(_ql_payoff("call", K), ql.EuropeanExercise(_ql_today() + 365))
        opt.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        def ql_greeks():
            out = []
            for s, v in zip(x["S"], x["sigma"], strict=True):
                spot.setValue(float(s))
                vol.setValue(float(v))
                out.append((opt.delta(), opt.gamma(), opt.vega(), opt.theta(), opt.rho()))
            return out

        fns["QuantLib"] = ql_greeks
        names["QuantLib"] = "QuantLib `AnalyticEuropeanEngine` (5 Greeks, scalar loop)"
        notes.append(
            "DerivFlow Finance computes 10 Greeks per call, QuantLib 5 here "
            "(delta, gamma, vega, theta, rho)."
        )
        ref = np.array(ql_greeks())
        mine = np.array(
            [
                (g.delta, g.gamma, g.vega * 100, g.theta * 365, g.rho * 100)
                for g in fns["derivflow"]()
            ]
        )
        for j, name in enumerate(["delta", "gamma", "vega", "theta", "rho"]):
            a, rel = err_cell(mine[:, j] - ref[:, j], ref[:, j])
            acc.append([name, f"{n:,}", a, rel])
    samples = interleaved(fns, reps)
    rows = [
        [names[k], f"{n:,}", rate_cell(v, n), ratio_cell(v, samples["derivflow"])]
        for k, v in samples.items()
    ]
    return {"rows": rows, "notes": notes, "accuracy": acc, "samples": samples}


def bench_monte_carlo(reps: int, quick: bool) -> dict[str, Any]:
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    truth = black_scholes_price(S, K, T, r, sigma)
    sizes = [1_000, 10_000, 100_000] if quick else [1_000, 10_000, 100_000, 1_000_000]
    methods = {
        "plain": {},
        "antithetic": {"antithetic": True},
        "control variate": {"control_variate": True},
    }
    rows, all_samples = [], {}
    for n in sizes:
        fns = {
            label: (
                lambda n=n, kw=kw: MonteCarloEngine(n, random_seed=11).price(
                    S, K, T, r, sigma, "call", **kw
                )
            )
            for label, kw in methods.items()
        }
        samples = interleaved(fns, reps)
        all_samples[n] = samples
        for label, kw in methods.items():
            res = MonteCarloEngine(n, random_seed=11).price(S, K, T, r, sigma, "call", **kw)
            rows.append(
                [
                    f"{n:,}",
                    label,
                    f"{res['price']:.4f}",
                    f"{res['std_error']:.4f}",
                    f"{abs(res['price'] - truth):.4f}",
                    time_cell(samples[label]),
                    ratio_cell(samples[label], samples["plain"]),
                ]
            )
    return {"rows": rows, "truth": truth, "samples": all_samples}


def bench_trees_heston(reps: int, quick: bool) -> dict[str, Any]:
    rows, notes = [], []
    S, K, T, r, sigma = 36.0, 40.0, 1.0, 0.06, 0.2
    ref = None
    tree_samples = {}
    if ql is not None:
        process = _ql_bsm(S, r, 0.0, sigma)
        today = _ql_today()
        fd = ql.VanillaOption(_ql_payoff("put", K), ql.AmericanExercise(today, today + 365))
        fd.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 4000, 4000))
        ref = fd.NPV()
        notes.append(
            f"Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid ({ref:.6f}). "
            "QuantLib's 'crr' tree uses a drift-adjusted up-probability, so its tree values differ "
            "from the textbook CRR used by DerivFlow Finance at the same step count."
        )
    for steps in (100, 500, 2000) if not quick else (100, 500):
        tree = BinomialTree(steps)
        fns = {"derivflow": lambda tree=tree: tree.price(S, K, T, r, sigma, "put", american=True)}
        if ql is not None:
            opt = ql.VanillaOption(_ql_payoff("put", K), ql.AmericanExercise(today, today + 365))
            opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))

            def ql_tree(opt=opt):
                opt.recalculate()  # QuantLib caches results; force a fresh valuation
                return opt.NPV()

            fns["QuantLib"] = ql_tree
        samples = interleaved(fns, reps)
        tree_samples[steps] = samples
        for k, v in samples.items():
            val = fns[k]()
            label = (
                f"DerivFlow Finance BinomialTree ({steps} steps)"
                if k == "derivflow"
                else f"QuantLib CRR tree ({steps} steps)"
            )
            rows.append(
                [
                    label,
                    f"{val:.6f}",
                    f"{abs(val - ref):.1e}" if ref else "-",
                    time_cell(v, 1, 1e3, "ms"),
                    ratio_cell(v, samples["derivflow"]),
                ]
            )

    heston_rows = []
    params = (0.04, 2.0, 0.04, 0.5, -0.7)
    model = HestonModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.set_parameters(*params)
    strikes = np.linspace(70, 130, 13)
    fns = {
        "derivflow": lambda: [
            model.analytic_price(100.0, k, 1.0, 0.03, "call", 0.01) for k in strikes
        ]
    }
    if ql is not None:
        today = _ql_today()
        dc = ql.Actual365Fixed()
        h = lambda x: ql.YieldTermStructureHandle(ql.FlatForward(today, x, dc))  # noqa: E731
        hp = ql.HestonProcess(h(0.03), h(0.01), ql.QuoteHandle(ql.SimpleQuote(100.0)), *params)
        engine = ql.AnalyticHestonEngine(ql.HestonModel(hp), 1e-12, 100_000)
        opts = []
        for k in strikes:
            o = ql.VanillaOption(_ql_payoff("call", float(k)), ql.EuropeanExercise(today + 365))
            o.setPricingEngine(engine)
            opts.append(o)

        def ql_prices():
            for o in opts:
                o.recalculate()
            return [o.NPV() for o in opts]

        fns["QuantLib"] = ql_prices
    samples = interleaved(fns, reps)
    ours = np.array(fns["derivflow"]())
    n = len(strikes)
    if ql is not None:
        ref_h = np.array(ql_prices())
        a, rel = err_cell(ours - ref_h, ref_h)
    else:
        a, rel = "-", "-"
    heston_rows.append(
        [
            "DerivFlow Finance `analytic_price` (scipy quad)",
            time_cell(samples["derivflow"], n, 1e3, "ms"),
            "1.00 [1.00, 1.00]",
            a,
            rel,
        ]
    )
    if ql is not None:
        heston_rows.append(
            [
                "QuantLib `AnalyticHestonEngine` (adaptive Gauss-Lobatto, rel. tol 1e-12)",
                time_cell(samples["QuantLib"], n, 1e3, "ms"),
                ratio_cell(samples["QuantLib"], samples["derivflow"]),
                "reference",
                "-",
            ]
        )
    return {
        "rows": rows,
        "notes": notes,
        "heston": heston_rows,
        "samples": {"trees": tree_samples, "heston": samples},
    }


def load_line(load: dict[str, Any]) -> str:
    if not load.get("measured"):
        return "Background load: not measured (install psutil)."
    busiest = ", ".join(
        f"{name} {sec} CPU-s" for name, sec in load["busiest_other_processes_cpu_seconds"]
    )
    return (
        f"Background load during this run ({load['wall_seconds']} s, {load['samples']} one-second "
        f"samples): total CPU median {load['total_cpu_percent_median']:.0f}% "
        f"[{load['total_cpu_percent_min']:.0f}%, {load['total_cpu_percent_max']:.0f}%] of "
        f"{os.cpu_count()} logical CPUs, including this benchmark's own single thread. "
        f"Busiest other processes: {busiest}. Absolute rates may be lower than on an idle "
        "machine; the ratios are "
        "measured side by side within each repetition."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="interleaved repetitions")
    parser.add_argument("--quick", action="store_true", help="smaller problem sizes")
    parser.add_argument("--output", help="also write the Markdown report to this file")
    parser.add_argument("--json", help="also write raw results as JSON to this file")
    args = parser.parse_args()

    env = environment()
    with LoadMonitor() as monitor:
        pricing = bench_pricing(args.reps, args.quick)
        iv = bench_implied_vol(args.reps, args.quick)
        greeks = bench_greeks(args.reps, args.quick)
        mc = bench_monte_carlo(args.reps, args.quick)
        th = bench_trees_heston(args.reps, args.quick)
    load = monitor.summary()

    cmd = (
        COMMAND
        + (" --quick" if args.quick else "")
        + (f" --reps {args.reps}" if args.reps != DEFAULT_REPS else "")
    )
    ratio = "time / DerivFlow Finance, median [min, max]"
    lines = [
        "# DerivFlow Finance benchmark results",
        "",
        f"Command: `{cmd}`",
        "",
        table(["item", "value"], [[k, str(v)] for k, v in env.items()]),
        "",
        f"Method: implementations in one table are timed in the same loop, {args.reps} repetitions "
        "after one warm-up call each, in a rotating order. Rates and times are median [min, max]; "
        f"the '{ratio}' column is the per-repetition ratio of time per unit to DerivFlow "
        "Finance's (below 1 means faster than DerivFlow Finance). Everything runs "
        "single-threaded in one Python process.",
        "",
        load_line(load),
        "",
        "## European option pricing throughput (options per second)",
        "",
        table(["implementation", "options", "options/s median [min, max]", ratio], pricing["rows"]),
        "",
        *[f"- {n}" for n in pricing["notes"]],
        "",
        "Accuracy against QuantLib (random inputs, maturities rounded to whole days; the largest "
        "relative errors are on the smallest, deep out-of-the-money prices):",
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
                ratio,
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
        table(["implementation", "options", "options/s median [min, max]", ratio], greeks["rows"]),
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
        "ATM European call, S=K=100, T=1, r=5%, sigma=20%, seed 11; "
        f"Black-Scholes value {mc['truth']:.4f}. The ratio column is relative to the plain "
        "estimator with the same number of paths.",
        "",
        table(
            [
                "paths",
                "method",
                "price",
                "std error",
                "abs error",
                "time median [min, max]",
                "time / plain",
            ],
            mc["rows"],
        ),
        "",
        "## American put (S=36, K=40, T=1, r=6%, sigma=20%) and Heston pricing",
        "",
        table(
            [
                "implementation",
                "price",
                "abs diff to FD reference",
                "time median [min, max]",
                ratio,
            ],
            th["rows"],
        ),
        "",
        *[f"- {n}" for n in th["notes"]],
        "- The ratio compares each tree with DerivFlow Finance's tree at the same number of steps.",
        "",
        "Heston analytic call prices, 13 strikes 70-130, T=1, r=3%, q=1%, v0=theta=0.04, kappa=2, "
        "sigma=0.5, rho=-0.7 (time per price):",
        "",
        table(
            [
                "implementation",
                "time per price median [min, max]",
                ratio,
                "max abs diff",
                "max rel diff",
            ],
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
                    "command": cmd,
                    "environment": env,
                    "background_load": load,
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
