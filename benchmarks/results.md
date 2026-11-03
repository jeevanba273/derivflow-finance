# derivflow benchmark results

Command: `python benchmarks/run_benchmarks.py`

| item | value |
|---|---|
| cpu | Intel(R) Core(TM) Ultra 9 185H |
| logical_cpus | 22 |
| ram_gb | 31.5 |
| os | Windows 11 (10.0.26300) |
| python | 3.13.5 |
| derivflow | 0.3.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| QuantLib | 1.44 |
| py_vollib | 1.0.12 |

Method: implementations in one table are timed in the same loop, 15 repetitions after one warm-up call each, in a rotating order. Rates and times are median [min, max]; the 'time / derivflow, median [min, max]' column is the per-repetition ratio of time per unit to derivflow's (below 1 means faster than derivflow). Everything runs single-threaded in one Python process.

Background load during this run (32.6 s, 31 one-second samples): total CPU median 19% [16%, 31%] of 22 logical CPUs, including this benchmark's own single thread. Busiest other processes: OneDrive.Sync.Service.exe 32.1 CPU-s, OneDrive.exe 30.8 CPU-s, Code.exe 6.1 CPU-s, System 5.5 CPU-s. Absolute rates may be lower than on an idle machine; the ratios are measured side by side within each repetition.

## European option pricing throughput (options per second)

| implementation | options | options/s median [min, max] | time / derivflow, median [min, max] |
|---|---|---|---|
| derivflow `black_scholes_price` (vectorised) | 1,000,000 | 8,878,385 [8,514,631, 9,285,663] | 1.00 [1.00, 1.00] |
| plain NumPy/SciPy formula, `scipy.special.ndtr` (vectorised) | 1,000,000 | 12,640,691 [11,313,907, 13,375,851] | 0.71 [0.67, 0.76] |
| textbook NumPy/SciPy formula, `scipy.stats.norm.cdf` (vectorised) | 1,000,000 | 8,461,998 [8,188,788, 8,766,747] | 1.05 [1.01, 1.12] |
| derivflow `BlackScholesAnalytical.price` (scalar loop) | 20,000 | 683,875 [638,030, 715,192] | 1.00 [1.00, 1.00] |
| py_vollib `black_scholes_merton` (scalar loop) | 20,000 | 413,726 [394,456, 445,380] | 1.66 [1.48, 1.73] |
| QuantLib `AnalyticEuropeanEngine` (scalar loop) | 20,000 | 274,897 [263,680, 289,243] | 2.46 [2.32, 2.67] |

- The `ndtr` row is the like-for-like baseline: the same normal CDF as derivflow, without derivflow's input checks, broadcasting of all six inputs, expiry handling and call/put branch. The `norm.cdf` row is the textbook form, which adds scipy.stats overhead.
- QuantLib reprices a prebuilt option after updating two quotes, so each call includes its observer notifications and the Python binding; the Python loops include interpreter overhead.
- Vectorised rows: random S, K in [80, 120], T in [0.05, 2], r in [0, 6%], q in [0, 3%], sigma in [10%, 60%].
- Scalar rows: K=100, T=1, r=3%, q=1%, the same random S and sigma for every library.

Accuracy against QuantLib (random inputs, maturities rounded to whole days; the largest relative errors are on the smallest, deep out-of-the-money prices):

| comparison | options | max abs error | max rel error |
|---|---|---|---|
| derivflow vs QuantLib | 2,000 | 5.0e-14 | 4.7e-08 |
| NumPy/SciPy formula with `ndtr` vs QuantLib | 2,000 | 5.0e-14 | 4.7e-08 |
| NumPy/SciPy formula with `norm.cdf` vs QuantLib | 2,000 | 5.0e-14 | 4.7e-08 |

## Implied volatility (time per quote and error against the true volatility)

| implementation | quotes | time per quote median [min, max] | time / derivflow, median [min, max] | max abs error | max abs error (vega > 1e-6 S) | max rel error |
|---|---|---|---|---|---|---|
| derivflow `implied_volatility` (scalar loop) | 5,000 | 12.5 [12.0, 13.3] us | 1.00 [1.00, 1.00] | 3.8e-07 | 1.8e-10 | 3.3e-06 |
| derivflow `implied_volatility_vectorized` | 99,825 | 3.9 [3.9, 4.1] us | 0.31 [0.30, 0.34] | 2.6e-06 | 2.5e-10 | 1.7e-05 |
| py_vollib (Let's Be Rational, scalar loop) | 5,000 | 18.1 [17.3, 19.1] us | 1.44 [1.34, 1.51] | 4.9e-07 | 1.8e-10 | 2.6e-06 |
| QuantLib `impliedVolatility` (scalar loop) | 2,000 | 12.6 [11.9, 16.9] us | 1.02 [0.93, 1.31] | 1.1e-02 | 1.9e-11 | 5.4e-02 |

- QuantLib's solver is given a price accuracy of 1e-12 and bounds [1e-7, 4]; 0 of 2000 quotes raised. Its maturities are rounded to whole days, so its quotes are repriced at those maturities.
- Inputs: S=100, K/S uniform in [0.7, 1.3], T uniform in [1 week, 2 years], sigma uniform in [8%, 80%], r=3%, q=1%, half calls and half puts, prices from known volatilities. Error columns: max |solved - true| over all quotes, over quotes with vega > 1e-6 S, and max relative error. The ratio compares time per quote with the scalar derivflow solver.

## Greeks (options per second) and accuracy against QuantLib

| implementation | options | options/s median [min, max] | time / derivflow, median [min, max] |
|---|---|---|---|
| derivflow `analytical_greeks` (10 Greeks, scalar loop) | 20,000 | 259,977 [247,925, 272,763] | 1.00 [1.00, 1.00] |
| QuantLib `AnalyticEuropeanEngine` (5 Greeks, scalar loop) | 20,000 | 197,959 [143,020, 204,296] | 1.31 [1.23, 1.88] |

- derivflow computes 10 Greeks per call, QuantLib 5 here (delta, gamma, vega, theta, rho).

| Greek (QuantLib units) | options | max abs error | max rel error |
|---|---|---|---|
| delta | 20,000 | 1.6e-15 | 1.3e-14 |
| gamma | 20,000 | 1.7e-16 | 1.1e-14 |
| vega | 20,000 | 1.2e-13 | 1.1e-14 |
| theta | 20,000 | 2.7e-14 | 1.1e-14 |
| rho | 20,000 | 1.6e-13 | 1.3e-14 |

## Monte Carlo: standard error against number of paths

ATM European call, S=K=100, T=1, r=5%, sigma=20%, seed 11; Black-Scholes value 10.4506. The ratio column is relative to the plain estimator with the same number of paths.

| paths | method | price | std error | abs error | time median [min, max] | time / plain |
|---|---|---|---|---|---|---|
| 1,000 | plain | 10.7168 | 0.4661 | 0.2662 | 48.2 [43.8, 156.6] us | 1.00 [1.00, 1.00] |
| 1,000 | antithetic | 10.0801 | 0.3250 | 0.3705 | 54.3 [49.8, 161.5] us | 1.11 [0.96, 2.23] |
| 1,000 | control variate | 10.4950 | 0.1770 | 0.0445 | 80.0 [75.0, 326.6] us | 1.70 [1.54, 2.27] |
| 10,000 | plain | 10.6368 | 0.1478 | 0.1862 | 156.6 [144.1, 258.7] us | 1.00 [1.00, 1.00] |
| 10,000 | antithetic | 10.4062 | 0.1050 | 0.0444 | 140.3 [125.7, 257.8] us | 0.86 [0.75, 1.68] |
| 10,000 | control variate | 10.4477 | 0.0563 | 0.0029 | 227.0 [202.6, 371.6] us | 1.40 [1.28, 2.37] |
| 100,000 | plain | 10.4534 | 0.0465 | 0.0028 | 1,315.3 [1,262.3, 2,056.9] us | 1.00 [1.00, 1.00] |
| 100,000 | antithetic | 10.4503 | 0.0328 | 0.0003 | 1,178.0 [1,000.7, 1,790.2] us | 0.82 [0.61, 1.27] |
| 100,000 | control variate | 10.4558 | 0.0178 | 0.0053 | 2,447.7 [2,115.6, 3,022.3] us | 1.73 [1.38, 2.36] |
| 1,000,000 | plain | 10.4491 | 0.0147 | 0.0015 | 31,742.7 [30,938.5, 35,467.0] us | 1.00 [1.00, 1.00] |
| 1,000,000 | antithetic | 10.4311 | 0.0104 | 0.0195 | 31,142.3 [30,181.4, 33,253.1] us | 0.98 [0.91, 1.06] |
| 1,000,000 | control variate | 10.4469 | 0.0056 | 0.0037 | 47,606.3 [45,719.5, 52,893.9] us | 1.50 [1.37, 1.54] |

## American put (S=36, K=40, T=1, r=6%, sigma=20%) and Heston pricing

| implementation | price | abs diff to FD reference | time median [min, max] | time / derivflow, median [min, max] |
|---|---|---|---|---|
| derivflow BinomialTree (100 steps) | 4.488050 | 1.5e-03 | 0.7 [0.7, 0.9] ms | 1.00 [1.00, 1.00] |
| QuantLib CRR tree (100 steps) | 4.488180 | 1.6e-03 | 0.1 [0.1, 0.1] ms | 0.09 [0.08, 0.13] |
| derivflow BinomialTree (500 steps) | 4.486375 | 1.9e-04 | 4.6 [4.4, 5.1] ms | 1.00 [1.00, 1.00] |
| QuantLib CRR tree (500 steps) | 4.486401 | 1.6e-04 | 1.1 [0.9, 1.2] ms | 0.22 [0.19, 0.28] |
| derivflow BinomialTree (2000 steps) | 4.486687 | 1.2e-04 | 28.3 [26.6, 29.8] ms | 1.00 [1.00, 1.00] |
| QuantLib CRR tree (2000 steps) | 4.486694 | 1.3e-04 | 14.7 [14.2, 17.0] ms | 0.52 [0.49, 0.59] |

- Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid (4.486563). QuantLib's 'crr' tree uses a drift-adjusted up-probability, so its tree values differ from the textbook CRR used by derivflow at the same step count.
- The ratio compares each tree with derivflow's tree at the same number of steps.

Heston analytic call prices, 13 strikes 70-130, T=1, r=3%, q=1%, v0=theta=0.04, kappa=2, sigma=0.5, rho=-0.7 (time per price):

| implementation | time per price median [min, max] | time / derivflow, median [min, max] | max abs diff | max rel diff |
|---|---|---|---|---|
| derivflow `analytic_price` (scipy quad) | 0.8 [0.7, 0.8] ms | 1.00 [1.00, 1.00] | 1.1e-12 | 4.3e-13 |
| QuantLib `AnalyticHestonEngine` (adaptive Gauss-Lobatto, rel. tol 1e-12) | 0.4 [0.4, 0.4] ms | 0.49 [0.44, 0.54] | reference | - |
