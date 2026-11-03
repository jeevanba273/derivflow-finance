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

Each timing: one warm-up run, then 7 repetitions; median and [min, max]. Everything runs single-threaded in one Python process.

## European option pricing throughput (options per second)

| implementation | options | options/s median [min, max] |
|---|---|---|
| derivflow `black_scholes_price` (vectorised) | 1,000,000 | 9,212,777 [9,010,072, 9,268,751] |
| plain NumPy/SciPy formula (vectorised) | 1,000,000 | 7,771,536 [7,043,985, 8,362,519] |
| derivflow `BlackScholesAnalytical.price` (scalar loop) | 20,000 | 762,137 [736,730, 772,714] |
| py_vollib `black_scholes_merton` (scalar loop) | 20,000 | 416,890 [404,115, 431,337] |
| QuantLib `AnalyticEuropeanEngine` (scalar loop) | 20,000 | 292,091 [282,321, 294,569] |

- QuantLib reprices a prebuilt option after updating two quotes, so each call includes its observer notifications; the Python loops include interpreter overhead.
- Vectorised rows: random S, K in [80, 120], T in [0.05, 2], r in [0, 6%], q in [0, 3%], sigma in [10%, 60%].
- Scalar rows: K=100, T=1, r=3%, q=1%, the same random S and sigma for every library.

Accuracy against QuantLib (random inputs, maturities rounded to whole days; the largest relative errors are on the smallest, deep out-of-the-money prices):

| comparison | options | max abs error | max rel error |
|---|---|---|---|
| derivflow vs QuantLib | 2,000 | 5.0e-14 | 4.7e-08 |
| NumPy/SciPy formula vs QuantLib | 2,000 | 5.0e-14 | 4.7e-08 |

## Implied volatility (time per quote and error against the true volatility)

| implementation | quotes | time per quote median [min, max] | max abs error | max abs error (vega > 1e-6 S) | max rel error |
|---|---|---|---|---|---|
| derivflow `implied_volatility` (scalar loop) | 5,000 | 12.4 [12.2, 13.1] us | 3.8e-07 | 1.8e-10 | 3.3e-06 |
| derivflow `implied_volatility_vectorized` | 99,825 | 4.2 [4.0, 5.0] us | 2.6e-06 | 2.5e-10 | 1.7e-05 |
| py_vollib (Let's Be Rational, scalar loop) | 5,000 | 19.8 [17.8, 22.0] us | 4.9e-07 | 1.8e-10 | 2.6e-06 |
| QuantLib `impliedVolatility` (scalar loop) | 2,000 | 12.9 [12.4, 14.1] us | 1.1e-02 | 1.9e-11 | 5.4e-02 |

- QuantLib's solver is given a price accuracy of 1e-12 and bounds [1e-7, 4]; 0 of 2000 quotes raised. Its maturities are rounded to whole days, so its quotes are repriced at those maturities.
- Inputs: S=100, K/S uniform in [0.7, 1.3], T uniform in [1 week, 2 years], sigma uniform in [8%, 80%], r=3%, q=1%, half calls and half puts, prices from known volatilities. Error columns: max |solved - true| over all quotes, over quotes with vega > 1e-6 S, and max relative error.

## Greeks (options per second) and accuracy against QuantLib

| implementation | options | options/s median [min, max] |
|---|---|---|
| derivflow `analytical_greeks` (10 Greeks, scalar loop) | 20,000 | 250,426 [156,124, 275,438] |
| QuantLib `AnalyticEuropeanEngine` (5 Greeks, scalar loop) | 20,000 | 200,938 [180,758, 204,601] |

- derivflow computes 10 Greeks per call, QuantLib 5 here (delta, gamma, vega, theta, rho).

| Greek (QuantLib units) | options | max abs error | max rel error |
|---|---|---|---|
| delta | 20,000 | 1.6e-15 | 1.3e-14 |
| gamma | 20,000 | 1.7e-16 | 1.1e-14 |
| vega | 20,000 | 1.2e-13 | 1.1e-14 |
| theta | 20,000 | 2.7e-14 | 1.1e-14 |
| rho | 20,000 | 1.6e-13 | 1.3e-14 |

## Monte Carlo: standard error against number of paths

ATM European call, S=K=100, T=1, r=5%, sigma=20%, seed 11; Black-Scholes value 10.4506.

| paths | method | price | std error | abs error | time median [min, max] |
|---|---|---|---|---|---|
| 1,000 | plain | 10.7168 | 0.4661 | 0.2662 | 48.2 [44.9, 62.0] us |
| 1,000 | antithetic | 10.0801 | 0.3250 | 0.3705 | 51.3 [48.7, 62.7] us |
| 1,000 | control variate | 10.4950 | 0.1770 | 0.0445 | 82.0 [79.3, 89.3] us |
| 10,000 | plain | 10.6368 | 0.1478 | 0.1862 | 152.7 [147.6, 168.7] us |
| 10,000 | antithetic | 10.4062 | 0.1050 | 0.0444 | 125.2 [122.8, 141.7] us |
| 10,000 | control variate | 10.4477 | 0.0563 | 0.0029 | 211.1 [206.4, 224.3] us |
| 100,000 | plain | 10.4534 | 0.0465 | 0.0028 | 1,554.1 [1,407.9, 2,085.3] us |
| 100,000 | antithetic | 10.4503 | 0.0328 | 0.0003 | 1,018.1 [981.5, 1,098.4] us |
| 100,000 | control variate | 10.4558 | 0.0178 | 0.0053 | 2,292.3 [2,182.7, 2,809.6] us |
| 1,000,000 | plain | 10.4491 | 0.0147 | 0.0015 | 31,818.7 [31,008.2, 35,240.5] us |
| 1,000,000 | antithetic | 10.4311 | 0.0104 | 0.0195 | 34,683.6 [33,160.3, 37,118.0] us |
| 1,000,000 | control variate | 10.4469 | 0.0056 | 0.0037 | 49,897.3 [48,336.0, 56,407.1] us |

## American put (S=36, K=40, T=1, r=6%, sigma=20%) and Heston pricing

| implementation | price | abs diff to FD reference | time median [min, max] |
|---|---|---|---|
| QuantLib CRR tree (100 steps) | 4.488180 | 1.6e-03 | 0.1 [0.1, 0.1] ms |
| QuantLib CRR tree (500 steps) | 4.486401 | 1.6e-04 | 1.0 [1.0, 1.0] ms |
| QuantLib CRR tree (2000 steps) | 4.486694 | 1.3e-04 | 14.4 [14.3, 15.3] ms |
| derivflow BinomialTree (100 steps) | 4.488050 | 1.5e-03 | 0.7 [0.7, 0.8] ms |
| derivflow BinomialTree (500 steps) | 4.486375 | 1.9e-04 | 5.7 [4.9, 6.6] ms |
| derivflow BinomialTree (2000 steps) | 4.486687 | 1.2e-04 | 33.3 [29.8, 49.3] ms |

- Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid (4.486563). QuantLib's 'crr' tree uses a drift-adjusted up-probability, so its tree values differ from the textbook CRR used by derivflow at the same step count.

Heston analytic call prices, 13 strikes 70-130, T=1, r=3%, q=1%, v0=theta=0.04, kappa=2, sigma=0.5, rho=-0.7 (time per price):

| implementation | time per price median [min, max] | max abs diff | max rel diff |
|---|---|---|---|
| derivflow `analytic_price` (scipy quad) | 0.7 [0.7, 0.8] ms | 1.1e-12 | 4.3e-13 |
| QuantLib `AnalyticHestonEngine` (adaptive Gauss-Lobatto, rel. tol 1e-12) | 0.4 [0.4, 0.5] ms | reference | - |
