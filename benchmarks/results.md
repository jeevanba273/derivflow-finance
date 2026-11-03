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
| derivflow `black_scholes_price` (vectorised) | 1,000,000 | 9,258,668 [8,300,264, 9,368,410] |
| plain NumPy/SciPy formula (vectorised) | 1,000,000 | 8,150,763 [7,400,769, 8,211,873] |
| derivflow `BlackScholesAnalytical.price` (scalar loop) | 20,000 | 721,688 [705,818, 738,825] |
| py_vollib `black_scholes_merton` (scalar loop) | 20,000 | 407,344 [361,613, 437,099] |
| QuantLib `AnalyticEuropeanEngine` (scalar loop) | 20,000 | 252,968 [232,180, 285,825] |

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
| derivflow `implied_volatility` (scalar loop) | 5,000 | 12.9 [12.5, 13.4] us | 3.8e-07 | 1.8e-10 | 3.3e-06 |
| derivflow `implied_volatility_vectorized` | 99,825 | 4.0 [3.9, 4.4] us | 2.6e-06 | 2.5e-10 | 1.7e-05 |
| py_vollib (Let's Be Rational, scalar loop) | 5,000 | 18.5 [17.6, 18.7] us | 4.9e-07 | 1.8e-10 | 2.6e-06 |
| QuantLib `impliedVolatility` (scalar loop) | 2,000 | 13.1 [12.7, 13.1] us | 1.1e-02 | 1.9e-11 | 5.4e-02 |

- QuantLib's solver is given a price accuracy of 1e-12 and bounds [1e-7, 4]; 0 of 2000 quotes raised. Its maturities are rounded to whole days, so its quotes are repriced at those maturities.
- Inputs: S=100, K/S uniform in [0.7, 1.3], T uniform in [1 week, 2 years], sigma uniform in [8%, 80%], r=3%, q=1%, half calls and half puts, prices from known volatilities. Error columns: max |solved - true| over all quotes, over quotes with vega > 1e-6 S, and max relative error.

## Greeks (options per second) and accuracy against QuantLib

| implementation | options | options/s median [min, max] |
|---|---|---|
| derivflow `analytical_greeks` (10 Greeks, scalar loop) | 20,000 | 243,358 [165,507, 260,906] |
| QuantLib `AnalyticEuropeanEngine` (5 Greeks, scalar loop) | 20,000 | 194,781 [164,002, 197,991] |

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
| 1,000 | plain | 10.7168 | 0.4661 | 0.2662 | 51.7 [45.8, 65.2] us |
| 1,000 | antithetic | 10.0801 | 0.3250 | 0.3705 | 52.8 [51.8, 77.0] us |
| 1,000 | control variate | 10.4950 | 0.1770 | 0.0445 | 103.4 [84.5, 180.5] us |
| 10,000 | plain | 10.6368 | 0.1478 | 0.1862 | 162.0 [156.3, 217.7] us |
| 10,000 | antithetic | 10.4062 | 0.1050 | 0.0444 | 125.7 [124.5, 163.8] us |
| 10,000 | control variate | 10.4477 | 0.0563 | 0.0029 | 214.7 [211.3, 287.3] us |
| 100,000 | plain | 10.4534 | 0.0465 | 0.0028 | 1,290.8 [1,226.3, 2,011.0] us |
| 100,000 | antithetic | 10.4503 | 0.0328 | 0.0003 | 1,032.1 [937.5, 1,644.5] us |
| 100,000 | control variate | 10.4558 | 0.0178 | 0.0053 | 2,516.2 [2,196.8, 3,053.3] us |
| 1,000,000 | plain | 10.4491 | 0.0147 | 0.0015 | 31,569.3 [31,268.6, 32,284.3] us |
| 1,000,000 | antithetic | 10.4311 | 0.0104 | 0.0195 | 32,925.6 [30,117.7, 34,767.7] us |
| 1,000,000 | control variate | 10.4469 | 0.0056 | 0.0037 | 49,212.9 [47,573.8, 52,700.5] us |

## American put (S=36, K=40, T=1, r=6%, sigma=20%) and Heston pricing

| implementation | price | abs diff to FD reference | time median [min, max] |
|---|---|---|---|
| QuantLib CRR tree (100 steps) | 4.488180 | 1.6e-03 | 0.1 [0.1, 0.1] ms |
| QuantLib CRR tree (500 steps) | 4.486401 | 1.6e-04 | 1.1 [1.0, 1.9] ms |
| QuantLib CRR tree (2000 steps) | 4.486694 | 1.3e-04 | 14.7 [14.6, 15.5] ms |
| derivflow BinomialTree (100 steps) | 4.488050 | 1.5e-03 | 0.7 [0.7, 0.9] ms |
| derivflow BinomialTree (500 steps) | 4.486375 | 1.9e-04 | 4.4 [4.2, 4.5] ms |
| derivflow BinomialTree (2000 steps) | 4.486687 | 1.2e-04 | 28.9 [26.2, 29.6] ms |

- Reference: QuantLib FdBlackScholesVanillaEngine on a 4000 x 4000 grid (4.486563). QuantLib's 'crr' tree uses a drift-adjusted up-probability, so its tree values differ from the textbook CRR used by derivflow at the same step count.

Heston analytic call prices, 13 strikes 70-130, T=1, r=3%, q=1%, v0=theta=0.04, kappa=2, sigma=0.5, rho=-0.7 (time per price):

| implementation | time per price median [min, max] | max abs diff | max rel diff |
|---|---|---|---|
| derivflow `analytic_price` (scipy quad) | 0.7 [0.7, 0.8] ms | 1.1e-12 | 4.3e-13 |
| QuantLib `AnalyticHestonEngine` (adaptive Gauss-Lobatto, rel. tol 1e-12) | 0.4 [0.4, 0.4] ms | reference | - |
