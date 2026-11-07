# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-08

The import package is renamed from `derivflow` to `derivflow_finance`, so that it matches the
distribution name `derivflow-finance` and cannot clash with an unrelated package called
`derivflow`. The pricing and risk code is otherwise unchanged from 0.3.1.

### Changed

- The import name is `derivflow_finance`, for example
  `from derivflow_finance import price_european_option`; module paths change accordingly
  (`derivflow_finance.core.pricing_engine` and so on).

### Deprecated

- The `derivflow` import name. In the 0.4 series `import derivflow` and
  `from derivflow.<module> import <name>` return the same objects as the new name and emit a
  `FutureWarning` once per process, and pickles that refer to `derivflow` still load. The old
  name will be removed in 0.5.0. To migrate, replace `derivflow` with `derivflow_finance` in
  import statements, or pin `derivflow-finance<0.4`.

## [0.3.1] - 2026-11-04

Documentation-only release. The library code is unchanged from 0.3.0.

### Changed

- README: the centred header with the full project name, the badge row and the built-with row
  is back, and the text names the project in full.
- The documentation and the benchmark and validation reports call the project DerivFlow
  Finance; `derivflow` remains the import name and `derivflow-finance` the package name. The
  benchmark and validation results shown were measured with 0.3.0, whose code is identical.
- Citation metadata: the title names DerivFlow Finance, and the Zenodo concept DOI
  (10.5281/zenodo.23932347) is in CITATION.cff and the README.

## [0.3.0] - 2026-11-04

This release corrects numerical errors found by checking the library against QuantLib,
py_vollib, published reference values and finite differences, and changes some defaults.
Results from 0.2.0 for the functions listed under "Fixed" should not be reused.

### Fixed

- Heston: `price_option(method="fourier")` raised `ValueError` for every input. Prices now come
  from numerical integration of the Lewis (2001) formula; they match Alan Lewis's reference
  prices to 1.1e-12 and QuantLib's `AnalyticHestonEngine` to 2.1e-10 on 27 cases.
- Heston Monte Carlo used Euler steps with a variance floor, which is biased when the Feller
  condition fails (10.06 against an analytic 9.04 for kappa=1, sigma=0.9, rho=-0.7). It now
  uses Andersen's QE scheme (default) or full-truncation Euler and reports a standard error.
- Heston calibration minimised unseeded Monte Carlo prices with finite-difference gradients,
  stopped at the starting point and reported success. It now fits analytic prices by bounded
  least squares and reports the RMSE.
- Parametric VaR applied annual volatility to a horizon in days: a 1-day 99% VaR of 100 shares
  at 100 with 20% volatility was 4,652.70 instead of 293.09. It also ignored option deltas and
  the contract multiplier, did not net positions in the same underlying and returned NaN for a
  flat book.
- Monte Carlo VaR drew an independent shock for every position, even on the same underlying;
  long and short 100 shares of one stock showed a 99% VaR of 420 instead of 0.
- `calculate_hedge_ratio` used the share delta summed across all underlyings.
- `risk_contributions` in `generate_risk_report` were position values divided by portfolio
  value; they are now component VaR per underlying and sum to the VaR.
- The no-touch probability of up barriers used the wrong reflection factor (0.398 instead of
  0.322 for S=100, H=110, T=1, r=5%, sigma=20%); barrier prices were not affected. Monte Carlo
  knock-in results reported the probability of hitting the barrier under the same name.
- `average_type="discrete"` geometric Asian options used the continuous formula (5.547 against
  QuantLib's 5.938 with 12 fixings).
- The arithmetic Asian control variate was biased because its analytical mean was for a
  different averaging schedule than the simulated one (5.786 against 5.683 +- 0.012 from plain
  Monte Carlo with 12 fixings).
- Greeks: color had the wrong sign and terms; volga was 100 times its documented unit;
  numerical theta was per year while analytical theta is per day; an expired in-the-money put
  had delta 0.
- Implied volatility stopped on a price tolerance, losing accuracy where vega is small, and
  returned arbitrary values when the time value was below double precision.
- Volatility surfaces: SVI interpolation ignored the strike; the default "cubic_spline"
  surface was a smoothed RBF that did not pass through the quotes; `check_arbitrage` tested the
  curvature of volatility instead of call prices.
- `VolatilitySurface.load_market_data` raised `TypeError` on the rows that
  `implied_vol_from_chain` and the market data client return, and loaded NaN volatilities;
  two quotes at the same strike and expiry (a call and a put) made the RBF surface singular;
  SVI expiries whose optimiser stopped early were dropped without a warning; `get_smile`
  pooled strikes from neighbouring expiries.
- `implied_vol_from_chain` rejected a caller's `on_fail` and ignored `price_field` for dict
  and DataFrame rows.
- Heston calibration treated `option_type="Call"` as a put.
- `black_scholes_price` accepted a negative volatility.
- Time to expiry was one hour off across a daylight-saving change.
- Market data: missing implied volatilities were filled with 0.25; `is_market_open` needed
  `pytz`, which was not a dependency; the 10-year yield was used as the short rate; time to
  expiry used 365 and 365.25 days in different places.
- `MonteCarloEngine`, `BarrierOptions`, `AsianOptions` and `create_sample_surface` seeded
  NumPy's global random state.

### Changed

These can change results or break existing code:

- `implied_volatility` raises `ValueError` when no volatility reproduces the price;
  `on_fail="nan"` gives the previous behaviour. `tol` is now a tolerance on volatility.
- `HestonModel.price_option` defaults to `method="analytic"`; `HestonResult` carries
  `std_error` and `confidence_interval` instead of Greek fields that were never filled.
- Asian Monte Carlo averages n fixings at t_i = i T / n and excludes today's spot.
- `VolatilitySurface` defaults to `method="rbf"`; `"cubic_spline"` is a deprecated alias.
  `check_arbitrage` takes `expiry` and `spot` and no longer returns `calendar_arbitrage`.
- `scenario_analysis` reprices options in full by default; `method="delta_gamma"` keeps the
  approximation.
- Seeded Monte Carlo results differ from 0.2.0 because every class has its own
  `numpy.random.Generator`.
- At expiry a barrier counts as touched when the spot is on or beyond it.
- Python 3.10 or later. The core dependencies are NumPy and SciPy; yfinance and pandas move to
  the `[data]` extra and plotly and matplotlib to `[plot]`. `import derivflow` no longer
  imports any of them.
- Scalar Black-Scholes prices and Greeks use the `math` module instead of `scipy.stats`, which
  removes most of the per-call overhead.
- `create_sample_portfolio` uses hypothetical underlyings ASSET_A and ASSET_B.

### Added

- `black_scholes_price`, a vectorised Black-Scholes-Merton pricer.
- Antithetic variates and a control variate in `MonteCarloEngine`; dividend yield in
  `BinomialTree`, `MonteCarloEngine` and `HestonModel`.
- `derivflow.portfolio.risk_measures`: VaR and expected shortfall of a P&L sample and closed
  forms for normal and Student-t P&L.
- `PortfolioRiskAnalyzer.set_correlation`, `dollar_delta_by_underlying`,
  `calculate_greeks_by_underlying`, `var_contributions` and `simulate_pnl`.
- Input validation with errors that name the parameter.
- Tests: 91 before, 2,127 now, including Hull, Haug and Lewis reference values, QuantLib and
  py_vollib cross-checks, every Greek against finite differences, and runs of every example,
  notebook and documentation snippet.
- `benchmarks/run_benchmarks.py` and `benchmarks/validation_report.py` with their results, a
  rewritten documentation site and README, and `CITATION.cff`.
- CI runs ruff, the tests on Python 3.10 to 3.13 (plus Windows and macOS), the oldest
  supported NumPy and SciPy, and the documentation build.

### Removed

- `demo_derivflow`, the `derivflow-demo` and `derivflow-info` console scripts, the empty
  `calibration` subpackage and the `__main__` demo blocks in library modules.
- Result fields that were never filled: the Greeks of `BarrierOptionResult` and
  `HestonResult`, and `sharpe_ratio`, `max_drawdown` and `correlation_matrix` of
  `RiskMetrics`.
- The numba kernel and extra, `setup.py` and `requirements.txt` (`pyproject.toml` is the only
  packaging file).

## [0.2.0] - 2026-06-29

### Added
- **Implied volatility solver** (`implied_volatility`) using Newton-Raphson with a
  Brent bracketing fallback, plus `implied_volatility_vectorized` and
  `implied_vol_from_chain` for option chains.
- **Dividend-yield (`q`) support** in `BlackScholesAnalytical`, `price_european_option`,
  and `GreeksCalculator` (generalized Black-Scholes-Merton). `q=0.0` is the default and
  reproduces previous results exactly.
- Top-level exports for previously-hidden classes: `AsianOptions`, `HestonModel`,
  `PortfolioRiskAnalyzer`, `DerivativesDashboard`, `AdvancedMarketData`, and the new
  implied-volatility helpers.
- A real `pytest` test suite with known-value benchmarks (Hull/Haug reference values,
  put-call parity, barrier in/out parity, analytical-vs-numerical Greeks, IV round-trip,
  Monte Carlo within standard errors).
- GitHub Actions CI (test matrix across Python 3.8-3.12) and automated PyPI publishing on
  version tags via Trusted Publishing (OIDC).
- numba-accelerated Heston Monte Carlo kernel with a pure-numpy fallback (numba is now an
  optional but used accelerator).
- Benchmark script `benchmarks/run_benchmarks.py`.
- Sphinx documentation site, Read the Docs configuration, runnable example scripts under
  `examples/`, and tutorial notebooks under `notebooks/`.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CHANGELOG.md`, and GitHub issue/PR templates.

### Fixed
- **Barrier analytical pricing was incorrect.** The down/up knock-out formulas used the
  wrong log-moneyness term and discount exponent. Rewritten with Haug's standard A/B/C/D
  building blocks and case table; e.g. the down-and-out call (S=K=100, H=90, T=1, r=5%,
  sigma=20%) now returns 8.6655 (matching the Reiner-Rubinstein reference and Monte Carlo)
  instead of the previous incorrect 3.96.
- `AdvancedMarketData.__init__` was misspelled `_init_`, so the constructor never ran and
  the market-data class raised `AttributeError` on first use.
- `derivflow.visualization` was un-importable: it imported a non-existent
  `MarketDataProvider`/`get_current_price`. Fixed to use `AdvancedMarketData`. Also fixed
  `DerivativesDashboard` reading a non-existent `vol_surface.data_points` attribute (now
  `vol_data`) and calling a non-existent `get_market_environment`.
- `PortfolioRiskAnalyzer.add_option_position` accepted `dividend_yield` but never applied
  it; it is now forwarded to pricing and Greeks.
- `__version__` was hardcoded to `"1.0.0"`, diverging from the packaged version; it is now
  sourced from the `_version.py` file written by setuptools_scm.

### Changed
- Monte Carlo path simulation is vectorized (cumulative sum of log-returns), bit-identical
  to the previous exact-GBM scheme and faster; applied to the core, Asian, and barrier
  engines.
- Removed the unimplemented "Lookback options" claim from the feature list (kept on the
  roadmap).
- Moved `jupyter` from a core dependency to a `[notebooks]` extra so a base install stays
  lean.

## [0.1.2] - 2025-07-09
### Fixed
- Build configuration fix to publish the package successfully.

## [0.1.1] - 2025-07-09
### Fixed
- Hardcoded `install_requires` to resolve a build error.

## [0.1.0] - 2025-07-09
### Added
- Initial public release: Black-Scholes / binomial / Monte Carlo pricing, first- to
  third-order Greeks, barrier and Asian exotics, Heston model, volatility surface,
  portfolio VaR, yfinance market data, and Plotly dashboards.
