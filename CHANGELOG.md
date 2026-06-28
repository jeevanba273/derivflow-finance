# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  sourced from the setuptools_scm-generated `_version.py`.

### Changed
- Monte Carlo path generation is vectorized (cumulative sum of log-returns), bit-identical
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
