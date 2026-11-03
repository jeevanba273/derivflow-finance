"""
Risk of a book of stocks and European options on one or more underlyings.

Conventions:

- Volatilities are annualised; VaR horizons are in trading days and a horizon
  of h days has standard deviation sigma * sqrt(h / 252)
  (``TRADING_DAYS_PER_YEAR``).
- Option quantities are contracts of ``contract_multiplier`` units (100 by
  default); options are priced with Black-Scholes-Merton at the position's
  implied volatility.
- VaR and expected shortfall are positive loss amounts in currency units.

Methods: delta-normal VaR (:meth:`PortfolioRiskAnalyzer.calculate_var_parametric`),
Monte Carlo VaR by full revaluation
(:meth:`PortfolioRiskAnalyzer.calculate_var_monte_carlo`), component VaR,
Greeks per underlying, delta hedge ratios and scenario P&L. Historical VaR
on a P&L series is available in :mod:`derivflow.portfolio.risk_measures`.

Limitations: volatilities and correlations are inputs (not estimated), there
is no volatility or rate risk in the VaR, and the underlyings' returns are
jointly lognormal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from ..core.pricing_engine import PricingEngine, price_european_option
from ..greeks.calculator import GreeksCalculator

TRADING_DAYS_PER_YEAR = 252
"""Annualisation convention: volatilities are annual, horizons are trading days."""


def _check_var_inputs(confidence_level: float, time_horizon: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if time_horizon <= 0:
        raise ValueError("time_horizon must be positive (trading days)")


@dataclass
class Position:
    """One position. ``current_price`` is the underlying's price (also for
    options); ``quantity`` is shares or contracts, negative for short;
    ``expiry`` is in years; ``volatility`` is annualised (the option's implied
    volatility for options). Greeks are per unit of the option, in the units
    of :class:`~derivflow.greeks.calculator.GreeksResult`."""

    symbol: str
    position_type: str  # 'stock' or 'option'
    quantity: float
    current_price: float

    strike: float | None = None
    expiry: float | None = None
    option_type: str | None = None

    volatility: float | None = None
    risk_free_rate: float | None = None
    dividend_yield: float | None = None

    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


@dataclass
class RiskMetrics:
    """Portfolio risk report (see :meth:`PortfolioRiskAnalyzer.generate_risk_report`).

    VaR and expected shortfall are 1-day delta-normal values, positive losses.
    ``risk_contributions`` maps each underlying to its component (Euler) 95%
    VaR; the components sum to ``var_95``. ``portfolio_volatility`` is the
    1-day standard deviation of P&L in currency units. Greeks are summed over
    positions (see :meth:`PortfolioRiskAnalyzer.calculate_portfolio_greeks`).
    """

    portfolio_value: float

    # VaR metrics
    var_95: float | None = None
    var_99: float | None = None
    expected_shortfall_95: float | None = None
    expected_shortfall_99: float | None = None

    # Greeks
    portfolio_delta: float | None = None
    portfolio_gamma: float | None = None
    portfolio_theta: float | None = None
    portfolio_vega: float | None = None
    portfolio_rho: float | None = None

    # Risk decomposition
    risk_contributions: dict[str, float] | None = None
    portfolio_volatility: float | None = None


@dataclass
class ScenarioResult:
    """P&L of the book when each underlying's spot moves by a relative change."""

    scenario_name: str
    spot_changes: dict[str, float]  # Symbol -> % change
    portfolio_pnl: float
    new_portfolio_value: float
    individual_pnl: dict[str, float]  # Position -> P&L


class PortfolioRiskAnalyzer:
    """Book of stock and European option positions; see the module docstring
    for conventions and limitations."""

    def __init__(self, risk_free_rate: float = 0.05, contract_multiplier: float = 100.0):
        """
        Initialize portfolio risk analyzer

        Parameters:
        -----------
        risk_free_rate : float
            Continuously compounded risk-free rate used to price options.
        contract_multiplier : float
            Units of the underlying per option contract (default 100).
        """
        self.positions: list[Position] = []
        self.risk_free_rate = risk_free_rate
        self.contract_multiplier = float(contract_multiplier)
        self.pricing_engine = PricingEngine()
        self.greeks_calc = GreeksCalculator()
        self._corr_symbols: list[str] | None = None
        self._corr_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Underlyings, exposures and correlation
    # ------------------------------------------------------------------
    def underlyings(self) -> list[str]:
        """Distinct underlying symbols, in the order they were first added."""
        return list(dict.fromkeys(p.symbol for p in self.positions))

    def _underlying_spot(self, symbol: str) -> float:
        return next(p.current_price for p in self.positions if p.symbol == symbol)

    def _underlying_volatility(self, symbol: str) -> float:
        """Annualised return volatility of an underlying.

        Taken from a stock position in that symbol if there is one, otherwise
        from the first option position (its implied volatility is then used as
        the proxy for the underlying's volatility).
        """
        stocks = [p for p in self.positions if p.symbol == symbol and p.position_type == "stock"]
        source = stocks[0] if stocks else next(p for p in self.positions if p.symbol == symbol)
        if not source.volatility or source.volatility <= 0:
            raise ValueError(f"No positive volatility available for {symbol!r}")
        return float(source.volatility)

    def dollar_delta_by_underlying(self) -> dict[str, float]:
        """Delta exposure per underlying in currency units: sum of quantity x
        multiplier x delta x spot (stocks have delta 1 and multiplier 1)."""
        exposure = dict.fromkeys(self.underlyings(), 0.0)
        for p in self.positions:
            if p.position_type == "stock":
                exposure[p.symbol] += p.quantity * p.current_price
            elif p.position_type == "option":
                exposure[p.symbol] += (
                    p.quantity * self.contract_multiplier * (p.delta or 0.0) * p.current_price
                )
        return exposure

    def set_correlation(self, symbols: list[str], matrix: Any) -> None:
        """Set the correlation of daily returns between underlyings.

        Underlyings not listed are treated as uncorrelated with all others,
        which is also the default when no matrix is set.
        """
        C = np.asarray(matrix, dtype=float)
        n = len(symbols)
        if C.shape != (n, n):
            raise ValueError(f"correlation matrix must be {n}x{n}")
        if not np.allclose(C, C.T, atol=1e-12):
            raise ValueError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(C), 1.0, atol=1e-12) or np.any(np.abs(C) > 1 + 1e-12):
            raise ValueError("correlation matrix needs a unit diagonal and entries in [-1, 1]")
        if np.linalg.eigvalsh(C).min() < -1e-10:
            raise ValueError("correlation matrix must be positive semi-definite")
        self._corr_symbols = list(symbols)
        self._corr_matrix = C

    def _correlation(self, symbols: list[str]) -> np.ndarray:
        C = np.eye(len(symbols))
        if self._corr_matrix is not None and self._corr_symbols is not None:
            index = {s: n for n, s in enumerate(self._corr_symbols)}
            for a, sa in enumerate(symbols):
                for b, sb in enumerate(symbols):
                    if sa in index and sb in index:
                        C[a, b] = self._corr_matrix[index[sa], index[sb]]
        return C

    def _daily_covariance(self, symbols: list[str]) -> np.ndarray:
        vols = np.array([self._underlying_volatility(s) for s in symbols])
        daily = vols / np.sqrt(TRADING_DAYS_PER_YEAR)
        return np.outer(daily, daily) * self._correlation(symbols)

    def add_stock_position(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> None:
        """
        Add stock position to portfolio

        Parameters:
        -----------
        symbol : str
            Stock symbol
        quantity : float
            Number of shares (positive = long, negative = short)
        current_price : float
            Current stock price
        volatility : float
            Stock volatility
        dividend_yield : float
            Dividend yield
        """
        position = Position(
            symbol=symbol,
            position_type="stock",
            quantity=quantity,
            current_price=current_price,
            volatility=volatility,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=dividend_yield,
            delta=1.0,  # Stock delta is always 1
            gamma=0.0,  # Stock gamma is 0
            theta=0.0,  # Stock theta is 0
            vega=0.0,  # Stock vega is 0
            rho=0.0,  # Stock rho is 0
        )
        self.positions.append(position)

    def add_option_position(
        self,
        symbol: str,
        quantity: float,
        current_price: float,
        strike: float,
        expiry: float,
        option_type: str,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> None:
        """
        Add a European option position, priced and with Greeks at the
        analyzer's current ``risk_free_rate``. The rate is stored on the
        position; changing the analyzer's rate later does not reprice it.

        Parameters:
        -----------
        symbol : str
            Underlying symbol
        quantity : float
            Number of contracts (positive = long, negative = short)
        current_price : float
            Current underlying price
        strike : float
            Option strike price
        expiry : float
            Time to expiry in years
        option_type : str
            'call' or 'put'
        volatility : float
            Implied volatility
        dividend_yield : float
            Continuous dividend yield.
        """
        greeks = self.greeks_calc.calculate_greeks(
            current_price,
            strike,
            expiry,
            self.risk_free_rate,
            volatility,
            option_type,
            q=dividend_yield,
        )

        position = Position(
            symbol=symbol,
            position_type="option",
            quantity=quantity,
            current_price=current_price,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            volatility=volatility,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=dividend_yield,
            delta=greeks.delta,
            gamma=greeks.gamma,
            theta=greeks.theta,
            vega=greeks.vega,
            rho=greeks.rho,
        )
        self.positions.append(position)

    def calculate_portfolio_value(self) -> float:
        """Mark-to-model value: shares x price plus contracts x multiplier x
        Black-Scholes-Merton option price."""
        total_value = 0.0

        for position in self.positions:
            if position.position_type == "stock":
                position_value = position.quantity * position.current_price
            elif position.position_type == "option":
                # Recalculate option price
                option_price = price_european_option(
                    position.current_price,
                    position.strike,
                    position.expiry,
                    position.risk_free_rate,
                    position.volatility,
                    position.option_type,
                    q=position.dividend_yield or 0.0,
                )
                position_value = position.quantity * option_price * self.contract_multiplier
            else:
                position_value = 0.0

            total_value += position_value

        return total_value

    def calculate_greeks_by_underlying(self) -> dict[str, dict[str, float]]:
        """Greeks per underlying, in position units.

        delta and gamma are in shares of the underlying (stock delta 1, option
        Greeks times quantity times the contract multiplier); theta, vega and
        rho are in currency per day, per vol point and per rate point, as in
        :class:`~derivflow.greeks.calculator.GreeksResult`.
        """
        names = ("delta", "gamma", "theta", "vega", "rho")
        out = {sym: dict.fromkeys(names, 0.0) for sym in self.underlyings()}
        for p in self.positions:
            g = out[p.symbol]
            if p.position_type == "stock":
                g["delta"] += p.quantity
            elif p.position_type == "option":
                units = p.quantity * self.contract_multiplier
                for name in names:
                    g[name] += units * (getattr(p, name) or 0.0)
        return out

    def calculate_portfolio_greeks(self) -> dict[str, float]:
        """Greeks summed over all positions.

        theta, vega and rho add up across underlyings. delta and gamma are in
        shares of each underlying, so their sum is only meaningful for a book
        on a single underlying; use :meth:`calculate_greeks_by_underlying` or
        :meth:`dollar_delta_by_underlying` otherwise.
        """
        total = dict.fromkeys(("delta", "gamma", "theta", "vega", "rho"), 0.0)
        for greeks in self.calculate_greeks_by_underlying().values():
            for name, value in greeks.items():
                total[name] += value
        return total

    def calculate_var_parametric(
        self, confidence_level: float = 0.95, time_horizon: int = 1
    ) -> tuple[float, float]:
        """
        Delta-normal (variance-covariance) VaR and expected shortfall.

        With e the vector of dollar deltas per underlying (see
        :meth:`dollar_delta_by_underlying`) and Sigma the covariance of daily
        log-returns, built from annualised volatilities divided by sqrt(252)
        and the correlation set with :meth:`set_correlation` (identity by
        default), the P&L over h trading days is taken as normal with mean 0
        and standard deviation s = sqrt(e^T Sigma e) * sqrt(h). Then

            VaR = z_c * s,   ES = phi(z_c) / (1 - c) * s,   z_c = N^{-1}(c).

        This is the RiskMetrics (1996) linear approximation: gamma, vega and
        theta are ignored, so for option books it is a first-order estimate;
        use :meth:`calculate_var_monte_carlo` for full revaluation.

        Parameters:
        -----------
        confidence_level : float
            c in (0, 1), e.g. 0.99.
        time_horizon : int
            Horizon h in trading days.

        Returns:
        --------
        Tuple[float, float]
            (VaR, expected shortfall) as positive loss amounts.
        """
        _check_var_inputs(confidence_level, time_horizon)
        symbols = self.underlyings()
        if not symbols:
            return 0.0, 0.0
        exposure = self.dollar_delta_by_underlying()
        e = np.array([exposure[s] for s in symbols])
        variance = float(e @ self._daily_covariance(symbols) @ e)
        sigma = np.sqrt(max(variance, 0.0) * time_horizon)
        z = stats.norm.ppf(confidence_level)
        var = z * sigma
        expected_shortfall = stats.norm.pdf(z) / (1.0 - confidence_level) * sigma
        return float(var), float(expected_shortfall)

    def simulate_pnl(
        self, time_horizon: int = 1, num_sims: int = 10_000, random_seed: int | None = None
    ) -> np.ndarray:
        """
        Simulated portfolio P&L over ``time_horizon`` trading days, by full
        revaluation.

        One correlated draw per underlying (covariance as in
        :meth:`calculate_var_parametric`); log-returns are N(-s^2/2, s^2) so
        the expected price change is zero over the short horizon. Stocks are
        revalued linearly; options are repriced with Black-Scholes-Merton at
        the simulated spot, the same implied volatility and time to expiry
        reduced by h / 252 years (an option that expires within the horizon is
        valued at its intrinsic value).
        """
        from ..core.pricing_engine import black_scholes_price

        if num_sims < 2:
            raise ValueError("num_sims must be at least 2")
        symbols = self.underlyings()
        if not symbols:
            return np.zeros(num_sims)
        rng = np.random.default_rng(random_seed)
        cov = self._daily_covariance(symbols) * time_horizon
        # Symmetric square root handles singular (e.g. perfectly correlated) matrices.
        eigval, eigvec = np.linalg.eigh(cov)
        root = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
        log_ret = rng.standard_normal((num_sims, len(symbols))) @ root.T - 0.5 * np.diag(cov)
        spot_paths = {
            sym: self._underlying_spot(sym) * np.exp(log_ret[:, n]) for n, sym in enumerate(symbols)
        }
        dt = time_horizon / TRADING_DAYS_PER_YEAR
        pnl = np.zeros(num_sims)
        for p in self.positions:
            new_spot = spot_paths[p.symbol]
            if p.position_type == "stock":
                pnl += p.quantity * (new_spot - p.current_price)
            elif p.position_type == "option":
                q = p.dividend_yield or 0.0
                r = p.risk_free_rate if p.risk_free_rate is not None else self.risk_free_rate
                v0 = black_scholes_price(
                    p.current_price, p.strike, p.expiry, r, p.volatility, p.option_type, q
                )
                v1 = black_scholes_price(
                    new_spot, p.strike, max(p.expiry - dt, 0.0), r, p.volatility, p.option_type, q
                )
                pnl += p.quantity * self.contract_multiplier * (v1 - v0)
        return pnl

    def calculate_var_monte_carlo(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        num_sims: int = 10_000,
        random_seed: int | None = None,
    ) -> tuple[float, float]:
        """
        Monte Carlo VaR and expected shortfall by full revaluation.

        Simulates the P&L with :meth:`simulate_pnl` and reads VaR and ES off
        the sample with :func:`derivflow.portfolio.risk_measures.value_at_risk`
        and :func:`~derivflow.portfolio.risk_measures.expected_shortfall`.
        The estimates carry sampling error that shrinks like 1/sqrt(num_sims).

        Returns:
        --------
        Tuple[float, float]
            (VaR, expected shortfall) as positive loss amounts.
        """
        from .risk_measures import expected_shortfall, value_at_risk

        _check_var_inputs(confidence_level, time_horizon)
        pnl = self.simulate_pnl(time_horizon, num_sims, random_seed)
        if not np.any(pnl):
            return 0.0, 0.0
        return value_at_risk(pnl, confidence_level), expected_shortfall(pnl, confidence_level)

    def scenario_analysis(
        self, scenarios: dict[str, dict[str, float]], method: str = "full"
    ) -> dict[str, ScenarioResult]:
        """
        P&L for instantaneous spot moves.

        Parameters:
        -----------
        scenarios : dict
            {scenario name: {symbol: relative spot change}}, e.g.
            {"down 10%": {"SPX": -0.10}}. Unlisted symbols do not move.
        method : str
            'full' (default) reprices every option with Black-Scholes-Merton
            at the shocked spot, same volatility and expiry. 'delta_gamma'
            uses delta * dS + gamma * dS^2 / 2, which is inaccurate for large
            moves.

        Returns:
        --------
        dict
            scenario name -> :class:`ScenarioResult`.
        """
        from ..core.pricing_engine import black_scholes_price

        if method not in ("full", "delta_gamma"):
            raise ValueError("method must be 'full' or 'delta_gamma'")
        current_value = self.calculate_portfolio_value()
        results = {}
        for name, changes in scenarios.items():
            total = 0.0
            individual = {}
            for i, p in enumerate(self.positions):
                move = changes.get(p.symbol, 0.0)
                dS = p.current_price * move
                if p.position_type == "stock":
                    pnl = p.quantity * dS
                elif p.position_type == "option":
                    units = p.quantity * self.contract_multiplier
                    if method == "full":
                        q = p.dividend_yield or 0.0
                        v0 = black_scholes_price(
                            p.current_price,
                            p.strike,
                            p.expiry,
                            p.risk_free_rate,
                            p.volatility,
                            p.option_type,
                            q,
                        )
                        v1 = black_scholes_price(
                            p.current_price + dS,
                            p.strike,
                            p.expiry,
                            p.risk_free_rate,
                            p.volatility,
                            p.option_type,
                            q,
                        )
                        pnl = units * (v1 - v0)
                    else:
                        pnl = units * ((p.delta or 0.0) * dS + 0.5 * (p.gamma or 0.0) * dS**2)
                else:
                    pnl = 0.0
                individual[f"{p.position_type}_{p.symbol}_{i}"] = float(pnl)
                total += float(pnl)
            results[name] = ScenarioResult(
                scenario_name=name,
                spot_changes=changes,
                portfolio_pnl=total,
                new_portfolio_value=current_value + total,
                individual_pnl=individual,
            )
        return results

    def calculate_hedge_ratio(self, hedge_symbol: str) -> dict[str, float | str]:
        """
        Shares of ``hedge_symbol`` that make the book delta-neutral in that
        underlying (positions in other underlyings are not affected).

        Returns:
        --------
        Dict
            'hedge_symbol', 'hedge_quantity' (shares, negative = sell),
            'hedge_notional', 'current_portfolio_delta' and
            'hedged_portfolio_delta' (both in shares of ``hedge_symbol``).
        """
        by_underlying = self.calculate_greeks_by_underlying()
        if hedge_symbol not in by_underlying:
            raise ValueError(f"No position found for hedge symbol: {hedge_symbol}")
        delta = by_underlying[hedge_symbol]["delta"]
        hedge_quantity = -delta
        return {
            "hedge_symbol": hedge_symbol,
            "hedge_quantity": hedge_quantity,
            "hedge_notional": hedge_quantity * self._underlying_spot(hedge_symbol),
            "current_portfolio_delta": delta,
            "hedged_portfolio_delta": delta + hedge_quantity,
        }

    def var_contributions(
        self, confidence_level: float = 0.95, time_horizon: int = 1
    ) -> dict[str, float]:
        """Component (Euler) delta-normal VaR per underlying.

        contribution_i = z_c * sqrt(h) * e_i (Sigma e)_i / sqrt(e^T Sigma e),
        which sums to the parametric VaR (Jorion, Value at Risk, ch. 7).
        """
        _check_var_inputs(confidence_level, time_horizon)
        symbols = self.underlyings()
        exposure = self.dollar_delta_by_underlying()
        e = np.array([exposure[sym] for sym in symbols])
        cov = self._daily_covariance(symbols)
        sigma = float(np.sqrt(max(float(e @ cov @ e), 0.0)))
        if sigma == 0.0:
            return dict.fromkeys(symbols, 0.0)
        scale = stats.norm.ppf(confidence_level) * np.sqrt(time_horizon) / sigma
        return {sym: float(scale * e[n] * (cov @ e)[n]) for n, sym in enumerate(symbols)}

    def generate_risk_report(self) -> RiskMetrics:
        """Value, summed Greeks, 1-day delta-normal VaR/ES at 95% and 99%,
        component VaR per underlying and 1-day P&L volatility."""
        greeks = self.calculate_portfolio_greeks()
        var_95, es_95 = self.calculate_var_parametric(0.95)
        var_99, es_99 = self.calculate_var_parametric(0.99)
        return RiskMetrics(
            portfolio_value=self.calculate_portfolio_value(),
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            portfolio_delta=greeks["delta"],
            portfolio_gamma=greeks["gamma"],
            portfolio_theta=greeks["theta"],
            portfolio_vega=greeks["vega"],
            portfolio_rho=greeks["rho"],
            risk_contributions=self.var_contributions(0.95),
            portfolio_volatility=var_95 / stats.norm.ppf(0.95) if var_95 else 0.0,
        )


# Convenience functions
def create_sample_portfolio() -> PortfolioRiskAnalyzer:
    """Small illustrative book on two hypothetical underlyings, ASSET_A and
    ASSET_B (made-up prices and volatilities), used in examples and tests."""
    portfolio = PortfolioRiskAnalyzer(risk_free_rate=0.05)
    portfolio.add_stock_position("ASSET_A", 100, 150.0, 0.25)
    portfolio.add_stock_position("ASSET_B", -50, 300.0, 0.22)
    portfolio.add_option_position("ASSET_A", 10, 150.0, 155.0, 0.25, "call", 0.25)
    portfolio.add_option_position("ASSET_A", -5, 150.0, 145.0, 0.25, "put", 0.25)
    portfolio.add_option_position("ASSET_B", 3, 300.0, 295.0, 0.33, "call", 0.22)
    return portfolio
