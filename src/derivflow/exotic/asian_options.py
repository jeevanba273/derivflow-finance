"""
Average-price (Asian) options under Black-Scholes dynamics, no dividend yield.

The payoff is max(A - K, 0) for a call and max(K - A, 0) for a put, where A
is the average of the underlying over the fixing dates. Discrete averaging
uses n equally spaced fixings t_i = i T / n, i = 1..n; today's spot is not a
fixing.

- Geometric average: closed form, continuous (Kemna and Vorst 1990) or
  discrete (lognormal moments of the geometric mean of the fixings).
- Arithmetic average: Monte Carlo with exact GBM sampling at the fixings and,
  by default, the discrete geometric option as a control variate (Kemna and
  Vorst 1990; Glasserman 2003, section 4.1). The control's mean is the
  closed form for the same fixing schedule, so the estimator is unbiased.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats as stats

from .._validation import check_live_inputs, check_option_type


class AsianType(Enum):
    """Asian option averaging types"""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class AverageType(Enum):
    """Averaging calculation types"""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class AsianOptionResult:
    """Asian option price. Monte Carlo results carry the standard error
    (ddof=1), a 95% confidence interval and, with the control variate, the
    fitted coefficient and the variance reduction ratio in ``convergence_info``."""

    price: float
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    num_simulations: int | None = None
    asian_type: str | None = None
    average_type: str | None = None
    pricing_method: str | None = None
    convergence_info: dict | None = None


class AsianOptions:
    """Asian option pricer; see the module docstring for the methods."""

    def __init__(self, num_sims: int = 100000, random_seed: int | None = None):
        """
        Parameters:
        -----------
        num_sims : int
            Number of simulated paths for the Monte Carlo methods.
        random_seed : int, optional
            Seed of the engine's private random number generator.
        """
        if num_sims < 2:
            raise ValueError("num_sims must be at least 2")
        self.num_sims = int(num_sims)
        # Private generator: NumPy's global random state is never touched.
        self.rng = np.random.default_rng(random_seed)

    def price_geometric_asian_analytical(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        average_type: str = "continuous",
        num_fixings: int = 252,
    ) -> AsianOptionResult:
        """
        Price a geometric-average-price Asian option in closed form.

        The geometric average G of a lognormal price is lognormal, so the price
        is a discounted lognormal expectation, e^{-rT} [F N(d1) - K N(d2)] for a
        call, with F = exp(m + s^2 / 2), d2 = (m - ln K) / s and d1 = d2 + s.

        - ``average_type='continuous'``: m = ln S + (r - sigma^2/2) T / 2 and
          s^2 = sigma^2 T / 3 (Kemna and Vorst, 1990).
        - ``average_type='discrete'``: ``num_fixings`` = n equally spaced fixings
          at t_i = i T / n, i = 1..n (today's spot is not a fixing), giving
          m = ln S + (r - sigma^2/2) T (n+1) / (2n) and
          s^2 = sigma^2 T (n+1)(2n+1) / (6 n^2). With n = 1 this is the
          Black-Scholes price.

        Parameters:
        -----------
        S, K : float
            Spot and strike.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str
            'call' or 'put'
        average_type : str
            'continuous' or 'discrete'
        num_fixings : int
            Number of fixings for ``average_type='discrete'``.

        Returns:
        --------
        AsianOptionResult
            Pricing result with analytical price
        """
        check_live_inputs(S, K, T, sigma)
        check_option_type(option_type)
        if average_type == "continuous":
            mean_time = T / 2.0
            var_g = sigma**2 * T / 3.0
        elif average_type == "discrete":
            n = int(num_fixings)
            if n < 1:
                raise ValueError("num_fixings must be at least 1")
            mean_time = T * (n + 1) / (2.0 * n)
            var_g = sigma**2 * T * (n + 1) * (2 * n + 1) / (6.0 * n**2)
        else:
            raise ValueError("average_type must be 'continuous' or 'discrete'")

        m = np.log(S) + (r - 0.5 * sigma**2) * mean_time
        s_g = np.sqrt(var_g)
        forward_g = np.exp(m + 0.5 * var_g)
        d2 = (m - np.log(K)) / s_g
        d1 = d2 + s_g
        disc = np.exp(-r * T)

        if option_type.lower() == "call":
            price = disc * (forward_g * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        elif option_type.lower() == "put":
            price = disc * (K * stats.norm.cdf(-d2) - forward_g * stats.norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return AsianOptionResult(
            price=max(float(price), 0.0),
            asian_type="geometric",
            average_type=average_type,
            pricing_method="analytical",
        )

    def _generate_asian_paths(
        self, S: float, T: float, r: float, sigma: float, num_steps: int = 252
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate GBM exactly at the fixing dates t_i = i T / n, i = 1..n.

        The averages are taken over the n fixings only; today's spot S is not
        a fixing. This is the same schedule as the discrete closed form in
        :meth:`price_geometric_asian_analytical`.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (final_prices, arithmetic_averages, geometric_averages)
        """
        dt = T / num_steps
        Z = self.rng.standard_normal((self.num_sims, num_steps))
        increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.log(S) + np.cumsum(increments, axis=1)  # columns t_1..t_n
        paths = np.exp(log_paths)

        arithmetic_avg = paths.mean(axis=1)
        geometric_avg = np.exp(log_paths.mean(axis=1))
        final_prices = paths[:, -1]

        return final_prices, arithmetic_avg, geometric_avg

    def price_arithmetic_asian_mc(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        num_steps: int = 252,
        use_control_variate: bool = True,
    ) -> AsianOptionResult:
        """
        Monte Carlo price of an arithmetic-average option with ``num_steps``
        fixings.

        With ``use_control_variate=True`` the estimator is
        mean(Y - beta (G - E[G])), where Y and G are the discounted arithmetic
        and geometric payoffs on the same paths, E[G] is the discrete
        geometric closed form and beta = cov(Y, G) / var(G) is estimated from
        the sample.

        Parameters:
        -----------
        S, K : float
            Spot and strike.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        option_type : str
            'call' or 'put'
        num_steps : int
            Number of fixings.
        use_control_variate : bool
            Use the geometric option as a control variate.

        Returns:
        --------
        AsianOptionResult
        """
        check_live_inputs(S, K, T, sigma)
        check_option_type(option_type)
        # Generate paths
        _final_prices, arithmetic_avg, geometric_avg = self._generate_asian_paths(
            S, T, r, sigma, num_steps
        )

        # Calculate arithmetic Asian payoffs
        if option_type.lower() == "call":
            arithmetic_payoffs = np.maximum(arithmetic_avg - K, 0)
        elif option_type.lower() == "put":
            arithmetic_payoffs = np.maximum(K - arithmetic_avg, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Discount payoffs
        discounted_arithmetic = arithmetic_payoffs * np.exp(-r * T)

        if use_control_variate:
            # Use geometric Asian as control variate
            if option_type.lower() == "call":
                geometric_payoffs = np.maximum(geometric_avg - K, 0)
            else:
                geometric_payoffs = np.maximum(K - geometric_avg, 0)

            discounted_geometric = geometric_payoffs * np.exp(-r * T)

            # Get analytical price for geometric Asian (control variate)
            # The control's analytical mean must be for the same fixing schedule
            # as the simulated geometric average, otherwise the estimate is biased.
            geometric_analytical = self.price_geometric_asian_analytical(
                S, K, T, r, sigma, option_type, average_type="discrete", num_fixings=num_steps
            )

            # Calculate optimal control variate coefficient
            covariance = np.cov(discounted_arithmetic, discounted_geometric)[0, 1]
            variance_control = np.var(discounted_geometric)

            if variance_control > 1e-10:  # Avoid division by zero
                beta = covariance / variance_control

                # Apply control variate
                controlled_payoffs = discounted_arithmetic - beta * (
                    discounted_geometric - geometric_analytical.price
                )

                price = np.mean(controlled_payoffs)
                std_error = np.std(controlled_payoffs, ddof=1) / np.sqrt(self.num_sims)

                # Variance reduction ratio
                original_variance = np.var(discounted_arithmetic)
                controlled_variance = np.var(controlled_payoffs)
                variance_reduction = (
                    original_variance / controlled_variance if controlled_variance > 0 else 1
                )

                convergence_info = {
                    "control_variate_used": True,
                    "beta_coefficient": beta,
                    "variance_reduction_ratio": variance_reduction,
                    "geometric_analytical_price": geometric_analytical.price,
                }
            else:
                # Fallback to standard MC
                price = np.mean(discounted_arithmetic)
                std_error = np.std(discounted_arithmetic, ddof=1) / np.sqrt(self.num_sims)
                convergence_info = {
                    "control_variate_used": False,
                    "reason": "Zero variance in control",
                }
        else:
            # Standard Monte Carlo without control variate
            price = np.mean(discounted_arithmetic)
            std_error = np.std(discounted_arithmetic, ddof=1) / np.sqrt(self.num_sims)
            convergence_info = {"control_variate_used": False}

        # Calculate confidence interval
        confidence_interval = (price - 1.96 * std_error, price + 1.96 * std_error)

        return AsianOptionResult(
            price=max(price, 0),
            std_error=std_error,
            confidence_interval=confidence_interval,
            num_simulations=self.num_sims,
            asian_type="arithmetic",
            average_type="discrete",
            pricing_method="monte_carlo",
            convergence_info=convergence_info,
        )

    def price_geometric_asian_mc(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        num_steps: int = 252,
    ) -> AsianOptionResult:
        """Monte Carlo price of a geometric-average option with ``num_steps``
        fixings (used to check the closed form)."""
        check_live_inputs(S, K, T, sigma)
        check_option_type(option_type)
        # Generate paths
        _final_prices, _arithmetic_avg, geometric_avg = self._generate_asian_paths(
            S, T, r, sigma, num_steps
        )

        # Calculate geometric Asian payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(geometric_avg - K, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(K - geometric_avg, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)

        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.num_sims)
        confidence_interval = (price - 1.96 * std_error, price + 1.96 * std_error)

        return AsianOptionResult(
            price=max(price, 0),
            std_error=std_error,
            confidence_interval=confidence_interval,
            num_simulations=self.num_sims,
            asian_type="geometric",
            average_type="discrete",
            pricing_method="monte_carlo",
        )

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        asian_type: str = "arithmetic",
        average_type: str = "discrete",
        method: str = "auto",
        num_steps: int = 252,
        **kwargs,
    ) -> AsianOptionResult:
        """
        Price an Asian option.

        Parameters:
        -----------
        S, K, T, r, sigma, option_type
            As in :meth:`price_arithmetic_asian_mc`.
        asian_type : str
            'arithmetic' or 'geometric'.
        average_type : str
            'discrete' (``num_steps`` fixings) or 'continuous' (geometric
            closed form only).
        method : str
            'auto' (closed form for geometric, Monte Carlo for arithmetic),
            'analytical' or 'monte_carlo'.
        num_steps : int
            Number of fixings for discrete averaging.
        **kwargs
            ``use_control_variate`` for the arithmetic Monte Carlo.

        Returns:
        --------
        AsianOptionResult
        """
        check_live_inputs(S, K, T, sigma)
        check_option_type(option_type)
        if average_type not in ("discrete", "continuous"):
            raise ValueError("average_type must be 'discrete' or 'continuous'")
        if asian_type == "arithmetic" and average_type == "continuous":
            raise ValueError(
                "continuous arithmetic averaging has no closed form here; use "
                "average_type='discrete' with enough fixings (num_steps)"
            )

        # Method selection logic
        if method == "auto":
            if asian_type == "geometric":
                method = "analytical"
            else:  # arithmetic
                method = "monte_carlo"

        # Route to appropriate pricing method
        if asian_type == "geometric" and method == "analytical":
            return self.price_geometric_asian_analytical(
                S, K, T, r, sigma, option_type, average_type, num_fixings=num_steps
            )
        elif asian_type == "geometric" and method == "monte_carlo":
            return self.price_geometric_asian_mc(S, K, T, r, sigma, option_type, num_steps)
        elif asian_type == "arithmetic" and method == "monte_carlo":
            use_control_variate = kwargs.get("use_control_variate", True)
            return self.price_arithmetic_asian_mc(
                S, K, T, r, sigma, option_type, num_steps, use_control_variate
            )
        else:
            raise ValueError(f"Unsupported combination: asian_type={asian_type}, method={method}")

    def compare_methods(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        asian_type: str = "arithmetic",
    ) -> dict[str, AsianOptionResult]:
        """
        Geometric: closed form vs Monte Carlo. Arithmetic: Monte Carlo with
        and without the control variate.
        """
        results = {}

        if asian_type == "geometric":
            # Compare analytical vs Monte Carlo for geometric
            try:
                results["analytical"] = self.price(
                    S, K, T, r, sigma, option_type, "geometric", method="analytical"
                )
            except Exception as e:
                results["analytical"] = f"Error: {e!s}"

            try:
                results["monte_carlo"] = self.price(
                    S, K, T, r, sigma, option_type, "geometric", method="monte_carlo"
                )
            except Exception as e:
                results["monte_carlo"] = f"Error: {e!s}"

        elif asian_type == "arithmetic":
            # Compare with and without control variate
            try:
                results["mc_with_cv"] = self.price(
                    S,
                    K,
                    T,
                    r,
                    sigma,
                    option_type,
                    "arithmetic",
                    method="monte_carlo",
                    use_control_variate=True,
                )
            except Exception as e:
                results["mc_with_cv"] = f"Error: {e!s}"

            try:
                results["mc_without_cv"] = self.price(
                    S,
                    K,
                    T,
                    r,
                    sigma,
                    option_type,
                    "arithmetic",
                    method="monte_carlo",
                    use_control_variate=False,
                )
            except Exception as e:
                results["mc_without_cv"] = f"Error: {e!s}"

        return results

    def sensitivity_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        asian_type: str = "arithmetic",
        param: str = "spot",
        bump_size: float = 0.01,
    ) -> dict[str, float]:
        """
        Finite-difference sensitivities of :meth:`price`.

        Central differences with relative bumps for spot (delta and gamma)
        and volatility (vega, per unit of volatility) and absolute bumps for
        the rate (rho, per unit of rate) and time (theta, -dV/dT per year).
        All revaluations use the same random numbers (common random numbers),
        so Monte Carlo noise largely cancels.

        Parameters:
        -----------
        param : str
            'spot', 'volatility', 'rate' or 'time'.
        bump_size : float
            Relative bump for spot and volatility, absolute for rate and time.
        """
        # Common random numbers: every revaluation below reuses the same draws,
        # so Monte Carlo noise largely cancels in the differences.
        main_rng = self.rng
        seed = int(main_rng.integers(2**63 - 1))

        def value(S_, K_, T_, r_, sigma_):
            self.rng = np.random.default_rng(seed)
            return self.price(S_, K_, T_, r_, sigma_, option_type, asian_type).price

        try:
            base_price = value(S, K, T, r, sigma)
            if param == "spot":
                up = value(S * (1 + bump_size), K, T, r, sigma)
                down = value(S * (1 - bump_size), K, T, r, sigma)
                delta = (up - down) / (2 * S * bump_size)
                gamma = (up - 2 * base_price + down) / (S * bump_size) ** 2
                return {"delta": delta, "gamma": gamma}
            if param == "volatility":
                up = value(S, K, T, r, sigma * (1 + bump_size))
                down = value(S, K, T, r, sigma * (1 - bump_size))
                return {"vega": (up - down) / (2 * sigma * bump_size)}
            if param == "rate":
                up = value(S, K, T, r + bump_size, sigma)
                down = value(S, K, T, r - bump_size, sigma)
                return {"rho": (up - down) / (2 * bump_size)}
            if param == "time":
                up = value(S, K, T + bump_size, r, sigma)
                down = value(S, K, T - bump_size, r, sigma)
                return {"theta": -(up - down) / (2 * bump_size)}
            raise ValueError(f"Unknown parameter: {param}")
        finally:
            self.rng = main_rng


# Convenience functions
def price_asian_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    asian_type: str = "arithmetic",
) -> float:
    """Price of an Asian option with :meth:`AsianOptions.price` defaults
    (252 fixings; arithmetic by Monte Carlo with a control variate, so the
    value is a random estimate)."""
    asian_engine = AsianOptions()
    result = asian_engine.price(S, K, T, r, sigma, option_type, asian_type)
    return result.price


def compare_asian_types(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
) -> dict[str, float]:
    """Arithmetic (Monte Carlo) and geometric (closed form) prices with 252
    fixings and their difference."""
    asian_engine = AsianOptions()

    arithmetic_result = asian_engine.price(S, K, T, r, sigma, option_type, "arithmetic")
    geometric_result = asian_engine.price(S, K, T, r, sigma, option_type, "geometric")

    return {
        "arithmetic_asian": arithmetic_result.price,
        "geometric_asian": geometric_result.price,
        "price_difference": arithmetic_result.price - geometric_result.price,
        "relative_difference": (arithmetic_result.price - geometric_result.price)
        / geometric_result.price,
    }
