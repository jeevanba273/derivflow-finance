"""
European and American option pricing under Black-Scholes-Merton dynamics.

The underlying follows dS = (r - q) S dt + sigma S dW under the risk-neutral
measure. Conventions used throughout the package:

- T is the time to expiry in years;
- r (risk-free rate) and q (dividend yield) are continuously compounded
  annual rates;
- sigma is the annualised volatility of log returns.

Pricers: :func:`black_scholes_price` (vectorised closed form),
:class:`BlackScholesAnalytical` (scalar closed form), :class:`BinomialTree`
(Cox-Ross-Rubinstein, European and American exercise) and
:class:`MonteCarloEngine` (exact terminal sampling with optional variance
reduction). :class:`PricingEngine` dispatches to them by name.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from .._validation import check_live_inputs, check_market_inputs, check_option_type, norm_cdf


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: str = "call",
    q: ArrayLike = 0.0,
) -> np.ndarray | float:
    """
    Black-Scholes-Merton price of European options, vectorised over NumPy arrays.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2),  P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T)),  d2 = d1 - sigma sqrt(T).

    All inputs broadcast against each other. T is in years, r and q are
    continuously compounded annual rates, sigma is annualised. Where T <= 0 the
    intrinsic value is returned; where sigma == 0 the discounted forward
    intrinsic value. S and K must be positive and sigma non-negative, or
    ValueError is raised. Returns a float for scalar input, otherwise an array.
    Reference: Hull, Options, Futures, and Other Derivatives, ch. 15 and 17.
    """
    S_, K_, T_, r_, sig_, q_ = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    )
    ot = option_type.lower()
    if ot not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if not np.all(S_ > 0):
        raise ValueError("S must be positive (spot price)")
    if not np.all(K_ > 0):
        raise ValueError("K must be positive (strike)")
    if not np.all(sig_ >= 0):
        raise ValueError("sigma must be non-negative (annualised volatility)")
    sign = 1.0 if ot == "call" else -1.0
    live = (T_ > 0) & (sig_ > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_t = np.sqrt(np.where(T_ > 0, T_, 1.0))
        vol_t = np.where(live, sig_ * sqrt_t, 1.0)
        d1 = (np.log(S_ / K_) + (r_ - q_ + 0.5 * sig_**2) * T_) / vol_t
        d2 = d1 - vol_t
        disc_s = S_ * np.exp(-q_ * np.maximum(T_, 0.0))
        disc_k = K_ * np.exp(-r_ * np.maximum(T_, 0.0))
        price = sign * (disc_s * ndtr(sign * d1) - disc_k * ndtr(sign * d2))
    intrinsic = np.where(
        T_ > 0, np.maximum(sign * (disc_s - disc_k), 0.0), np.maximum(sign * (S_ - K_), 0.0)
    )
    out = np.where(live, np.maximum(price, 0.0), intrinsic)
    return float(out) if out.ndim == 0 else out


def _validate_market_inputs(S: float, K: float, T: float, sigma: float) -> None:
    """Raise ValueError for inputs outside the domain of the pricing formulas."""
    check_live_inputs(S, K, T, sigma)


class PricingMethod(ABC):
    """Interface of the pricers used by :class:`PricingEngine`."""

    @abstractmethod
    def price(self, *args: Any, **kwargs: Any) -> Any:
        """Price an instrument."""


class BlackScholesAnalytical(PricingMethod):
    """
    Black-Scholes-Merton closed form for one European option.

    C = S e^{-qT} N(d1) - K e^{-rT} N(d2),
    P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1),
    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T)), d2 = d1 - sigma sqrt(T)

    (Black and Scholes 1973; Merton 1973 for the dividend yield). See
    :func:`black_scholes_price` for an array version.
    """

    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))."""
        return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """d2 = d1 - sigma sqrt(T)."""
        return BlackScholesAnalytical._d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        """
        Price a European option.

        Parameters:
        -----------
        S, K : float
            Spot and strike (positive).
        T : float
            Time to expiry in years; T <= 0 returns the intrinsic value.
        r, q : float
            Continuously compounded risk-free rate and dividend yield.
        sigma : float
            Annualised volatility (positive).
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float
            Option price
        """
        check_market_inputs(S, K, T, sigma)
        check_option_type(option_type)
        if T <= 0:
            if option_type.lower() == "call":
                return float(max(S - K, 0.0))
            return float(max(K - S, 0.0))

        d1 = self._d1(S, K, T, r, sigma, q)
        d2 = d1 - sigma * math.sqrt(T)
        discount_S = S * math.exp(-q * T)
        discount_K = K * math.exp(-r * T)

        if option_type.lower() == "call":
            price = discount_S * norm_cdf(d1) - discount_K * norm_cdf(d2)
        else:
            price = discount_K * norm_cdf(-d2) - discount_S * norm_cdf(-d1)

        return float(max(price, 0.0))


class BinomialTree(PricingMethod):
    """
    Cox-Ross-Rubinstein (1979) binomial tree for European and American options.

    u = exp(sigma sqrt(dt)), d = 1 / u, p = (exp((r - q) dt) - d) / (u - d),
    dt = T / steps. Values are rolled back with discount exp(-r dt); for
    American exercise each node takes max(continuation, intrinsic). European
    prices converge to Black-Scholes-Merton with error O(1/steps), oscillating
    between odd and even step counts. The backward induction is vectorised
    over the nodes of each time step.
    """

    def __init__(self, steps: int = 100):
        """
        Parameters:
        -----------
        steps : int
            Number of time steps (>= 1).
        """
        if int(steps) < 1:
            raise ValueError("steps must be at least 1")
        self.steps = int(steps)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        american: bool = False,
        q: float = 0.0,
    ) -> float:
        """
        Price an option on the tree.

        Parameters:
        -----------
        S, K : float
            Spot and strike.
        T : float
            Time to expiry in years; T <= 0 returns the intrinsic value.
        r, q : float
            Continuously compounded risk-free rate and dividend yield.
        sigma : float
            Annualised volatility.
        option_type : str
            'call' or 'put'
        american : bool
            Allow early exercise at every node.

        Returns:
        --------
        float
            Option price
        """
        ot = option_type.lower()
        if ot not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        sign = 1.0 if ot == "call" else -1.0
        if np.isfinite(T) and T <= 0:
            if not (S > 0 and K > 0):
                raise ValueError("S and K must be positive")
            return max(sign * (S - K), 0.0)
        _validate_market_inputs(S, K, T, sigma)

        n = self.steps
        dt = T / n
        step = sigma * np.sqrt(dt)
        u = np.exp(step)
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        if not 0.0 < p < 1.0:
            raise ValueError(
                f"risk-neutral probability {p:.4f} is outside (0, 1); increase the number of steps"
            )
        disc = np.exp(-r * dt)

        def spots(level: int) -> np.ndarray:
            # Node j at time level has price S u^(level - j) d^j = S exp((level - 2 j) step).
            return S * np.exp((level - 2.0 * np.arange(level + 1)) * step)

        values = np.maximum(sign * (spots(n) - K), 0.0)
        for level in range(n - 1, -1, -1):
            values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
            if american:
                values = np.maximum(values, sign * (spots(level) - K))
        return float(values[0])


class MonteCarloEngine(PricingMethod):
    """
    Monte Carlo pricer for European payoffs under Black-Scholes-Merton dynamics.

    Terminal prices are sampled exactly, S_T = S exp((r - q - sigma^2/2) T +
    sigma sqrt(T) Z), so there is no time-discretisation error. Optional
    variance reduction:

    - antithetic variates: each normal draw Z is paired with -Z and the
      estimator averages the pair (Glasserman, Monte Carlo Methods in
      Financial Engineering, 2003, section 4.2);
    - control variate: the discounted terminal price, whose expectation
      S e^{-qT} is known, with the regression coefficient estimated from the
      same sample (Glasserman, section 4.1).

    Random numbers come from a private ``numpy.random.Generator``; NumPy's
    global random state is never touched.
    """

    def __init__(
        self,
        num_sims: int = 100000,
        random_seed: int | None = None,
        antithetic: bool = False,
        control_variate: bool = False,
    ):
        """
        Parameters:
        -----------
        num_sims : int
            Number of simulated payoffs (with antithetic variates: number of
            draws, i.e. num_sims / 2 antithetic pairs).
        random_seed : int, optional
            Seed for the engine's random number generator.
        antithetic, control_variate : bool
            Default variance-reduction settings for :meth:`price`.
        """
        if num_sims < 2:
            raise ValueError("num_sims must be at least 2")
        self.num_sims = int(num_sims)
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.rng = np.random.default_rng(random_seed)

    def generate_gbm_paths(
        self, S0: float, T: float, r: float, sigma: float, num_steps: int = 252, q: float = 0.0
    ) -> np.ndarray:
        """
        Simulate GBM paths on an equally spaced grid with the exact log-normal step.

        Returns:
        --------
        np.ndarray
            Shape (num_sims, num_steps + 1); column 0 is S0.
        """
        dt = T / num_steps
        Z = self.rng.standard_normal((self.num_sims, num_steps))
        increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.zeros((self.num_sims, num_steps + 1))
        np.cumsum(increments, axis=1, out=log_paths[:, 1:])
        return S0 * np.exp(log_paths)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        payoff_func: Callable | None = None,
        *,
        q: float = 0.0,
        antithetic: bool | None = None,
        control_variate: bool | None = None,
    ) -> dict[str, Any]:
        """
        Monte Carlo price of a European option.

        Parameters:
        -----------
        S, K : float
            Spot and strike.
        T : float
            Time to expiry in years.
        r, q : float
            Continuously compounded risk-free rate and dividend yield.
        sigma : float
            Annualised volatility.
        option_type : str
            'call' or 'put' (ignored when ``payoff_func`` is given).
        payoff_func : callable, optional
            ``payoff_func(S_T, K)`` returning payoffs for an array of terminal
            prices.
        antithetic, control_variate : bool, optional
            Override the engine defaults.

        Returns:
        --------
        dict
            'price', 'std_error' (sample standard deviation / sqrt(number of
            independent samples), ddof=1), 'confidence_interval' (95%, price
            +- 1.96 standard errors), 'num_simulations' and 'method'.
        """
        _validate_market_inputs(S, K, T, sigma)
        antithetic = self.antithetic if antithetic is None else antithetic
        control_variate = self.control_variate if control_variate is None else control_variate
        if payoff_func is None:
            ot = option_type.lower()
            if ot not in ("call", "put"):
                raise ValueError("option_type must be 'call' or 'put'")
            sign = 1.0 if ot == "call" else -1.0

            def payoff_func(s_t: np.ndarray, strike: float) -> np.ndarray:
                return np.maximum(sign * (s_t - strike), 0.0)

        n = self.num_sims
        n_draws = n // 2 if antithetic else n
        Z = self.rng.standard_normal(n_draws)
        drift = (r - q - 0.5 * sigma**2) * T
        vol = sigma * np.sqrt(T)
        disc = np.exp(-r * T)

        def discounted(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s_t = S * np.exp(drift + vol * z)
            return disc * payoff_func(s_t, K), disc * s_t

        y, x = discounted(Z)
        if antithetic:
            y2, x2 = discounted(-Z)
            y, x = 0.5 * (y + y2), 0.5 * (x + x2)  # one sample per antithetic pair
        if control_variate:
            # E[e^{-rT} S_T] = S e^{-qT}
            cov = np.cov(y, x, ddof=1)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
            y = y - beta * (x - S * np.exp(-q * T))

        price = float(y.mean())
        std_error = float(y.std(ddof=1) / np.sqrt(y.size))
        method = "+".join(
            m for m, on in (("antithetic", antithetic), ("control_variate", control_variate)) if on
        )
        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (price - 1.96 * std_error, price + 1.96 * std_error),
            "num_simulations": n_draws * (2 if antithetic else 1),
            "method": method or "plain",
        }


class PricingEngine:
    """
    Dispatch to a pricer by name: 'black_scholes', 'binomial' (100-step CRR
    tree) or 'monte_carlo' (100,000 draws, unseeded).

    Deterministic results are cached by their inputs; Monte Carlo results are
    not, so repeated calls give independent estimates.
    """

    def __init__(self) -> None:
        self.methods: dict[str, PricingMethod] = {
            "black_scholes": BlackScholesAnalytical(),
            "binomial": BinomialTree(),
            "monte_carlo": MonteCarloEngine(),
        }
        self.results_cache: dict[str, Any] = {}

    def price_option(
        self,
        method: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        **kwargs: Any,
    ) -> float | dict[str, Any]:
        """
        Price an option with the named method.

        Parameters:
        -----------
        method : str
            'black_scholes', 'binomial' or 'monte_carlo'.
        S, K, T, r, sigma, option_type
            As in :meth:`BlackScholesAnalytical.price`.
        kwargs
            Keyword arguments: Passed to the pricer, e.g. ``q`` (all methods) or ``american``
            (binomial).

        Returns:
        --------
        float or dict
            A price for 'black_scholes' and 'binomial'; the Monte Carlo result
            dict (price, std_error, confidence_interval, ...) otherwise.
        """
        if method not in self.methods:
            raise ValueError(f"Unknown pricing method: {method}")

        # Create cache key (include method-specific kwargs, e.g. dividend yield q,
        # so calls that differ only by a kwarg do not collide in the cache)
        extra = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        cache_key = f"{method}_{S}_{K}_{T}_{r}_{sigma}_{option_type}_{extra}"

        cacheable = method != "monte_carlo"
        if cacheable and cache_key in self.results_cache:
            return self.results_cache[cache_key]

        # Price using selected method
        pricing_method = self.methods[method]

        try:
            result = pricing_method.price(
                S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, **kwargs
            )

            if cacheable:
                self.results_cache[cache_key] = result
            return result
        except Exception as e:
            raise ValueError(f"Pricing failed with method {method}: {e!s}") from e

    def compare_methods(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        methods: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Price the same European option with several methods.

        Returns:
        --------
        dict
            method name -> {'price': ...} (plus the Monte Carlo statistics), or
            {'error': message} if that method failed.
        """
        if methods is None:
            methods = list(self.methods.keys())

        results = {}

        for method in methods:
            try:
                result = self.price_option(
                    method=method, S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type
                )

                if isinstance(result, dict):
                    results[method] = result
                else:
                    results[method] = {"price": result}

            except Exception as e:
                results[method] = {"error": str(e)}

        return results

    def validate_put_call_parity(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        method: str = "black_scholes",
        q: float = 0.0,
    ) -> dict[str, Any]:
        """
        Check put-call parity C - P = S e^{-qT} - K e^{-rT} for a method.

        Returns:
        --------
        dict
            call and put prices, both sides of the identity, absolute and
            relative error, the 'tolerance' and 'parity_satisfied' (absolute
            error below the tolerance). The tolerance is 1e-6 for the closed
            form and the tree; Monte Carlo prices the call and the put from
            different draws, so there it is four times the sum of their
            standard errors.
        """
        call_price = self.price_option(
            method=method, S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", q=q
        )
        put_price = self.price_option(
            method=method, S=S, K=K, T=T, r=r, sigma=sigma, option_type="put", q=q
        )

        tolerance = 1e-6
        if isinstance(call_price, dict) and isinstance(put_price, dict):  # Monte Carlo
            tolerance = 4.0 * (call_price["std_error"] + put_price["std_error"])
            call_price, put_price = call_price["price"], put_price["price"]

        # Calculate put-call parity components
        left_side = call_price - put_price
        right_side = S * np.exp(-q * T) - K * np.exp(-r * T)
        error = abs(left_side - right_side)
        relative_error = error / abs(right_side) if right_side != 0 else float("inf")

        return {
            "call_price": call_price,
            "put_price": put_price,
            "left_side": left_side,
            "right_side": right_side,
            "absolute_error": error,
            "relative_error": relative_error,
            "tolerance": tolerance,
            "parity_satisfied": bool(error < tolerance),
        }


def price_european_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    method: str = "black_scholes",
    q: float = 0.0,
) -> float:
    """
    Price a European option and return the price as a float.

    Parameters:
    -----------
    S, K : float
        Spot and strike.
    T : float
        Time to expiry in years.
    r, q : float
        Continuously compounded risk-free rate and dividend yield.
    sigma : float
        Annualised volatility.
    option_type : str
        'call' or 'put'
    method : str
        'black_scholes' (closed form), 'binomial' (100-step tree) or
        'monte_carlo' (100,000 draws; the estimate is random).

    Returns:
    --------
    float
        Option price
    """
    result = PricingEngine().price_option(method, S, K, T, r, sigma, option_type, q=q)
    if isinstance(result, dict):
        return float(result["price"])
    return float(result)
