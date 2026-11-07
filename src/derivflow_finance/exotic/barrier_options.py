"""
Single-barrier European options (up/down, in/out, calls and puts, no rebate).

- :meth:`BarrierOptions.analytical_price`: closed form for a continuously
  monitored barrier under Black-Scholes dynamics with no dividend yield
  (Merton 1973; Reiner and Rubinstein 1991), written with the A, B, C, D
  building blocks of Haug, The Complete Guide to Option Pricing Formulas
  (2nd ed., 2007), section 4.17.1. Matches QuantLib's AnalyticBarrierEngine
  to about 1e-14 in the validation tests.
- :meth:`BarrierOptions.monte_carlo_price`: exact GBM paths monitored at
  ``num_steps`` equally spaced dates, i.e. a discretely monitored barrier.
  Its price differs from the continuous closed form by the discrete
  monitoring bias; Broadie, Glasserman and Kou (1997) show the discrete
  price is close to the continuous one with the barrier shifted by
  exp(+-0.5826 sigma sqrt(dt)) away from the spot.

Knock-in plus knock-out equals the vanilla option (in-out parity).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats as stats

from .._validation import check_market_inputs, check_option_type


class BarrierType(Enum):
    """Enumeration of barrier option types"""

    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"


@dataclass
class BarrierOptionResult:
    """Barrier option price.

    ``probability_survival`` is the risk-neutral probability that the barrier
    is not touched before expiry (continuous monitoring for the closed form,
    the simulated frequency for Monte Carlo). ``std_error`` is set for Monte
    Carlo prices only.
    """

    price: float
    probability_survival: float | None = None
    std_error: float | None = None


class BarrierOptions:
    """
    Pricer for single-barrier European options; see the module docstring for
    the methods and their assumptions.
    """

    def __init__(self, random_seed: int | None = None):
        """
        Parameters:
        -----------
        random_seed : int, optional
            Seed of the generator used by :meth:`monte_carlo_price`.
        """
        # Private generator: NumPy's global random state is never touched.
        self.rng = np.random.default_rng(random_seed)

    def _validate_barrier_parameters(
        self,
        S: float,
        K: float,
        H: float,
        barrier_type: str,
        option_type: str,
        T: float = 1.0,
        sigma: float | None = None,
    ) -> None:
        """Validate inputs; warn when the barrier is placed where it has no effect."""
        check_market_inputs(S, K, T, sigma)
        check_option_type(option_type)
        if not (np.isfinite(H) and H > 0):
            raise ValueError(f"H must be positive and finite (barrier level), got {H!r}")
        bt = str(barrier_type).lower()
        if bt not in ("up_and_out", "up_and_in", "down_and_out", "down_and_in"):
            raise ValueError(
                "barrier_type must be 'up_and_out', 'up_and_in', 'down_and_out' or "
                f"'down_and_in', got {barrier_type!r}"
            )
        if bt.startswith("up") and H <= max(S, K):
            warnings.warn(f"Up barrier ({H}) should be above max(S={S}, K={K})", stacklevel=3)
        elif bt.startswith("down") and H >= min(S, K):
            warnings.warn(f"Down barrier ({H}) should be below min(S={S}, K={K})", stacklevel=3)

    def analytical_price(
        self,
        S: float,
        K: float,
        H: float,
        T: float,
        r: float,
        sigma: float,
        barrier_type: str,
        option_type: str = "call",
    ) -> BarrierOptionResult:
        """
        Closed-form price of a continuously monitored barrier option.

        Valid for Black-Scholes dynamics without dividends and zero rebate.
        If the barrier is already breached at t = 0, a knock-in is worth the
        vanilla option and a knock-out is worth 0. At T <= 0 see
        :meth:`_handle_expiry`.

        Parameters:
        -----------
        S, K, H : float
            Spot, strike and barrier level.
        T : float
            Time to expiry in years.
        r : float
            Continuously compounded risk-free rate.
        sigma : float
            Annualised volatility.
        barrier_type : str
            'up_and_out', 'up_and_in', 'down_and_out' or 'down_and_in'.
        option_type : str
            'call' or 'put'

        Returns:
        --------
        BarrierOptionResult
            price and probability_survival.
        """
        self._validate_barrier_parameters(S, K, H, barrier_type, option_type, T, sigma)

        if T <= 0:
            return self._handle_expiry(S, K, H, barrier_type, option_type)

        # Already-breached barrier at t=0: a knock-in is immediately activated and
        # equals the vanilla option; a knock-out is already dead and worth 0. The
        # Haug image formulas below are only valid for an un-breached barrier
        # (down: S > H, up: S < H), so short-circuit these degenerate cases to
        # preserve the no-arbitrage bound knock_in <= vanilla.
        bt_lower = barrier_type.lower()
        is_down = bt_lower.startswith("down")
        already_breached = (is_down and S <= H) or ((not is_down) and S >= H)
        if already_breached:
            if bt_lower.endswith("out"):
                price = 0.0
            else:
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                if option_type.lower() == "call":
                    price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
                else:
                    price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            survival_prob = self._calculate_survival_probability(S, H, T, r, sigma, barrier_type)
            return BarrierOptionResult(price=max(price, 0), probability_survival=survival_prob)

        # Reiner-Rubinstein / Haug standard barrier formulas (zero rebate).
        # Building blocks A, B, C, D parameterized by phi (call=+1, put=-1) and
        # eta (down barrier=+1, up barrier=-1). See Haug, "The Complete Guide to
        # Option Pricing Formulas".
        sig_t = sigma * np.sqrt(T)
        mu = (r - 0.5 * sigma**2) / sigma**2
        lam_exp = mu + 1  # = (r + 0.5*sigma**2)/sigma**2

        x1 = np.log(S / K) / sig_t + lam_exp * sig_t
        x2 = np.log(S / H) / sig_t + lam_exp * sig_t
        y1 = np.log(H**2 / (S * K)) / sig_t + lam_exp * sig_t
        y2 = np.log(H / S) / sig_t + lam_exp * sig_t

        N = stats.norm.cdf
        disc = np.exp(-r * T)

        def A(phi):
            return phi * S * N(phi * x1) - phi * K * disc * N(phi * x1 - phi * sig_t)

        def B(phi):
            return phi * S * N(phi * x2) - phi * K * disc * N(phi * x2 - phi * sig_t)

        def C(phi, eta):
            return phi * S * (H / S) ** (2 * lam_exp) * N(eta * y1) - phi * K * disc * (H / S) ** (
                2 * mu
            ) * N(eta * y1 - eta * sig_t)

        def D(phi, eta):
            return phi * S * (H / S) ** (2 * lam_exp) * N(eta * y2) - phi * K * disc * (H / S) ** (
                2 * mu
            ) * N(eta * y2 - eta * sig_t)

        bt = barrier_type.lower()
        is_call = option_type.lower() == "call"
        phi = 1.0 if is_call else -1.0
        k_ge_h = K >= H

        if is_call:
            eta = 1.0 if bt.startswith("down") else -1.0
            if bt == "down_and_in":
                price = C(phi, eta) if k_ge_h else A(phi) - B(phi) + D(phi, eta)
            elif bt == "down_and_out":
                price = A(phi) - C(phi, eta) if k_ge_h else B(phi) - D(phi, eta)
            elif bt == "up_and_in":
                price = A(phi) if k_ge_h else B(phi) - C(phi, eta) + D(phi, eta)
            elif bt == "up_and_out":
                price = 0.0 if k_ge_h else A(phi) - B(phi) + C(phi, eta) - D(phi, eta)
            else:
                raise ValueError(f"Unknown barrier type: {barrier_type}")
        else:  # put
            eta = 1.0 if bt.startswith("down") else -1.0
            if bt == "down_and_in":
                price = B(phi) - C(phi, eta) + D(phi, eta) if k_ge_h else A(phi)
            elif bt == "down_and_out":
                price = A(phi) - B(phi) + C(phi, eta) - D(phi, eta) if k_ge_h else 0.0
            elif bt == "up_and_in":
                price = A(phi) - B(phi) + D(phi, eta) if k_ge_h else C(phi, eta)
            elif bt == "up_and_out":
                price = B(phi) - D(phi, eta) if k_ge_h else A(phi) - C(phi, eta)
            else:
                raise ValueError(f"Unknown barrier type: {barrier_type}")

        # Calculate survival probability
        survival_prob = self._calculate_survival_probability(S, H, T, r, sigma, barrier_type)

        return BarrierOptionResult(
            price=max(price, 0),  # Ensure non-negative price
            probability_survival=survival_prob,
        )

    def _calculate_survival_probability(
        self, S: float, H: float, T: float, r: float, sigma: float, barrier_type: str
    ) -> float:
        """Risk-neutral probability that a continuously monitored barrier H is not
        touched before T, from the reflection principle for X_t = ln(S_t / S)
        with drift mu = r - sigma^2 / 2."""
        mu = r - 0.5 * sigma**2

        if barrier_type.lower() in ["down_and_out", "down_and_in"]:
            if S <= H:
                return 0.0  # Already hit
            # Probability of not hitting down barrier
            d = (np.log(S / H) + mu * T) / (sigma * np.sqrt(T))
            prob = stats.norm.cdf(d) - (H / S) ** (2 * mu / sigma**2) * stats.norm.cdf(
                d - 2 * np.log(S / H) / (sigma * np.sqrt(T))
            )

        elif barrier_type.lower() in ["up_and_out", "up_and_in"]:
            if S >= H:
                return 0.0  # Already hit
            # Probability of not hitting up barrier: the reflection term carries
            # exp(2 mu b / sigma^2) = (H/S)^(2 mu / sigma^2) with b = ln(H/S).
            d = (np.log(H / S) - mu * T) / (sigma * np.sqrt(T))
            prob = stats.norm.cdf(d) - (H / S) ** (2 * mu / sigma**2) * stats.norm.cdf(
                d - 2 * np.log(H / S) / (sigma * np.sqrt(T))
            )

        else:
            prob = 1.0

        return max(0.0, min(1.0, prob))

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        H: float,
        T: float,
        r: float,
        sigma: float,
        barrier_type: str,
        option_type: str = "call",
        num_sims: int = 100000,
        num_steps: int = 252,
    ) -> BarrierOptionResult:
        """
        Monte Carlo price with the barrier monitored at ``num_steps`` dates.

        Paths are exact GBM at t_i = i T / num_steps; a path is knocked when
        any S(t_i) is on or beyond H. This is a discretely monitored barrier,
        so the price is not the continuous closed form (see the module
        docstring). Returns the price, its standard error (ddof=1) and the
        simulated no-touch frequency.
        """
        self._validate_barrier_parameters(S, K, H, barrier_type, option_type, T, sigma)
        if T <= 0:
            raise ValueError(f"T must be positive (years to expiry), got {T!r}")
        dt = T / num_steps
        discount_factor = np.exp(-r * T)

        # Simulate exact GBM paths on the monitoring dates
        Z = self.rng.standard_normal((num_sims, num_steps))

        # Vectorized exact-GBM via cumulative sum of log-returns
        increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        cum_log_returns = np.zeros((num_sims, num_steps + 1))
        np.cumsum(increments, axis=1, out=cum_log_returns[:, 1:])
        paths = S * np.exp(cum_log_returns)

        # Check barrier conditions
        barrier_hit = np.zeros(num_sims, dtype=bool)

        if barrier_type.lower() in ["up_and_out", "up_and_in"]:
            barrier_hit = np.any(paths >= H, axis=1)
        elif barrier_type.lower() in ["down_and_out", "down_and_in"]:
            barrier_hit = np.any(paths <= H, axis=1)

        # Calculate payoffs
        final_prices = paths[:, -1]

        if option_type.lower() == "call":
            intrinsic_values = np.maximum(final_prices - K, 0)
        else:
            intrinsic_values = np.maximum(K - final_prices, 0)

        # Apply barrier conditions
        if barrier_type.lower() in ["up_and_out", "down_and_out"]:
            # Knock-out: payoff only if barrier not hit
            payoffs = np.where(barrier_hit, 0, intrinsic_values)
        else:  # knock-in
            # Knock-in: payoff only if barrier was hit
            payoffs = np.where(barrier_hit, intrinsic_values, 0)

        # Calculate price and statistics
        discounted_payoffs = payoffs * discount_factor
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(num_sims)
        survival_probability = np.mean(~barrier_hit)

        # Same quantity as the analytical result: the probability that the
        # barrier is not touched, for knock-in and knock-out options alike.
        return BarrierOptionResult(
            price=float(price),
            probability_survival=float(survival_probability),
            std_error=float(std_error),
        )

    def _handle_expiry(
        self, S: float, K: float, H: float, barrier_type: str, option_type: str
    ) -> BarrierOptionResult:
        """Value at expiry. The path history is unknown, so the barrier counts as
        touched only if the current spot is on or beyond it."""
        payoff = max(S - K, 0.0) if option_type.lower() == "call" else max(K - S, 0.0)
        bt = barrier_type.lower()
        touched = S <= H if bt.startswith("down") else S >= H
        knocked_in = bt.endswith("in")
        alive = touched if knocked_in else not touched
        return BarrierOptionResult(
            price=payoff if alive else 0.0, probability_survival=0.0 if touched else 1.0
        )

    def price(
        self,
        S: float,
        K: float,
        H: float,
        T: float,
        r: float,
        sigma: float,
        barrier_type: str,
        option_type: str = "call",
        method: str = "analytical",
    ) -> BarrierOptionResult:
        """
        Price with ``method='analytical'`` (continuous barrier, default) or
        ``'monte_carlo'`` (discrete monitoring, 100,000 paths x 252 dates).
        """
        if method.lower() == "analytical":
            return self.analytical_price(S, K, H, T, r, sigma, barrier_type, option_type)
        elif method.lower() == "monte_carlo":
            return self.monte_carlo_price(S, K, H, T, r, sigma, barrier_type, option_type)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(
        self,
        S: float,
        K: float,
        H: float,
        T: float,
        r: float,
        sigma: float,
        barrier_type: str,
        option_type: str = "call",
    ) -> dict[str, BarrierOptionResult]:
        """
        Closed form and Monte Carlo side by side ('analytical',
        'monte_carlo'). The two differ by the discrete-monitoring bias as well
        as by Monte Carlo error. Invalid inputs raise ValueError.
        """
        return {
            "analytical": self.analytical_price(S, K, H, T, r, sigma, barrier_type, option_type),
            "monte_carlo": self.monte_carlo_price(
                S, K, H, T, r, sigma, barrier_type, option_type
            ),
        }
