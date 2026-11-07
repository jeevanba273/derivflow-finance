"""
Black-Scholes-Merton Greeks: closed forms and finite differences.

Units (the same for both methods and for every Greek reported here):

=========  ===========================================  =======================
Greek      Definition                                   Reported per
=========  ===========================================  =======================
delta      dV/dS                                        1 unit of spot
gamma      d2V/dS2                                      1 unit of spot
theta      dV/dt (calendar time, = -dV/dT)              calendar day (1/365 y)
vega       dV/dsigma                                    1 vol point (0.01)
rho        dV/dr                                        1 rate point (0.01)
volga      d2V/dsigma2                                  (1 vol point)^2
vanna      d2V/dS dsigma                                1 vol point
speed      d3V/dS3                                      1 unit of spot
zomma      d3V/dS2 dsigma                               1 vol point
color      d3V/dS2 dt (calendar time)                   calendar day
=========  ===========================================  =======================

The closed forms are those of Haug, The Complete Guide to Option Pricing
Formulas (2nd ed., 2007), chapter 2, for the generalised Black-Scholes-Merton
model with continuous dividend yield q. Every analytical Greek is checked
against central finite differences of the price in the test suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._validation import check_market_inputs, check_option_type, norm_cdf, norm_pdf


@dataclass
class GreeksResult:
    """Greeks of one option, in the units listed in the module docstring.
    Second- and third-order Greeks are None for the finite-difference method."""

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    # Second-order Greeks
    volga: float | None = None
    vanna: float | None = None
    # Third-order Greeks
    speed: float | None = None
    zomma: float | None = None
    color: float | None = None


class GreeksCalculator:
    """
    Greeks of European options.

    method='analytical' uses the Black-Scholes-Merton closed forms;
    method='numerical' bumps the inputs of a pricing function (Black-Scholes by
    default, or any ``pricing_func(S, K, T, r, sigma, option_type, ...)``) and
    takes central differences for delta, gamma, vega and rho and a one-day
    forward difference for theta.
    """

    def __init__(self, method: str = "analytical"):
        """
        Parameters:
        -----------
        method : str
            'analytical' or 'numerical'.
        """
        if method not in ("analytical", "numerical"):
            raise ValueError("method must be 'analytical' or 'numerical'")
        self.method = method
        self.default_bumps = {
            "delta": 0.01,  # 1% bump for delta
            "gamma": 0.01,  # 1% bump for gamma
            "theta": 1 / 365,  # 1 day for theta
            "vega": 0.01,  # 1% vol bump
            "rho": 0.01,  # 1% rate bump
        }

    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        """Black-Scholes-Merton price used as the default for numerical Greeks."""
        if T <= 0:
            if option_type.lower() == "call":
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        disc_S = S * math.exp(-q * T)
        disc_K = K * math.exp(-r * T)

        if option_type.lower() == "call":
            return disc_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
        return disc_K * norm_cdf(-d2) - disc_S * norm_cdf(-d1)

    def analytical_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> GreeksResult:
        """
        Closed-form Greeks of a European option.

        With d1, d2 as in Black-Scholes-Merton, phi the normal density and
        N the normal CDF (call shown; puts follow from parity)::

            delta = e^{-qT} N(d1)
            gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))
            vega  = S e^{-qT} phi(d1) sqrt(T)
            theta = -S e^{-qT} phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
            rho   = K T e^{-rT} N(d2)
            volga = vega d1 d2 / sigma
            vanna = -e^{-qT} phi(d1) d2 / sigma
            speed = -gamma (d1 / (sigma sqrt(T)) + 1) / S
            zomma = gamma (d1 d2 - 1) / sigma
            color = gamma (q + (r - q) d1 / (sigma sqrt(T)) + (1 - d1 d2) / (2T))

        These are then scaled to the units in the module docstring. At expiry (T <= 0)
        delta is the payoff slope and the other Greeks are 0.

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

        Returns:
        --------
        GreeksResult
        """
        check_market_inputs(S, K, T, sigma)
        check_option_type(option_type)
        if T <= 0:
            # At expiry the value is the payoff: delta is 1 for an in-the-money
            # call, -1 for an in-the-money put and 0 otherwise.
            if option_type.lower() == "call":
                expiry_delta = 1.0 if S > K else 0.0
            else:
                expiry_delta = -1.0 if S < K else 0.0
            return GreeksResult(
                delta=expiry_delta,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
            )

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc_q = math.exp(-q * T)
        disc_r = math.exp(-r * T)

        phi_d1 = norm_pdf(d1)
        N_d1 = norm_cdf(d1)
        N_d2 = norm_cdf(d2)
        N_minus_d1 = norm_cdf(-d1)
        N_minus_d2 = norm_cdf(-d2)

        # First-order Greeks
        if option_type.lower() == "call":
            delta = disc_q * N_d1
            theta = (
                -disc_q * S * phi_d1 * sigma / (2 * sqrt_t)
                - r * K * disc_r * N_d2
                + q * S * disc_q * N_d1
            )
            rho = K * T * disc_r * N_d2
        else:  # put
            delta = disc_q * (N_d1 - 1)
            theta = (
                -disc_q * S * phi_d1 * sigma / (2 * sqrt_t)
                + r * K * disc_r * N_minus_d2
                - q * S * disc_q * N_minus_d1
            )
            rho = -K * T * disc_r * N_minus_d2

        # Second-order Greeks (same for calls and puts)
        gamma = disc_q * phi_d1 / (S * sigma * sqrt_t)
        vega = S * disc_q * phi_d1 * sqrt_t

        # Higher-order Greeks
        volga = vega * d1 * d2 / sigma  # Volga (vega sensitivity to vol)
        vanna = -disc_q * phi_d1 * d2 / sigma  # Vanna (delta sensitivity to vol)

        # Third-order Greeks
        speed = -gamma * (d1 / (sigma * sqrt_t) + 1) / S  # Speed (gamma sensitivity to spot)
        zomma = gamma * (d1 * d2 - 1) / sigma  # Zomma (gamma sensitivity to vol)
        # Color = dGamma/dt in calendar time (= -dGamma/dT), the same sign
        # convention as theta. Differentiating the gamma closed form in T gives
        # dGamma/dt = Gamma * (q + (r - q) d1 / (sigma sqrt(T)) + (1 - d1 d2) / (2 T)).
        color = gamma * (q + (r - q) * d1 / (sigma * sqrt_t) + (1 - d1 * d2) / (2 * T))

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta / 365,  # Convert to per-day
            vega=vega / 100,  # Convert to per 1% vol change
            rho=rho / 100,  # Convert to per 1% rate change
            volga=volga / 1e4,  # change in vega (per vol point) per vol point
            vanna=vanna / 100,  # Convert to per 1% vol change
            speed=speed,
            zomma=zomma / 100,
            color=color / 365,  # Convert to per-day
        )

    def numerical_greeks(
        self,
        pricing_func: Callable[..., float],
        base_params: dict[str, Any],
        bumps: dict[str, float] | None = None,
    ) -> GreeksResult:
        """
        Finite-difference Greeks of ``pricing_func(**base_params)``.

        Default bumps: spot +-1% (relative, central differences for delta and
        gamma), sigma +-1% (relative, central), r +-0.01 (absolute, central),
        T -1 day (forward). Results use the module's units.

        Parameters:
        -----------
        pricing_func : callable
            Pricer taking S, K, T, r, sigma (and any other base_params) as
            keyword arguments.
        base_params : dict
            Inputs at which the Greeks are computed.
        bumps : dict, optional
            Override of ``default_bumps`` (keys 'delta', 'theta', 'vega', 'rho').

        Returns:
        --------
        GreeksResult
            Greeks with the higher-order fields left as None.
        """
        check_market_inputs(
            base_params["S"], base_params["K"], base_params["T"], base_params["sigma"]
        )
        if bumps is None:
            bumps = self.default_bumps.copy()

        # Base price
        base_price = pricing_func(**base_params)

        # Delta: dP/dS
        params_up = base_params.copy()
        params_down = base_params.copy()
        params_up["S"] = base_params["S"] * (1 + bumps["delta"])
        params_down["S"] = base_params["S"] * (1 - bumps["delta"])

        price_up = pricing_func(**params_up)
        price_down = pricing_func(**params_down)
        delta = (price_up - price_down) / (2 * base_params["S"] * bumps["delta"])

        # Gamma: d2P/dS2
        gamma = (price_up - 2 * base_price + price_down) / (base_params["S"] * bumps["delta"]) ** 2

        # Theta: forward difference in calendar time, scaled to per calendar day
        # (the same unit as the analytical theta). Divide by the step actually
        # taken, which is shorter than the bump when T is within a bump of expiry.
        params_theta = base_params.copy()
        params_theta["T"] = max(0.5 * base_params["T"], base_params["T"] - bumps["theta"])
        step = base_params["T"] - params_theta["T"]
        if step > 0:
            price_theta = pricing_func(**params_theta)
            theta = (price_theta - base_price) / step / 365.0
        else:  # at expiry (T = 0) no time is left to decay, as in analytical_greeks
            theta = 0.0

        # Vega: dP/dsigma
        params_vega_up = base_params.copy()
        params_vega_down = base_params.copy()
        params_vega_up["sigma"] = base_params["sigma"] * (1 + bumps["vega"])
        params_vega_down["sigma"] = base_params["sigma"] * (1 - bumps["vega"])

        price_vega_up = pricing_func(**params_vega_up)
        price_vega_down = pricing_func(**params_vega_down)
        vega = (price_vega_up - price_vega_down) / (2 * base_params["sigma"] * bumps["vega"])

        # Rho: dP/dr
        params_rho_up = base_params.copy()
        params_rho_down = base_params.copy()
        params_rho_up["r"] = base_params["r"] + bumps["rho"]
        params_rho_down["r"] = base_params["r"] - bumps["rho"]

        price_rho_up = pricing_func(**params_rho_up)
        price_rho_down = pricing_func(**params_rho_down)
        rho = (price_rho_up - price_rho_down) / (2 * bumps["rho"])

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega / 100,  # Convert to per 1% vol change
            rho=rho / 100,  # Convert to per 1% rate change
        )

    def portfolio_greeks(self, positions: list[dict[str, Any]]) -> dict[str, float]:
        """
        Sum Greeks over positions, weighting each by quantity x notional

        Parameters:
        -----------
        positions : List[Dict]
            List of position dictionaries with 'quantity', 'greeks', and optionally 'notional'

        Returns:
        --------
        Dict[str, float]
            Aggregated portfolio Greeks
        """
        portfolio_greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "total_notional": 0.0,
        }

        for position in positions:
            quantity = position["quantity"]
            greeks = position["greeks"]
            notional = position.get("notional", 1.0)

            # Weight Greeks by quantity and notional
            weight = quantity * notional

            portfolio_greeks["delta"] += weight * greeks.delta
            portfolio_greeks["gamma"] += weight * greeks.gamma
            portfolio_greeks["theta"] += weight * greeks.theta
            portfolio_greeks["vega"] += weight * greeks.vega
            portfolio_greeks["rho"] += weight * greeks.rho
            portfolio_greeks["total_notional"] += abs(weight)

        return portfolio_greeks

    def greeks_ladder(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        spot_range: tuple[float, float] = (0.8, 1.2),
        num_points: int = 21,
        option_type: str = "call",
        q: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """
        Calculate Greeks across a range of spot prices (Greeks ladder)

        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiry
        r : float
            Risk-free rate
        sigma : float
            Volatility
        spot_range : tuple
            Range of spot prices as (min_ratio, max_ratio) relative to current spot
        num_points : int
            Number of points in the ladder
        option_type : str
            'call' or 'put'

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with spot prices and corresponding Greeks
        """
        spot_prices = np.linspace(S * spot_range[0], S * spot_range[1], num_points)

        results = {
            "spot_prices": spot_prices,
            "delta": np.zeros(num_points),
            "gamma": np.zeros(num_points),
            "theta": np.zeros(num_points),
            "vega": np.zeros(num_points),
            "rho": np.zeros(num_points),
        }

        for i, spot in enumerate(spot_prices):
            greeks = self.analytical_greeks(spot, K, T, r, sigma, option_type, q)
            results["delta"][i] = greeks.delta
            results["gamma"][i] = greeks.gamma
            results["theta"][i] = greeks.theta
            results["vega"][i] = greeks.vega
            results["rho"][i] = greeks.rho

        return results

    def calculate_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        pricing_func: Callable[..., float] | None = None,
        *,
        q: float = 0.0,
        **kwargs: Any,
    ) -> GreeksResult:
        """
        Main interface for Greeks calculation

        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiry
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        pricing_func : callable, optional
            Custom pricing function for numerical Greeks, called with S, K, T,
            r, sigma, option_type, q (when q is not zero) and ``kwargs`` as
            keyword arguments.
        q : float, keyword-only
            Continuous dividend yield (default 0.0 reproduces classic Black-Scholes)
        kwargs
            Keyword arguments: Additional parameters

        Returns:
        --------
        GreeksResult
            Complete Greeks calculation
        """
        if self.method == "analytical":
            return self.analytical_greeks(S, K, T, r, sigma, option_type, q)

        elif self.method == "numerical":
            base_params = {
                "S": S,
                "K": K,
                "T": T,
                "r": r,
                "sigma": sigma,
                "option_type": option_type,
            }
            if pricing_func is None:
                # Default Black-Scholes pricer supports a dividend yield
                pricing_func = self._black_scholes_price
                base_params["q"] = q
            elif q != 0.0:
                # A custom pricer gets the dividend yield too; one that does not
                # accept q raises TypeError rather than silently ignoring it.
                base_params["q"] = q
            base_params.update(kwargs)

            return self.numerical_greeks(pricing_func, base_params)

        else:
            raise ValueError(f"Unknown method: {self.method}")


def format_greeks_report(
    greeks: GreeksResult, S: float, K: float, option_type: str = "call"
) -> str:
    """Plain-text summary of a :class:`GreeksResult` with the units of each Greek."""
    ot = option_type.lower()
    if abs(S - K) / K < 0.02:
        moneyness = "ATM (within 2% of strike)"
    elif (ot == "call" and S > K) or (ot == "put" and S < K):
        moneyness = "ITM"
    else:
        moneyness = "OTM"
    lines = [
        f"{ot.upper()} option, spot {S:.2f}, strike {K:.2f}, {moneyness}",
        f"  delta  {greeks.delta:12.6f}  per unit of spot",
        f"  gamma  {greeks.gamma:12.6f}  per unit of spot",
        f"  theta  {greeks.theta:12.6f}  per calendar day",
        f"  vega   {greeks.vega:12.6f}  per vol point",
        f"  rho    {greeks.rho:12.6f}  per rate point",
    ]
    extra = [
        ("volga", greeks.volga, "per vol point squared"),
        ("vanna", greeks.vanna, "per vol point"),
        ("speed", greeks.speed, "per unit of spot"),
        ("zomma", greeks.zomma, "per vol point"),
        ("color", greeks.color, "per calendar day"),
    ]
    lines += [f"  {n:<6} {v:12.6g}  {u}" for n, v, u in extra if v is not None]
    return "\n".join(lines)
