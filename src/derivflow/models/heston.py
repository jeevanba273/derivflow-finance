"""
DERIVFLOW-FINANCE: Heston Stochastic Volatility Model
===================================================

Complete implementation of the Heston model with:
- Analytical pricing (when available)
- Monte Carlo simulation
- Model calibration to market data
- Greeks calculation
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats as stats
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import ndtri



def _heston_cf(u, T, params):
    """Characteristic function of ln(S_T / F_T) (Albrecher et al. 2007 form)."""
    v0, kappa, theta, sigma, rho = (
        params.v0,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
    )
    beta = kappa - rho * sigma * 1j * u
    a = u**2 + 1j * u
    d = np.sqrt(beta**2 + sigma**2 * a)
    # beta - d computed without cancellation: (beta - d)(beta + d) = -sigma^2 a.
    beta_minus_d = -(sigma**2) * a / (beta + d)
    g = beta_minus_d / (beta + d)
    e = np.exp(-d * T)
    D = -a / (beta + d) * (1.0 - e) / (1.0 - g * e)
    C = (
        kappa
        * theta
        / sigma**2
        * (beta_minus_d * T - 2.0 * (np.log1p(-g * e) - np.log1p(-g)))
    )
    return np.exp(C + D * v0)


def heston_call_price(
    S: float, K: float, T: float, r: float, q: float, params: "HestonParameters"
) -> float:
    """European call under Heston via the Lewis (2001) single integral."""
    if T <= 0:
        return max(S - K, 0.0)
    k = np.log(S / K) + (r - q) * T

    def integrand(u: float) -> float:
        return (np.exp(1j * u * k) * _heston_cf(u - 0.5j, T, params)).real / (u * u + 0.25)

    integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500, epsabs=1e-12, epsrel=1e-10)
    price = S * np.exp(-q * T) - np.sqrt(S * K) * np.exp(-0.5 * (r + q) * T) / np.pi * integral
    return max(float(price), 0.0)


@dataclass
class HestonParameters:
    """Heston model parameters"""

    v0: float  # Initial volatility
    kappa: float  # Mean reversion speed
    theta: float  # Long-term volatility
    sigma: float  # Volatility of volatility
    rho: float  # Correlation between asset and volatility


@dataclass
class HestonResult:
    """Heston pricing result. Monte Carlo prices also carry their standard error
    and a 95% confidence interval (price +- 1.96 standard errors)."""

    price: float
    std_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    num_simulations: int | None = None


class HestonModel:
    """
    Heston Stochastic Volatility Model Implementation

    The Heston model assumes the following dynamics:
    dS = rS dt + √V S dW1
    dV = κ(θ - V) dt + σ√V dW2

    where dW1 dW2 = ρ dt
    """

    def __init__(self):
        """Initialize Heston model"""
        self.params = None
        self.calibrated = False

    def set_parameters(
        self, v0: float, kappa: float, theta: float, sigma: float, rho: float
    ) -> None:
        """
        Set Heston model parameters

        Parameters:
        -----------
        v0 : float
            Initial volatility
        kappa : float
            Mean reversion speed (> 0)
        theta : float
            Long-term volatility (> 0)
        sigma : float
            Volatility of volatility (> 0)
        rho : float
            Correlation (-1 < rho < 1)
        """
        # Validate parameters
        if not (kappa > 0 and theta > 0 and sigma > 0):
            raise ValueError("kappa, theta, and sigma must be positive")
        if not (-1 < rho < 1):
            raise ValueError("rho must be between -1 and 1")

        # Check Feller condition
        if 2 * kappa * theta <= sigma**2:
            warnings.warn(
                "Feller condition 2*kappa*theta > sigma**2 is violated; "
                "the variance process can reach zero.",
                stacklevel=2,
            )

        self.params = HestonParameters(v0, kappa, theta, sigma, rho)

    def analytic_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
    ) -> float:
        """
        European option price under the Heston model by numerical integration.

        Uses the single-integral representation of Lewis (2001),

            C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2} / pi *
                int_0^inf Re[e^{i u k} phi(u - i/2)] / (u^2 + 1/4) du,

        with k = ln(S/K) + (r - q) T and phi the characteristic function of
        ln(S_T / F_T), written in the "little Heston trap" form of Albrecher et
        al. (2007), which has no branch-cut discontinuity in the complex
        logarithm. Puts follow from put-call parity. The integral is evaluated
        with adaptive quadrature (``scipy.integrate.quad``).
        """
        if self.params is None:
            raise ValueError("Parameters not set")
        call = heston_call_price(S, K, T, r, q, self.params)
        if option_type.lower() == "call":
            return call
        if option_type.lower() == "put":
            return max(call - S * np.exp(-q * T) + K * np.exp(-r * T), 0.0)
        raise ValueError("option_type must be 'call' or 'put'")

    def monte_carlo_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        num_sims: int = 100_000,
        num_steps: int = 100,
        q: float = 0.0,
        scheme: str = "qe",
        random_seed: int | None = None,
    ) -> HestonResult:
        """
        Price a European option by simulating the Heston model.

        Schemes:

        - ``'qe'`` (default): the quadratic-exponential scheme of Andersen
          (2008) for the variance, with the log-price discretised as in his
          eq. (33) (gamma1 = gamma2 = 1/2). It stays accurate with few time
          steps when the Feller condition fails.
        - ``'euler'``: full-truncation Euler (Lord, Koekkoek and van Dijk,
          2010): negative variances are replaced by zero in the drift and
          diffusion of both processes.

        The paths are simulated one time step at a time, so memory use is
        O(num_sims). Returns the discounted mean payoff, its standard error
        (sample standard deviation / sqrt(num_sims)) and a 95% confidence
        interval.
        """
        if self.params is None:
            raise ValueError("Parameters not set")
        if num_sims < 2 or num_steps < 1:
            raise ValueError("num_sims must be >= 2 and num_steps >= 1")
        if T <= 0:
            raise ValueError("T must be positive")

        p = self.params
        v0, kappa, theta, sigma, rho = p.v0, p.kappa, p.theta, p.sigma, p.rho
        rng = np.random.default_rng(random_seed)
        dt = T / num_steps
        x = np.full(num_sims, np.log(S))
        v = np.full(num_sims, float(v0))

        scheme = scheme.lower()
        if scheme == "qe":
            ek = np.exp(-kappa * dt)
            k0 = -rho * kappa * theta * dt / sigma
            k1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
            k2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
            k3 = 0.5 * dt * (1.0 - rho**2)
            psi_c = 1.5
            for _ in range(num_steps):
                mean = theta + (v - theta) * ek
                var = v * sigma**2 * ek * (1 - ek) / kappa + theta * sigma**2 * (1 - ek) ** 2 / (
                    2 * kappa
                )
                psi = var / mean**2
                u = rng.random(num_sims)
                z = rng.standard_normal(num_sims)
                v_next = np.empty(num_sims)
                quad = psi <= psi_c
                inv = 2.0 / psi[quad]
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
                a = mean[quad] / (1.0 + b2)
                v_next[quad] = a * (np.sqrt(b2) + ndtri(u[quad])) ** 2
                expo = ~quad
                p_mass = (psi[expo] - 1.0) / (psi[expo] + 1.0)
                beta = (1.0 - p_mass) / mean[expo]
                u_e = u[expo]
                v_next[expo] = np.where(
                    u_e <= p_mass,
                    0.0,
                    np.log((1.0 - p_mass) / np.maximum(1.0 - u_e, 1e-300)) / beta,
                )
                x += (r - q) * dt + k0 + k1 * v + k2 * v_next + np.sqrt(k3 * (v + v_next)) * z
                v = v_next
        elif scheme == "euler":
            sqrt_dt = np.sqrt(dt)
            rho_c = np.sqrt(1.0 - rho**2)
            for _ in range(num_steps):
                z1 = rng.standard_normal(num_sims)
                z2 = rho * z1 + rho_c * rng.standard_normal(num_sims)
                v_pos = np.maximum(v, 0.0)
                sqrt_v = np.sqrt(v_pos)
                x += (r - q - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
                v += kappa * (theta - v_pos) * dt + sigma * sqrt_v * sqrt_dt * z2
        else:
            raise ValueError("scheme must be 'qe' or 'euler'")

        s_t = np.exp(x)
        if option_type.lower() == "call":
            payoffs = np.maximum(s_t - K, 0.0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(K - s_t, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        discounted = np.exp(-r * T) * payoffs
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof=1) / np.sqrt(num_sims))
        return HestonResult(
            price=price,
            std_error=std_error,
            confidence_interval=(price - 1.96 * std_error, price + 1.96 * std_error),
            num_simulations=num_sims,
        )

    def calibrate_to_market(self, market_data: list[dict], S: float, r: float) -> dict:
        """
        Calibrate Heston parameters to market option prices

        Parameters:
        -----------
        market_data : List[Dict]
            List of market observations with keys: 'K', 'T', 'price', 'option_type'
        S : float
            Current spot price
        r : float
            Risk-free rate

        Returns:
        --------
        Dict
            Calibration results and fitted parameters
        """

        def objective_function(params):
            v0, kappa, theta, sigma, rho = params

            # Parameter constraints
            if not (
                0.001 < v0 < 1.0
                and 0.001 < kappa < 10.0
                and 0.001 < theta < 1.0
                and 0.001 < sigma < 2.0
                and -0.99 < rho < 0.99
            ):
                return 1e6

            # Set parameters
            try:
                self.set_parameters(v0, kappa, theta, sigma, rho)
            except:
                return 1e6

            total_error = 0
            for data_point in market_data:
                try:
                    model_price = self.monte_carlo_price(
                        S,
                        data_point["K"],
                        data_point["T"],
                        r,
                        data_point.get("option_type", "call"),
                        num_sims=10000,
                    ).price

                    market_price = data_point["price"]
                    error = (model_price - market_price) ** 2 / market_price**2
                    total_error += error

                except:
                    total_error += 1e3

            return total_error

        # Initial parameter guess
        initial_guess = [0.04, 2.0, 0.04, 0.3, -0.7]  # Typical market values

        # Parameter bounds
        bounds = [
            (0.001, 1.0),  # v0
            (0.001, 10.0),  # kappa
            (0.001, 1.0),  # theta
            (0.001, 2.0),  # sigma
            (-0.99, 0.99),  # rho
        ]

        # Optimize
        try:
            result = minimize(
                objective_function,
                initial_guess,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100},
            )

            if result.success:
                v0, kappa, theta, sigma, rho = result.x
                self.set_parameters(v0, kappa, theta, sigma, rho)
                self.calibrated = True

                return {
                    "success": True,
                    "parameters": {
                        "v0": v0,
                        "kappa": kappa,
                        "theta": theta,
                        "sigma": sigma,
                        "rho": rho,
                    },
                    "objective_value": result.fun,
                    "message": "Calibration successful",
                }
            else:
                return {"success": False, "message": f"Optimization failed: {result.message}"}

        except Exception as e:
            return {"success": False, "message": f"Calibration error: {e!s}"}

    def price_option(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        method: str = "analytic",
        q: float = 0.0,
        **mc_kwargs: Any,
    ) -> HestonResult:
        """
        Price a European option under the current Heston parameters.

        Parameters:
        -----------
        S, K : float
            Spot and strike.
        T : float
            Time to expiry in years.
        r, q : float
            Continuously compounded risk-free rate and dividend yield.
        option_type : str
            'call' or 'put'
        method : str
            'analytic' (default; numerical integration, see
            :meth:`analytic_price`; 'fourier' is accepted as an alias) or
            'monte_carlo' (see :meth:`monte_carlo_price`; extra keyword
            arguments such as ``num_sims`` or ``random_seed`` are passed on).

        Returns:
        --------
        HestonResult
            Pricing results
        """
        if self.params is None:
            raise ValueError("Model parameters not set")

        method = method.lower()
        if method in ("analytic", "fourier"):
            return HestonResult(price=self.analytic_price(S, K, T, r, option_type, q))
        if method == "monte_carlo":
            return self.monte_carlo_price(S, K, T, r, option_type, q=q, **mc_kwargs)
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Example usage and testing
    print("DERIVFLOW-FINANCE: Heston Stochastic Volatility Model")
    print("=" * 65)

    # Initialize model
    heston = HestonModel()

    # Set realistic parameters
    heston.set_parameters(
        v0=0.04,  # 4% initial volatility
        kappa=2.0,  # Mean reversion speed
        theta=0.04,  # 4% long-term volatility
        sigma=0.3,  # 30% vol-of-vol
        rho=-0.7,  # Negative correlation (leverage effect)
    )

    # Test parameters
    S, K, T, r = 100, 105, 0.25, 0.05

    print("Heston Model Parameters:")
    print(f"   v₀ = {heston.params.v0:.3f} (initial vol)")
    print(f"   κ  = {heston.params.kappa:.1f} (mean reversion)")
    print(f"   θ  = {heston.params.theta:.3f} (long-term vol)")
    print(f"   σ  = {heston.params.sigma:.1f} (vol-of-vol)")
    print(f"   ρ  = {heston.params.rho:.1f} (correlation)")
    print()

    # Feller condition check
    feller_condition = 2 * heston.params.kappa * heston.params.theta - heston.params.sigma**2
    print(f"Feller Condition: 2κθ - σ² = {feller_condition:.3f}")
    print(f"   Status: {'Satisfied' if feller_condition > 0 else 'Violated'}")
    print()

    print("Option Pricing Comparison:")
    print(f"   Spot: ${S} | Strike: ${K} | Time: {T} years")
    print("-" * 50)

    # Price using Monte Carlo
    mc_result = heston.price_option(S, K, T, r, "call", method="monte_carlo")
    print(f"Heston Monte Carlo:  ${mc_result.price:.4f}")

    # Compare to Black-Scholes (inline calculation)
    vol_bs = np.sqrt(heston.params.v0)
    d1 = (np.log(S / K) + (r + 0.5 * vol_bs**2) * T) / (vol_bs * np.sqrt(T))
    d2 = d1 - vol_bs * np.sqrt(T)
    bs_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

    print(f"Black-Scholes:       ${bs_price:.4f}")
    print(f"Difference:          ${mc_result.price - bs_price:.4f}")
    print(f"Relative Diff:       {(mc_result.price - bs_price) / bs_price:.1%}")

    print("\nHESTON VS BLACK-SCHOLES ANALYSIS:")
    print("-" * 50)
    print("Heston captures stochastic volatility effects")
    print(
        f"Difference of {(mc_result.price - bs_price) / bs_price:.1%} shows volatility smile impact"
    )
    print(f"Negative correlation (ρ={heston.params.rho}) creates leverage effect")
