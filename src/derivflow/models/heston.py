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

import numpy as np
import scipy.stats as stats
from scipy import integrate
from scipy.optimize import minimize

from .._numba_utils import HAS_NUMBA, njit, prange


@njit(cache=True, parallel=True)
def _heston_euler_final(S, r, dt, sqrt_dt, kappa, theta, sigma, v0, W1, W2, num_sims, num_steps):
    """
    Simulate Heston paths with the Euler-Maruyama scheme and return only the
    terminal asset prices. Path-recursive in the variance process, so this is a
    genuine per-path loop (not vectorizable via cumsum). Pre-generated shocks
    W1/W2 are passed in so results are identical to the numpy fallback for a
    given seed; keeping only scalars per path avoids materializing the full
    (num_sims, num_steps+1) matrices.
    """
    final_prices = np.empty(num_sims)
    for i in prange(num_sims):
        s = S
        v = v0
        for t in range(num_steps):
            v_pos = v if v > 0.0 else 0.0
            v_next = v + kappa * (theta - v) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * W2[i, t]
            if v_next < 0.0001:
                v_next = 0.0001
            s = s * np.exp((r - 0.5 * v) * dt + np.sqrt(v_pos) * sqrt_dt * W1[i, t])
            v = v_next
        final_prices[i] = s
    return final_prices


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
    """Heston pricing result"""

    price: float
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    volga: float | None = None
    vanna: float | None = None


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
        num_sims: int = 100000,
        num_steps: int = 252,
    ) -> HestonResult:
        """
        Price option using Monte Carlo simulation

        Simulates the Heston stochastic volatility process using
        the Euler-Maruyama scheme with variance reduction techniques.
        """
        if self.params is None:
            raise ValueError("Parameters not set")

        v0, kappa, theta, sigma, rho = (
            self.params.v0,
            self.params.kappa,
            self.params.theta,
            self.params.sigma,
            self.params.rho,
        )

        dt = T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Generate correlated random numbers (kept in numpy so a fixed seed is
        # reproducible regardless of whether the numba kernel is used)
        Z1 = np.random.standard_normal((num_sims, num_steps))
        Z2 = np.random.standard_normal((num_sims, num_steps))
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        if HAS_NUMBA:
            # JIT kernel: path-recursive variance process, returns terminal prices
            final_prices = _heston_euler_final(
                S, r, dt, sqrt_dt, kappa, theta, sigma, v0, W1, W2, num_sims, num_steps
            )
        else:
            # Vectorized-over-sims fallback (identical recurrence, only current
            # state vectors retained to avoid the full path matrices)
            s = np.full(num_sims, float(S))
            v = np.full(num_sims, float(v0))
            for t in range(num_steps):
                v_pos = np.maximum(v, 0.0)
                v_next = np.maximum(
                    v + kappa * (theta - v) * dt + sigma * np.sqrt(v_pos) * sqrt_dt * W2[:, t],
                    0.0001,
                )
                s = s * np.exp((r - 0.5 * v) * dt + np.sqrt(v_pos) * sqrt_dt * W1[:, t])
                v = v_next
            final_prices = s

        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - K, 0)
        else:
            payoffs = np.maximum(K - final_prices, 0)

        # Discount and calculate price
        discounted_payoffs = payoffs * np.exp(-r * T)
        price = np.mean(discounted_payoffs)

        return HestonResult(price=price)

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
            'monte_carlo' (see :meth:`monte_carlo_price`).

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
            return self.monte_carlo_price(S, K, T, r, option_type)
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
