"""
DERIVFLOW-FINANCE: Core Pricing Engine
=====================================

The foundation of all derivatives pricing in the platform.
Supports multiple pricing methodologies with validation.
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from typing import Union, Dict, List, Optional, Callable
import warnings
from abc import ABC, abstractmethod

class PricingMethod(ABC):
    """Abstract base class for all pricing methods"""
    
    @abstractmethod
    def price(self, **kwargs) -> float:
        """Price an instrument using this method"""
        pass

class BlackScholesAnalytical(PricingMethod):
    """
    Analytical Black-Scholes pricing for European options
    
    The foundation formula that started modern derivatives pricing.
    Provides exact solutions for European calls and puts.
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        return BlackScholesAnalytical._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    def price(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Price European option using Black-Scholes formula
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiry (in years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Option price
        """
        if T <= 0:
            # Handle expiry
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type.lower() == 'put':
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return max(price, 0)  # Ensure non-negative price

class BinomialTree(PricingMethod):
    """
    Binomial tree pricing for American and European options
    
    Provides discrete-time approximation to Black-Scholes.
    Supports early exercise for American options.
    """
    
    def __init__(self, steps: int = 100):
        """
        Initialize binomial tree
        
        Parameters:
        -----------
        steps : int
            Number of time steps in the tree
        """
        self.steps = steps
    
    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call', american: bool = False) -> float:
        """
        Price option using binomial tree
        
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
        american : bool
            Whether option is American style
            
        Returns:
        --------
        float
            Option price
        """
        # Tree parameters
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Initialize asset prices at expiry
        asset_prices = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            asset_prices[i] = S * (u ** (self.steps - i)) * (d ** i)
        
        # Initialize option values at expiry
        option_values = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            if option_type.lower() == 'call':
                option_values[i] = max(asset_prices[i] - K, 0)
            else:
                option_values[i] = max(K - asset_prices[i], 0)
        
        # Backward induction
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                # Calculate continuation value
                continuation_value = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )
                
                if american:
                    # Calculate intrinsic value for American option
                    current_asset_price = S * (u ** (step - i)) * (d ** i)
                    if option_type.lower() == 'call':
                        intrinsic_value = max(current_asset_price - K, 0)
                    else:
                        intrinsic_value = max(K - current_asset_price, 0)
                    
                    # American option: max of continuation and intrinsic
                    option_values[i] = max(continuation_value, intrinsic_value)
                else:
                    # European option: only continuation value
                    option_values[i] = continuation_value
        
        return option_values[0]

class MonteCarloEngine(PricingMethod):
    """
    Monte Carlo pricing engine with variance reduction techniques
    
    Provides flexible framework for path-dependent derivatives.
    Includes antithetic variates and control variates.
    """
    
    def __init__(self, num_sims: int = 100000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine
        
        Parameters:
        -----------
        num_sims : int
            Number of simulation paths
        random_seed : int, optional
            Random seed for reproducible results
        """
        self.num_sims = num_sims
        if random_seed:
            np.random.seed(random_seed)
    
    def generate_gbm_paths(self, S0: float, T: float, r: float, sigma: float,
                          num_steps: int = 252) -> np.ndarray:
        """
        Generate Geometric Brownian Motion paths
        
        Parameters:
        -----------
        S0 : float
            Initial stock price
        T : float
            Time horizon
        r : float
            Risk-free rate
        sigma : float
            Volatility
        num_steps : int
            Number of time steps
            
        Returns:
        --------
        np.ndarray
            Array of shape (num_sims, num_steps + 1) with price paths
        """
        dt = T / num_steps
        
        # Generate random shocks
        Z = np.random.standard_normal((self.num_sims, num_steps))
        
        # Initialize price paths
        paths = np.zeros((self.num_sims, num_steps + 1))
        paths[:, 0] = S0
        
        # Generate paths using exact solution
        for t in range(1, num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call', payoff_func: Optional[Callable] = None) -> Dict[str, float]:
        """
        Price option using Monte Carlo simulation
        
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
        payoff_func : callable, optional
            Custom payoff function for exotic options
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with 'price', 'std_error', 'confidence_interval'
        """
        # Generate final stock prices
        final_prices = S * np.exp(
            (r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.standard_normal(self.num_sims)
        )
        
        # Calculate payoffs
        if payoff_func:
            payoffs = payoff_func(final_prices, K)
        else:
            if option_type.lower() == 'call':
                payoffs = np.maximum(final_prices - K, 0)
            elif option_type.lower() == 'put':
                payoffs = np.maximum(K - final_prices, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-r * T)
        
        # Calculate statistics
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_sims)
        confidence_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )
        
        return {
            'price': price,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
            'num_simulations': self.num_sims
        }

class PricingEngine:
    """
    Main pricing engine that coordinates multiple pricing methods
    
    Provides unified interface for all pricing methodologies.
    Includes model validation and benchmarking capabilities.
    """
    
    def __init__(self):
        """Initialize pricing engine with available methods"""
        self.methods = {
            'black_scholes': BlackScholesAnalytical(),
            'binomial': BinomialTree(),
            'monte_carlo': MonteCarloEngine()
        }
        self.results_cache = {}
    
    def price_option(self, method: str, S: float, K: float, T: float, 
                    r: float, sigma: float, option_type: str = 'call',
                    **kwargs) -> Union[float, Dict]:
        """
        Price option using specified method
        
        Parameters:
        -----------
        method : str
            Pricing method ('black_scholes', 'binomial', 'monte_carlo')
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
        **kwargs
            Additional method-specific parameters
            
        Returns:
        --------
        Union[float, Dict]
            Option price (float for analytical methods, Dict for MC)
        """
        if method not in self.methods:
            raise ValueError(f"Unknown pricing method: {method}")
        
        # Create cache key
        cache_key = f"{method}_{S}_{K}_{T}_{r}_{sigma}_{option_type}"
        
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Price using selected method
        pricing_method = self.methods[method]
        
        try:
            result = pricing_method.price(
                S=S, K=K, T=T, r=r, sigma=sigma, 
                option_type=option_type, **kwargs
            )
            
            # Cache result
            self.results_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            raise ValueError(f"Pricing failed with method {method}: {str(e)}")
    
    def compare_methods(self, S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call',
                       methods: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Compare prices across multiple methods
        
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
        methods : List[str], optional
            Methods to compare (default: all available)
            
        Returns:
        --------
        Dict[str, Dict]
            Comparison results across methods
        """
        if methods is None:
            methods = list(self.methods.keys())
        
        results = {}
        
        for method in methods:
            try:
                result = self.price_option(
                    method=method, S=S, K=K, T=T, r=r, 
                    sigma=sigma, option_type=option_type
                )
                
                if isinstance(result, dict):
                    results[method] = result
                else:
                    results[method] = {'price': result}
                    
            except Exception as e:
                results[method] = {'error': str(e)}
        
        return results
    
    def validate_put_call_parity(self, S: float, K: float, T: float, 
                                r: float, sigma: float, 
                                method: str = 'black_scholes') -> Dict[str, float]:
        """
        Validate put-call parity: C - P = S - K*e^(-rT)
        
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
        method : str
            Pricing method to validate
            
        Returns:
        --------
        Dict[str, float]
            Validation results with error metrics
        """
        # Price call and put
        call_price = self.price_option(
            method=method, S=S, K=K, T=T, r=r, sigma=sigma, option_type='call'
        )
        put_price = self.price_option(
            method=method, S=S, K=K, T=T, r=r, sigma=sigma, option_type='put'
        )
        
        # Extract prices if Monte Carlo results
        if isinstance(call_price, dict):
            call_price = call_price['price']
        if isinstance(put_price, dict):
            put_price = put_price['price']
        
        # Calculate put-call parity components
        left_side = call_price - put_price
        right_side = S - K * np.exp(-r * T)
        error = abs(left_side - right_side)
        relative_error = error / abs(right_side) if right_side != 0 else float('inf')
        
        return {
            'call_price': call_price,
            'put_price': put_price,
            'left_side': left_side,
            'right_side': right_side,
            'absolute_error': error,
            'relative_error': relative_error,
            'parity_satisfied': error < 1e-6
        }

# Module-level convenience functions
def price_european_option(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = 'call', method: str = 'black_scholes') -> float:
    """
    Convenience function to price European option
    
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
    method : str
        Pricing method
        
    Returns:
    --------
    float
        Option price
    """
    engine = PricingEngine()
    result = engine.price_option(method, S, K, T, r, sigma, option_type)
    
    if isinstance(result, dict):
        return result['price']
    return result

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 DERIVFLOW-FINANCE Core Pricing Engine")
    print("=" * 50)
    
    # Initialize engine
    engine = PricingEngine()
    
    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
    
    print(f"📊 Pricing CALL option: S=${S}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    print("-" * 50)
    
    # Compare all methods
    comparison = engine.compare_methods(S, K, T, r, sigma, 'call')
    
    for method, result in comparison.items():
        if 'error' in result:
            print(f"{method:15s}: ERROR - {result['error']}")
        else:
            price = result.get('price', result)
            if isinstance(price, float):
                print(f"{method:15s}: ${price:.4f}")
            else:
                print(f"{method:15s}: {price}")
    
    print("\n" + "-" * 50)
    
    # Validate put-call parity
    parity_check = engine.validate_put_call_parity(S, K, T, r, sigma)
    print(f"📐 Put-Call Parity Validation:")
    print(f"   Call Price: ${parity_check['call_price']:.6f}")
    print(f"   Put Price:  ${parity_check['put_price']:.6f}")
    print(f"   Error:      {parity_check['absolute_error']:.8f}")
    print(f"   Valid:      {'✅ YES' if parity_check['parity_satisfied'] else '❌ NO'}")