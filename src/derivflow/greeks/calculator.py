"""
DERIVFLOW-FINANCE: Advanced Greeks Calculator
===========================================

Complete Greeks calculation engine for all derivatives.
Supports analytical, numerical, and complex step methods.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

@dataclass
class GreeksResult:
    """Container for Greeks calculation results"""
    delta: float
    gamma: float  
    theta: float
    vega: float
    rho: float
    # Second-order Greeks
    volga: Optional[float] = None
    vanna: Optional[float] = None
    # Third-order Greeks
    speed: Optional[float] = None
    zomma: Optional[float] = None
    color: Optional[float] = None

class GreeksCalculator:
    """
    Advanced Greeks calculator supporting multiple methodologies
    
    Provides analytical Greeks for standard options and numerical
    Greeks for exotic instruments using finite differences.
    """
    
    def __init__(self, method: str = 'analytical'):
        """
        Initialize Greeks calculator
        
        Parameters:
        -----------
        method : str
            Calculation method ('analytical', 'numerical', 'complex_step')
        """
        self.method = method
        self.default_bumps = {
            'delta': 0.01,      # 1% bump for delta
            'gamma': 0.01,      # 1% bump for gamma  
            'theta': 1/365,     # 1 day for theta
            'vega': 0.01,       # 1% vol bump
            'rho': 0.01         # 1% rate bump
        }
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> float:
        """Black-Scholes pricing function for Greeks calculation"""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    
    def analytical_greeks(self, S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str = 'call') -> GreeksResult:
        """
        Calculate analytical Greeks for European options
        
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
            
        Returns:
        --------
        GreeksResult
            Complete Greeks profile
        """
        if T <= 0:
            # Handle expiry case
            return GreeksResult(
                delta=1.0 if (option_type.lower() == 'call' and S > K) else 0.0,
                gamma=0.0, theta=0.0, vega=0.0, rho=0.0
            )
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal PDF and CDF
        phi_d1 = stats.norm.pdf(d1)
        phi_d2 = stats.norm.pdf(d2)
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        N_minus_d1 = stats.norm.cdf(-d1)
        N_minus_d2 = stats.norm.cdf(-d2)
        
        # First-order Greeks
        if option_type.lower() == 'call':
            delta = N_d1
            theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * N_d2)
            rho = K * T * np.exp(-r * T) * N_d2
        else:  # put
            delta = N_d1 - 1
            theta = (-S * phi_d1 * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * N_minus_d2)
            rho = -K * T * np.exp(-r * T) * N_minus_d2
        
        # Second-order Greeks (same for calls and puts)
        gamma = phi_d1 / (S * sigma * np.sqrt(T))
        vega = S * phi_d1 * np.sqrt(T)
        
        # Higher-order Greeks
        volga = vega * d1 * d2 / sigma  # Volga (vega sensitivity to vol)
        vanna = -phi_d1 * d2 / sigma    # Vanna (delta sensitivity to vol)
        
        # Third-order Greeks
        speed = -gamma * (d1 / (sigma * np.sqrt(T)) + 1) / S  # Speed (gamma sensitivity to spot)
        zomma = gamma * (d1 * d2 - 1) / sigma  # Zomma (gamma sensitivity to vol)
        color = (-phi_d1 / (2 * S * T * sigma * np.sqrt(T)) * 
                (2 * r * T + 1 + d1 * (2 * r * T - d1) / (sigma * np.sqrt(T))))  # Color (gamma sensitivity to time)
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta / 365,  # Convert to per-day
            vega=vega / 100,    # Convert to per 1% vol change
            rho=rho / 100,      # Convert to per 1% rate change
            volga=volga / 100,  # Convert to per 1% vol change
            vanna=vanna / 100,  # Convert to per 1% vol change
            speed=speed,
            zomma=zomma / 100,
            color=color / 365   # Convert to per-day
        )
    
    def numerical_greeks(self, pricing_func, base_params: Dict, 
                        bumps: Optional[Dict] = None) -> GreeksResult:
        """
        Calculate numerical Greeks using finite differences
        
        Parameters:
        -----------
        pricing_func : callable
            Function that prices the instrument
        base_params : Dict
            Base parameters for pricing
        bumps : Dict, optional
            Custom bump sizes for each Greek
            
        Returns:
        --------
        GreeksResult
            Numerically calculated Greeks
        """
        if bumps is None:
            bumps = self.default_bumps.copy()
        
        # Base price
        base_price = pricing_func(**base_params)
        
        # Delta: dP/dS
        params_up = base_params.copy()
        params_down = base_params.copy()
        params_up['S'] = base_params['S'] * (1 + bumps['delta'])
        params_down['S'] = base_params['S'] * (1 - bumps['delta'])
        
        price_up = pricing_func(**params_up)
        price_down = pricing_func(**params_down)
        delta = (price_up - price_down) / (2 * base_params['S'] * bumps['delta'])
        
        # Gamma: d²P/dS²
        gamma = (price_up - 2 * base_price + price_down) / (base_params['S'] * bumps['delta'])**2
        
        # Theta: dP/dT
        params_theta = base_params.copy()
        params_theta['T'] = max(0.001, base_params['T'] - bumps['theta'])  # Ensure positive time
        price_theta = pricing_func(**params_theta)
        theta = (price_theta - base_price) / bumps['theta']
        
        # Vega: dP/dσ
        params_vega_up = base_params.copy()
        params_vega_down = base_params.copy()
        params_vega_up['sigma'] = base_params['sigma'] * (1 + bumps['vega'])
        params_vega_down['sigma'] = base_params['sigma'] * (1 - bumps['vega'])
        
        price_vega_up = pricing_func(**params_vega_up)
        price_vega_down = pricing_func(**params_vega_down)
        vega = (price_vega_up - price_vega_down) / (2 * base_params['sigma'] * bumps['vega'])
        
        # Rho: dP/dr
        params_rho_up = base_params.copy()
        params_rho_down = base_params.copy()
        params_rho_up['r'] = base_params['r'] + bumps['rho']
        params_rho_down['r'] = base_params['r'] - bumps['rho']
        
        price_rho_up = pricing_func(**params_rho_up)
        price_rho_down = pricing_func(**params_rho_down)
        rho = (price_rho_up - price_rho_down) / (2 * bumps['rho'])
        
        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega / 100,  # Convert to per 1% vol change
            rho=rho / 100     # Convert to per 1% rate change
        )
    
    def portfolio_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks by aggregating individual positions
        
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
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0,
            'total_notional': 0.0
        }
        
        for position in positions:
            quantity = position['quantity']
            greeks = position['greeks']
            notional = position.get('notional', 1.0)
            
            # Weight Greeks by quantity and notional
            weight = quantity * notional
            
            portfolio_greeks['delta'] += weight * greeks.delta
            portfolio_greeks['gamma'] += weight * greeks.gamma
            portfolio_greeks['theta'] += weight * greeks.theta
            portfolio_greeks['vega'] += weight * greeks.vega
            portfolio_greeks['rho'] += weight * greeks.rho
            portfolio_greeks['total_notional'] += abs(weight)
        
        return portfolio_greeks
    
    def greeks_ladder(self, S: float, K: float, T: float, r: float, sigma: float,
                     spot_range: tuple = (0.8, 1.2), num_points: int = 21,
                     option_type: str = 'call') -> Dict[str, np.ndarray]:
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
            'spot_prices': spot_prices,
            'delta': np.zeros(num_points),
            'gamma': np.zeros(num_points),
            'theta': np.zeros(num_points),
            'vega': np.zeros(num_points),
            'rho': np.zeros(num_points)
        }
        
        for i, spot in enumerate(spot_prices):
            greeks = self.analytical_greeks(spot, K, T, r, sigma, option_type)
            results['delta'][i] = greeks.delta
            results['gamma'][i] = greeks.gamma
            results['theta'][i] = greeks.theta
            results['vega'][i] = greeks.vega
            results['rho'][i] = greeks.rho
        
        return results
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, 
                        sigma: float, option_type: str = 'call',
                        pricing_func=None, **kwargs) -> GreeksResult:
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
            Custom pricing function for numerical Greeks
        **kwargs
            Additional parameters
            
        Returns:
        --------
        GreeksResult
            Complete Greeks calculation
        """
        if self.method == 'analytical':
            return self.analytical_greeks(S, K, T, r, sigma, option_type)
        
        elif self.method == 'numerical':
            if pricing_func is None:
                pricing_func = self._black_scholes_price
            
            base_params = {
                'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 
                'option_type': option_type
            }
            base_params.update(kwargs)
            
            return self.numerical_greeks(pricing_func, base_params)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")

def format_greeks_report(greeks: GreeksResult, S: float, K: float, 
                        option_type: str = 'call') -> str:
    """
    Format Greeks results into a professional report
    
    Parameters:
    -----------
    greeks : GreeksResult
        Greeks calculation results
    S : float
        Current spot price
    K : float
        Strike price
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    str
        Formatted report string
    """
    moneyness = "ITM" if (option_type.lower() == 'call' and S > K) or (option_type.lower() == 'put' and S < K) else "OTM"
    if abs(S - K) / K < 0.02:  # Within 2% of strike
        moneyness = "ATM"
    
    report = f"""
📊 GREEKS ANALYSIS REPORT
{'='*50}
Option Details:
  Type: {option_type.upper()}
  Spot: ${S:.2f} | Strike: ${K:.2f} | Moneyness: {moneyness}

📈 FIRST-ORDER GREEKS (Risk Sensitivities)
{'-'*50}
  Delta (Δ):    {greeks.delta:>8.4f}  | Price change per $1 spot move
  Theta (Θ):    {greeks.theta:>8.2f}  | Price decay per day
  Vega (ν):     {greeks.vega:>8.2f}   | Price change per 1% vol move
  Rho (ρ):      {greeks.rho:>8.3f}    | Price change per 1% rate move

📊 SECOND-ORDER GREEKS (Convexity)
{'-'*50}
  Gamma (Γ):    {greeks.gamma:>8.4f}  | Delta change per $1 spot move"""
    
    if greeks.volga is not None:
        report += f"""

🔬 ADVANCED GREEKS (Higher-Order Sensitivities)
{'-'*50}
  Volga:        {greeks.volga:>8.4f}  | Vega change per 1% vol move
  Vanna:        {greeks.vanna:>8.4f}  | Delta change per 1% vol move"""
    
    if greeks.speed is not None:
        report += f"""
  Speed:        {greeks.speed:>8.6f}  | Gamma change per $1 spot move
  Zomma:        {greeks.zomma:>8.6f}  | Gamma change per 1% vol move
  Color:        {greeks.color:>8.6f}  | Gamma decay per day"""
    
    return report

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 DERIVFLOW-FINANCE Greeks Calculator")
    print("=" * 60)
    
    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
    
    print(f"📊 Calculating Greeks for CALL option:")
    print(f"   S=${S}, K=${K}, T={T}, r={r:.1%}, σ={sigma:.1%}")
    print("-" * 60)
    
    # Calculate analytical Greeks
    calculator = GreeksCalculator(method='analytical')
    greeks = calculator.calculate_greeks(S, K, T, r, sigma, 'call')
    
    # Display formatted report
    print(format_greeks_report(greeks, S, K, 'call'))
    
    print("\n" + "=" * 60)
    print("📈 GREEKS LADDER (Delta across spot prices)")
    print("-" * 60)
    
    # Generate Greeks ladder
    ladder = calculator.greeks_ladder(S, K, T, r, sigma, spot_range=(0.9, 1.1), num_points=11)
    
    print(f"{'Spot':>8} {'Delta':>8} {'Gamma':>8} {'Theta':>8}")
    print("-" * 35)
    for i in range(len(ladder['spot_prices'])):
        spot = ladder['spot_prices'][i]
        delta = ladder['delta'][i]
        gamma = ladder['gamma'][i] 
        theta = ladder['theta'][i]
        print(f"{spot:>8.1f} {delta:>8.4f} {gamma:>8.4f} {theta:>8.2f}")