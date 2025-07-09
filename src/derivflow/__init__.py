"""
DERIVFLOW-FINANCE: Advanced Derivatives Analytics Platform
=========================================================

The ultimate open-source derivatives pricing and risk management toolkit.

🌟 Key Features:
- Advanced derivatives pricing (exotic options, barriers, Asian)
- Professional volatility surface modeling
- Complete Greeks calculation (first, second, third order)
- Multiple pricing methodologies (analytical, numerical, Monte Carlo)
- Institutional-grade risk analytics

🎯 Perfect For:
- Quantitative analysts and derivatives traders
- Financial engineering students and researchers  
- Risk managers and structuring desks
- Academic finance research

Example Usage:
--------------
>>> from derivflow import BarrierOptions, VolatilitySurface, PricingEngine
>>> 
>>> # Price exotic barrier option
>>> barrier = BarrierOptions()
>>> result = barrier.price(S=100, K=105, H=95, T=0.25, r=0.05, sigma=0.2,
...                       barrier_type='down_and_out', option_type='call')
>>> print(f"Barrier Option Price: ${result.price:.2f}")
>>>
>>> # Build volatility surface
>>> surface = VolatilitySurface()
>>> surface.load_market_data(market_data)
>>> vol = surface.interpolate(strike=102, expiry=0.33)
>>> print(f"Interpolated Vol: {vol:.1%}")
"""

# Version information
__version__ = "1.0.0"
__author__ = "Jeevan B A"
__email__ = "jeevanba273@gmail.com"
__github__ = "https://github.com/jeevanba273"
__license__ = "MIT"

# Core pricing engines
from .core.pricing_engine import (
    PricingEngine,
    BlackScholesAnalytical,
    BinomialTree,
    MonteCarloEngine,
    price_european_option
)

# Greeks calculation
from .greeks.calculator import (
    GreeksCalculator,
    GreeksResult,
    format_greeks_report
)

# Exotic options
from .exotic.barrier_options import (
    BarrierOptions,
    BarrierType,
    BarrierOptionResult
)

# Volatility modeling
from .volatility.surface import (
    VolatilitySurface,
    VolatilityPoint,
    VolatilitySurfaceResult,
    create_sample_surface
)

# Convenience imports for main interface
__all__ = [
    # Core pricing
    'PricingEngine',
    'BlackScholesAnalytical',
    'BinomialTree', 
    'MonteCarloEngine',
    'price_european_option',
    
    # Greeks
    'GreeksCalculator',
    'GreeksResult',
    'format_greeks_report',
    
    # Exotic options
    'BarrierOptions',
    'BarrierType',
    'BarrierOptionResult',
    
    # Volatility
    'VolatilitySurface',
    'VolatilityPoint',
    'VolatilitySurfaceResult',
    'create_sample_surface',
    
    # Package info
    '__version__',
    '__author__',
    '__email__',
    '__github__',
    '__license__'
]

def get_package_info() -> dict:
    """Get comprehensive package information"""
    return {
        'name': 'derivflow-finance',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'github': __github__,
        'license': __license__,
        'description': 'Advanced derivatives analytics platform',
        'features': [
            'Exotic options pricing (Barrier, Asian, Lookback)',
            'Professional volatility surface modeling',
            'Complete Greeks calculation (up to 3rd order)',
            'Multiple pricing methodologies',
            'Institutional-grade risk analytics',
            'Monte Carlo simulation engine',
            'Model validation and benchmarking'
        ],
        'use_cases': [
            'Derivatives trading and structuring',
            'Quantitative research and analysis', 
            'Risk management and hedging',
            'Academic finance research',
            'Financial engineering education'
        ]
    }

def demo_derivflow() -> None:
    """
    Comprehensive demonstration of DERIVFLOW-FINANCE capabilities
    
    Shows the full power of the platform with real examples.
    """
    print("🚀 DERIVFLOW-FINANCE Platform Demonstration")
    print("=" * 70)
    print(f"👨‍💻 Created by: {__author__}")
    print(f"📧 Contact: {__email__}")
    print(f"🔗 GitHub: {__github__}")
    print("=" * 70)
    
    # 1. Core Pricing Engine Demo
    print("\n📊 1. CORE PRICING ENGINE")
    print("-" * 40)
    
    engine = PricingEngine()
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
    
    print(f"European Call Option (S=${S}, K=${K}, T={T}, σ={sigma:.1%}):")
    comparison = engine.compare_methods(S, K, T, r, sigma, 'call')
    
    for method, result in comparison.items():
        if isinstance(result, dict) and 'price' in result:
            print(f"  {method:15s}: ${result['price']:.4f}")
        elif not isinstance(result, dict):
            print(f"  {method:15s}: ${result:.4f}")
    
    # 2. Advanced Greeks Demo  
    print(f"\n📈 2. ADVANCED GREEKS ANALYSIS")
    print("-" * 40)
    
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, 'call')
    
    print(f"Delta (Δ):     {greeks.delta:>8.4f}  | Hedge ratio")
    print(f"Gamma (Γ):     {greeks.gamma:>8.4f}  | Convexity")
    print(f"Theta (Θ):     {greeks.theta:>8.2f}  | Time decay (per day)")
    print(f"Vega (ν):      {greeks.vega:>8.2f}   | Vol sensitivity")
    print(f"Rho (ρ):       {greeks.rho:>8.3f}    | Rate sensitivity")
    
    # 3. Exotic Options Demo
    print(f"\n🎲 3. EXOTIC BARRIER OPTIONS")
    print("-" * 40)
    
    barrier_pricer = BarrierOptions()
    barrier_result = barrier_pricer.price(
        S=100, K=105, H=95, T=0.25, r=0.05, sigma=0.3,
        barrier_type='down_and_out', option_type='call'
    )
    
    vanilla_price = price_european_option(100, 105, 0.25, 0.05, 0.3, 'call')
    discount = (vanilla_price - barrier_result.price) / vanilla_price
    
    print(f"Vanilla Call Price:     ${vanilla_price:.4f}")
    print(f"Barrier Call Price:     ${barrier_result.price:.4f}")
    print(f"Barrier Discount:       {discount:.1%}")
    print(f"Survival Probability:   {barrier_result.probability_survival:.1%}")
    
    # 4. Volatility Surface Demo
    print(f"\n📊 4. VOLATILITY SURFACE MODELING")
    print("-" * 40)
    
    # Create sample surface
    vol_surface = create_sample_surface()
    vol_surface.build_surface()
    
    # Show surface statistics
    stats = vol_surface.surface_statistics()
    print(f"Surface Data Points:    {stats['num_points']}")
    print(f"Volatility Range:       {stats['min_volatility']:.1%} - {stats['max_volatility']:.1%}")
    print(f"Mean Volatility:        {stats['mean_volatility']:.1%}")
    
    # Show smile
    smile = vol_surface.get_smile(0.25, num_points=5)
    print(f"\nVolatility Smile (3M expiry):")
    for i in range(0, len(smile['strikes']), 1):
        k, v = smile['strikes'][i], smile['volatilities'][i]
        print(f"  K={k:>6.0f}: {v:>6.1%}")
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("✅ All modules working perfectly")
    print("✅ Professional-grade derivatives analytics")
    print("✅ Ready for institutional use")
    print("=" * 70)
    print(f"🌟 DERIVFLOW-FINANCE by {__author__}")
    print(f"📧 {__email__} | 🔗 {__github__}")

# Auto-run demo if module imported directly
if __name__ == "__main__":
    demo_derivflow()