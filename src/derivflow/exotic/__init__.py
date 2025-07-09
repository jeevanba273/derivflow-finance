"""
DERIVFLOW-FINANCE: Exotic Options Module
=======================================

Comprehensive exotic derivatives pricing capabilities:
- Barrier Options: All knock-in/knock-out variants
- Asian Options: Arithmetic and geometric averaging
- Professional-grade analytics and risk measures
"""

# Barrier Options
from .barrier_options import (
    BarrierOptions,
    BarrierType,
    BarrierOptionResult
)

# Asian Options
from .asian_options import (
    AsianOptions,
    AsianType,
    AverageType,
    AsianOptionResult,
    price_asian_option,
    compare_asian_types
)

# Convenience exports
__all__ = [
    # Barrier Options
    'BarrierOptions',
    'BarrierType', 
    'BarrierOptionResult',
    
    # Asian Options
    'AsianOptions',
    'AsianType',
    'AverageType', 
    'AsianOptionResult',
    'price_asian_option',
    'compare_asian_types'
]

# Module-level convenience functions
def price_barrier_option(S: float, K: float, H: float, T: float, r: float, 
                        sigma: float, barrier_type: str, option_type: str = 'call') -> float:
    """
    Quick function to price barrier option
    
    Parameters:
    -----------
    S : float
        Current spot price
    K : float
        Strike price
    H : float
        Barrier level
    T : float
        Time to expiry
    r : float
        Risk-free rate
    sigma : float
        Volatility
    barrier_type : str
        'up_and_out', 'up_and_in', 'down_and_out', 'down_and_in'
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float
        Option price
    """
    barrier_engine = BarrierOptions()
    result = barrier_engine.price(S, K, H, T, r, sigma, barrier_type, option_type)
    return result.price

def exotic_options_demo() -> None:
    """
    Comprehensive demonstration of exotic options capabilities
    """
    print("🚀 DERIVFLOW-FINANCE: Exotic Options Showcase")
    print("=" * 70)
    
    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.3
    
    print(f"📊 Market Parameters:")
    print(f"   Spot: ${S} | Strike: ${K} | Time: {T}y | Rate: {r:.1%} | Vol: {sigma:.1%}")
    print("-" * 70)
    
    # 1. Barrier Options Demo
    print("1. 🚧 BARRIER OPTIONS")
    print("-" * 40)
    
    barrier_engine = BarrierOptions()
    
    # Down-and-out call
    barrier_result = barrier_engine.price(
        S=S, K=K, H=95, T=T, r=r, sigma=sigma,
        barrier_type='down_and_out', option_type='call'
    )
    
    print(f"Down-and-Out Call:")
    print(f"  Price: ${barrier_result.price:.4f}")
    print(f"  Survival Probability: {barrier_result.probability_survival:.1%}")
    
    # Up-and-out call
    barrier_result2 = barrier_engine.price(
        S=S, K=K, H=115, T=T, r=r, sigma=sigma,
        barrier_type='up_and_out', option_type='call'
    )
    
    print(f"Up-and-Out Call:")
    print(f"  Price: ${barrier_result2.price:.4f}")
    print(f"  Survival Probability: {barrier_result2.probability_survival:.1%}")
    
    # 2. Asian Options Demo
    print(f"\n2. 📊 ASIAN OPTIONS")
    print("-" * 40)
    
    asian_engine = AsianOptions(num_sims=50000)
    
    # Arithmetic Asian call
    arith_result = asian_engine.price(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type='call', asian_type='arithmetic'
    )
    
    print(f"Arithmetic Asian Call:")
    print(f"  Price: ${arith_result.price:.4f} ± {arith_result.std_error:.4f}")
    
    # Geometric Asian call
    geom_result = asian_engine.price(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type='call', asian_type='geometric'
    )
    
    print(f"Geometric Asian Call:")
    print(f"  Price: ${geom_result.price:.4f}")
    
    # Comparison
    diff = arith_result.price - geom_result.price
    print(f"Price Difference: ${diff:.4f} ({diff/geom_result.price:.1%})")
    
    # 3. Summary
    print(f"\n🎉 EXOTIC OPTIONS SUMMARY:")
    print("-" * 40)
    print("✅ Barrier Options: All variants implemented")
    print("✅ Asian Options: Arithmetic & geometric averaging")
    print("✅ Analytical & Monte Carlo pricing")
    print("✅ Survival probabilities & error estimates")
    print("✅ Professional-grade exotic derivatives platform")
    
    print(f"\n🚀 Exotic options module complete!")

if __name__ == "__main__":
    exotic_options_demo()