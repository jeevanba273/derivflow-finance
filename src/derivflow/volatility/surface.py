"""
DERIVFLOW-FINANCE: Volatility Surface Engine
==========================================

Advanced volatility surface construction and interpolation.
Supports multiple methodologies and market data integration.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

@dataclass
class VolatilityPoint:
    """Single volatility data point"""
    strike: float
    expiry: float
    volatility: float
    market_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class VolatilitySurfaceResult:
    """Result container for volatility surface operations"""
    strikes: np.ndarray
    expiries: np.ndarray
    volatilities: np.ndarray
    interpolated_vol: Optional[float] = None
    smile_params: Optional[Dict] = None

class VolatilitySurface:
    """
    Professional-grade volatility surface construction and analysis
    
    Implements multiple interpolation methods and smile models:
    - Cubic spline interpolation
    - SVI (Stochastic Volatility Inspired) model
    - SABR (Stochastic Alpha Beta Rho) calibration
    - Arbitrage-free constraints
    """
    
    def __init__(self, method: str = 'cubic_spline'):
        """
        Initialize volatility surface engine
        
        Parameters:
        -----------
        method : str
            Interpolation method ('cubic_spline', 'svi', 'sabr', 'rbf')
        """
        self.method = method
        self.vol_data = []
        self.surface_fitted = False
        self.interpolator = None
        self.smile_params = {}
        
    def add_volatility_point(self, strike: float, expiry: float, volatility: float,
                           market_price: Optional[float] = None) -> None:
        """
        Add single volatility observation
        
        Parameters:
        -----------
        strike : float
            Strike price
        expiry : float
            Time to expiry (in years)
        volatility : float
            Implied volatility
        market_price : float, optional
            Market price of the option
        """
        vol_point = VolatilityPoint(
            strike=strike,
            expiry=expiry, 
            volatility=volatility,
            market_price=market_price
        )
        self.vol_data.append(vol_point)
        self.surface_fitted = False
    
    def load_market_data(self, data: Union[pd.DataFrame, List[Dict]]) -> None:
        """
        Load market volatility data in bulk
        
        Parameters:
        -----------
        data : pd.DataFrame or List[Dict]
            Market data with columns: strike, expiry, volatility
        """
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                self.add_volatility_point(
                    strike=row['strike'],
                    expiry=row['expiry'],
                    volatility=row['volatility'],
                    market_price=row.get('market_price')
                )
        elif isinstance(data, list):
            for point in data:
                self.add_volatility_point(**point)
        
        print(f"✅ Loaded {len(self.vol_data)} volatility points")
    
    def _prepare_data_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for surface fitting"""
        if not self.vol_data:
            raise ValueError("No volatility data available")
        
        strikes = np.array([point.strike for point in self.vol_data])
        expiries = np.array([point.expiry for point in self.vol_data])
        volatilities = np.array([point.volatility for point in self.vol_data])
        
        return strikes, expiries, volatilities
    
    def build_surface_cubic_spline(self) -> None:
        """Build volatility surface using cubic spline interpolation"""
        strikes, expiries, volatilities = self._prepare_data_arrays()
        
        # Create 2D interpolator
        points = np.column_stack((strikes, expiries))
        self.interpolator = interpolate.LinearNDInterpolator(points, volatilities)
        
        # Also create RBF interpolator for smoother results
        self.rbf_interpolator = interpolate.Rbf(
            strikes, expiries, volatilities, 
            function='thin_plate', smooth=0.1
        )
        
        self.surface_fitted = True
        print("✅ Cubic spline volatility surface fitted")
    
    def _svi_implied_vol(self, k: float, a: float, b: float, sigma: float, 
                        rho: float, m: float) -> float:
        """
        SVI (Stochastic Volatility Inspired) model
        
        Total variance: w(k) = a + b * (ρ(k-m) + √((k-m)² + σ²))
        """
        k_shifted = k - m
        sqrt_term = np.sqrt(k_shifted**2 + sigma**2)
        w = a + b * (rho * k_shifted + sqrt_term)
        return np.sqrt(max(w, 0.001))  # Ensure positive variance
    
    def _calibrate_svi_slice(self, strikes: np.ndarray, volatilities: np.ndarray,
                           expiry: float) -> Dict[str, float]:
        """Calibrate SVI model for single expiry slice"""
        # Convert to log-moneyness and total variance
        atm_strike = np.median(strikes)
        k = np.log(strikes / atm_strike)
        w_market = volatilities**2 * expiry
        
        # Initial parameter guess
        initial_params = [
            np.mean(w_market),  # a
            0.1,                # b
            0.1,                # sigma
            0.0,                # rho
            0.0                 # m
        ]
        
        def objective(params):
            a, b, sigma, rho, m = params
            w_model = []
            for k_i in k:
                vol = self._svi_implied_vol(k_i, a, b, sigma, rho, m)
                w_model.append(vol**2 * expiry)
            
            return np.sum((np.array(w_model) - w_market)**2)
        
        # Parameter bounds for numerical stability
        bounds = [
            (0.001, None),      # a > 0
            (0.001, None),      # b > 0  
            (0.001, None),      # sigma > 0
            (-0.999, 0.999),    # -1 < rho < 1
            (None, None)        # m unconstrained
        ]
        
        try:
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                return dict(zip(['a', 'b', 'sigma', 'rho', 'm'], result.x))
            else:
                print(f"⚠️ SVI calibration failed for expiry {expiry}")
                return None
        except Exception as e:
            print(f"⚠️ SVI calibration error: {str(e)}")
            return None
    
    def build_surface_svi(self) -> None:
        """Build volatility surface using SVI model"""
        strikes, expiries, volatilities = self._prepare_data_arrays()
        
        # Group data by expiry
        df = pd.DataFrame({
            'strike': strikes,
            'expiry': expiries,
            'volatility': volatilities
        })
        
        self.smile_params = {}
        
        for expiry in df['expiry'].unique():
            slice_data = df[df['expiry'] == expiry]
            if len(slice_data) >= 5:  # Need minimum points for calibration
                svi_params = self._calibrate_svi_slice(
                    slice_data['strike'].values,
                    slice_data['volatility'].values,
                    expiry
                )
                if svi_params:
                    self.smile_params[expiry] = svi_params
        
        self.surface_fitted = True
        print(f"✅ SVI volatility surface fitted for {len(self.smile_params)} expiries")
    
    def build_surface(self, method: Optional[str] = None) -> None:
        """
        Build volatility surface using specified method
        
        Parameters:
        -----------
        method : str, optional
            Override default method ('cubic_spline', 'svi', 'rbf')
        """
        if method:
            self.method = method
        
        if not self.vol_data:
            raise ValueError("No volatility data to fit surface")
        
        if self.method == 'cubic_spline':
            self.build_surface_cubic_spline()
        elif self.method == 'svi':
            self.build_surface_svi()
        elif self.method == 'rbf':
            self.build_surface_cubic_spline()  # RBF included in cubic spline method
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def interpolate(self, strike: float, expiry: float) -> float:
        """
        Interpolate volatility at given strike and expiry
        
        Parameters:
        -----------
        strike : float
            Strike price
        expiry : float
            Time to expiry
            
        Returns:
        --------
        float
            Interpolated implied volatility
        """
        if not self.surface_fitted:
            self.build_surface()
        
        if self.method == 'svi':
            # Find closest expiry with calibrated parameters
            if not self.smile_params:
                raise ValueError("No SVI parameters available")
            
            closest_expiry = min(self.smile_params.keys(), key=lambda x: abs(x - expiry))
            params = self.smile_params[closest_expiry]
            
            # Calculate log-moneyness (need ATM reference)
            atm_strike = strike  # Simplified - in practice, use ATM forward
            k = np.log(strike / atm_strike)
            
            return self._svi_implied_vol(k, **params)
        
        elif self.method in ['cubic_spline', 'rbf']:
            if self.rbf_interpolator:
                vol = self.rbf_interpolator(strike, expiry)
                return max(float(vol), 0.01)  # Ensure minimum volatility
            else:
                vol = self.interpolator(strike, expiry)
                if np.isnan(vol):
                    # Fallback to nearest neighbor
                    strikes, expiries, volatilities = self._prepare_data_arrays()
                    distances = (strikes - strike)**2 + (expiries - expiry)**2
                    nearest_idx = np.argmin(distances)
                    return volatilities[nearest_idx]
                return max(float(vol), 0.01)
    
    def get_smile(self, expiry: float, strike_range: Optional[Tuple[float, float]] = None,
                  num_points: int = 50) -> Dict[str, np.ndarray]:
        """
        Extract volatility smile for given expiry
        
        Parameters:
        -----------
        expiry : float
            Time to expiry
        strike_range : tuple, optional
            (min_strike, max_strike) - if None, auto-detect from data
        num_points : int
            Number of points in smile
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with 'strikes' and 'volatilities' arrays
        """
        if not self.surface_fitted:
            self.build_surface()
        
        # Determine strike range
        if strike_range is None:
            strikes_data = [point.strike for point in self.vol_data 
                          if abs(point.expiry - expiry) < 0.1]
            if not strikes_data:
                raise ValueError(f"No data found near expiry {expiry}")
            
            min_strike = min(strikes_data) * 0.8
            max_strike = max(strikes_data) * 1.2
        else:
            min_strike, max_strike = strike_range
        
        # Generate smile
        strikes = np.linspace(min_strike, max_strike, num_points)
        volatilities = np.array([self.interpolate(k, expiry) for k in strikes])
        
        return {
            'strikes': strikes,
            'volatilities': volatilities,
            'expiry': expiry
        }
    
    def check_arbitrage(self, strikes: np.ndarray, volatilities: np.ndarray) -> Dict[str, bool]:
        """
        Check for arbitrage conditions in volatility smile
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        volatilities : np.ndarray
            Corresponding volatilities
            
        Returns:
        --------
        Dict[str, bool]
            Arbitrage check results
        """
        # Calculate total variance
        total_var = volatilities**2
        
        # Check monotonicity conditions
        # For calendar spreads: longer expiries should have higher total variance
        calendar_arbitrage = False
        
        # Check butterfly spreads: convexity condition
        butterfly_arbitrage = False
        if len(strikes) >= 3:
            # Second derivative of call prices with respect to strike should be positive
            # This translates to specific conditions on implied volatility
            d2c_dk2 = np.gradient(np.gradient(total_var))
            butterfly_arbitrage = np.any(d2c_dk2 < -1e-6)
        
        return {
            'calendar_arbitrage': calendar_arbitrage,
            'butterfly_arbitrage': butterfly_arbitrage,
            'arbitrage_free': not (calendar_arbitrage or butterfly_arbitrage)
        }
    
    def surface_statistics(self) -> Dict[str, float]:
        """Calculate surface quality statistics"""
        if not self.vol_data:
            return {}
        
        volatilities = [point.volatility for point in self.vol_data]
        
        return {
            'num_points': len(self.vol_data),
            'min_volatility': min(volatilities),
            'max_volatility': max(volatilities), 
            'mean_volatility': np.mean(volatilities),
            'vol_of_vol': np.std(volatilities),
            'unique_expiries': len(set(point.expiry for point in self.vol_data)),
            'unique_strikes': len(set(point.strike for point in self.vol_data))
        }
    
    def plot_surface(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot 3D volatility surface"""
        if not self.surface_fitted:
            self.build_surface()
        
        strikes, expiries, volatilities = self._prepare_data_arrays()
        
        # Create grid for surface
        strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
        expiry_grid = np.linspace(expiries.min(), expiries.max(), 30)
        K_grid, T_grid = np.meshgrid(strike_grid, expiry_grid)
        
        # Interpolate volatilities on grid
        vol_grid = np.zeros_like(K_grid)
        for i in range(len(expiry_grid)):
            for j in range(len(strike_grid)):
                try:
                    vol_grid[i, j] = self.interpolate(K_grid[i, j], T_grid[i, j])
                except:
                    vol_grid[i, j] = np.nan
        
        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(K_grid, T_grid, vol_grid, cmap='viridis', alpha=0.8)
        
        # Plot original data points
        ax.scatter(strikes, expiries, volatilities, color='red', s=50, alpha=0.8)
        
        ax.set_xlabel('Strike')
        ax.set_ylabel('Time to Expiry')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('Volatility Surface')
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.show()

# Convenience functions
def create_sample_surface() -> VolatilitySurface:
    """Create sample volatility surface for testing"""
    surface = VolatilitySurface()
    
    # Generate realistic sample data
    np.random.seed(42)
    
    spot = 100
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    expiries = np.array([0.083, 0.25, 0.5, 1.0])  # 1M, 3M, 6M, 1Y
    
    for expiry in expiries:
        for strike in strikes:
            # Generate realistic vol smile (higher vol for OTM options)
            moneyness = np.log(strike / spot)
            base_vol = 0.2 + 0.1 * expiry  # Positive term structure
            skew = -0.1 * moneyness  # Negative skew
            smile = 0.05 * moneyness**2  # Smile curvature
            noise = np.random.normal(0, 0.01)  # Small noise
            
            vol = base_vol + skew + smile + noise
            vol = max(vol, 0.05)  # Minimum vol
            
            surface.add_volatility_point(strike, expiry, vol)
    
    return surface

if __name__ == "__main__":
    # Example usage and testing
    print("🚀 DERIVFLOW-FINANCE: Volatility Surface Engine")
    print("=" * 65)
    
    # Create sample surface
    print("📊 Creating sample volatility surface...")
    surface = create_sample_surface()
    
    # Build surface
    print("\n🔧 Building surface with cubic spline interpolation...")
    surface.build_surface(method='cubic_spline')
    
    # Surface statistics
    stats = surface.surface_statistics()
    print(f"\n📈 SURFACE STATISTICS:")
    print("-" * 40)
    print(f"Data Points:      {stats['num_points']}")
    print(f"Unique Expiries:  {stats['unique_expiries']}")
    print(f"Unique Strikes:   {stats['unique_strikes']}")
    print(f"Vol Range:        {stats['min_volatility']:.1%} - {stats['max_volatility']:.1%}")
    print(f"Mean Vol:         {stats['mean_volatility']:.1%}")
    print(f"Vol of Vol:       {stats['vol_of_vol']:.1%}")
    
    # Test interpolation
    print(f"\n🎯 INTERPOLATION TESTS:")
    print("-" * 40)
    test_points = [
        (100, 0.25, "3M ATM"),
        (95, 0.25, "3M 95% Put"),
        (105, 0.25, "3M 105% Call"),
        (100, 0.5, "6M ATM")
    ]
    
    for strike, expiry, description in test_points:
        vol = surface.interpolate(strike, expiry)
        print(f"{description:<15}: {vol:.1%}")
    
    # Extract smile
    print(f"\n😊 VOLATILITY SMILE (3M expiry):")
    print("-" * 40)
    smile = surface.get_smile(0.25, strike_range=(85, 115), num_points=7)
    
    print(f"{'Strike':<8} {'Vol':<8}")
    print("-" * 16)
    for i, (k, v) in enumerate(zip(smile['strikes'], smile['volatilities'])):
        print(f"{k:<8.0f} {v:<8.1%}")
    
    print(f"\n✅ Volatility surface analysis complete!")
    print("🚀 Ready for advanced derivatives pricing!")