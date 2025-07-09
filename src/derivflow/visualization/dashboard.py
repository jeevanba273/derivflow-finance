"""
DERIVFLOW-FINANCE: Advanced Visualization Dashboard
==================================================

Professional-grade interactive dashboards for derivatives analytics.
Supports real-time pricing, Greeks visualization, and volatility surfaces.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from datetime import datetime, timedelta

# Import our modules - handle both relative and absolute imports
try:
    # Try relative imports first (when used as module)
    from ..core.pricing_engine import PricingEngine, price_european_option
    from ..greeks.calculator import GreeksCalculator
    from ..volatility.surface import VolatilitySurface, create_sample_surface
    from ..utils.market_data import MarketDataProvider, get_current_price
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from derivflow.core.pricing_engine import PricingEngine, price_european_option
    from derivflow.greeks.calculator import GreeksCalculator
    from derivflow.volatility.surface import VolatilitySurface, create_sample_surface
    from derivflow.utils.market_data import MarketDataProvider, get_current_price

class DerivativesDashboard:
    """
    Professional derivatives analytics dashboard
    
    Creates interactive visualizations for:
    - Option pricing across parameters
    - Greeks analysis and sensitivity
    - Volatility surface modeling
    - P&L and risk analytics
    """
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize dashboard
        
        Parameters:
        -----------
        theme : str
            Plotly theme ('plotly_dark', 'plotly_white', 'seaborn', etc.)
        """
        self.theme = theme
        self.pricing_engine = PricingEngine()
        self.greeks_calc = GreeksCalculator()
        self.market_data = MarketDataProvider()
        
        # Color scheme for professional look
        self.colors = {
            'call': '#00ff88',      # Bright green
            'put': '#ff4444',       # Bright red
            'profit': '#00cc66',    # Green
            'loss': '#ff3366',      # Red
            'neutral': '#ffaa00',   # Orange
            'background': '#1e1e1e', # Dark background
            'text': '#ffffff',      # White text
            'grid': '#444444'       # Gray grid
        }
    
    def plot_option_payoff(self, K: float, premium: float, option_type: str = 'call',
                          spot_range: Optional[Tuple[float, float]] = None) -> go.Figure:
        """
        Create option payoff diagram
        
        Parameters:
        -----------
        K : float
            Strike price
        premium : float
            Option premium paid
        option_type : str
            'call' or 'put'
        spot_range : Tuple[float, float], optional
            (min_spot, max_spot) for plotting
            
        Returns:
        --------
        go.Figure
            Interactive payoff diagram
        """
        if spot_range is None:
            spot_range = (K * 0.7, K * 1.3)
        
        # Generate spot price range
        spot_prices = np.linspace(spot_range[0], spot_range[1], 100)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            intrinsic_values = np.maximum(spot_prices - K, 0)
            net_payoffs = intrinsic_values - premium
            color = self.colors['call']
            title_text = f"Call Option Payoff (K=${K:.0f}, Premium=${premium:.2f})"
        else:
            intrinsic_values = np.maximum(K - spot_prices, 0)
            net_payoffs = intrinsic_values - premium
            color = self.colors['put']
            title_text = f"Put Option Payoff (K=${K:.0f}, Premium=${premium:.2f})"
        
        # Create figure
        fig = go.Figure()
        
        # Add intrinsic value line
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=intrinsic_values,
            mode='lines',
            name='Intrinsic Value',
            line=dict(color=color, width=2, dash='dash'),
            hovertemplate='Spot: $%{x:.2f}<br>Intrinsic: $%{y:.2f}<extra></extra>'
        ))
        
        # Add net payoff line
        fig.add_trace(go.Scatter(
            x=spot_prices,
            y=net_payoffs,
            mode='lines',
            name='Net P&L',
            line=dict(color=color, width=3),
            fill='tonexty',
            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.3])}',
            hovertemplate='Spot: $%{x:.2f}<br>Net P&L: $%{y:.2f}<extra></extra>'
        ))
        
        # Add breakeven line
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
        
        # Add strike price line
        fig.add_vline(x=K, line_dash="dot", line_color="yellow", opacity=0.7,
                     annotation_text=f"Strike: ${K:.0f}")
        
        # Calculate and show breakeven point
        if option_type.lower() == 'call':
            breakeven = K + premium
        else:
            breakeven = K - premium
        
        fig.add_vline(x=breakeven, line_dash="dot", line_color="orange", opacity=0.7,
                     annotation_text=f"Breakeven: ${breakeven:.0f}")
        
        # Update layout
        fig.update_layout(
            title=title_text,
            xaxis_title="Underlying Price ($)",
            yaxis_title="Profit/Loss ($)",
            template=self.theme,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_price_sensitivity(self, S: float, K: float, T: float, r: float,
                             sigma: float, option_type: str = 'call',
                             param: str = 'spot', param_range: float = 0.3) -> go.Figure:
        """
        Plot option price sensitivity to various parameters
        
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
        param : str
            Parameter to vary ('spot', 'volatility', 'time', 'rate')
        param_range : float
            Range around current value (as percentage)
            
        Returns:
        --------
        go.Figure
            Price sensitivity chart
        """
        # Generate parameter range
        if param == 'spot':
            param_values = np.linspace(S * (1 - param_range), S * (1 + param_range), 50)
            param_label = "Spot Price ($)"
            base_value = S
            
            prices = [self.pricing_engine.price_option('black_scholes', p, K, T, r, sigma, option_type) 
                     for p in param_values]
                     
        elif param == 'volatility':
            param_values = np.linspace(sigma * (1 - param_range), sigma * (1 + param_range), 50)
            param_label = "Volatility"
            base_value = sigma
            
            prices = [self.pricing_engine.price_option('black_scholes', S, K, T, r, v, option_type) 
                     for v in param_values]
                     
        elif param == 'time':
            param_values = np.linspace(max(T * (1 - param_range), 0.01), T * (1 + param_range), 50)
            param_label = "Time to Expiry (years)"
            base_value = T
            
            prices = [self.pricing_engine.price_option('black_scholes', S, K, t, r, sigma, option_type) 
                     for t in param_values]
                     
        elif param == 'rate':
            param_values = np.linspace(r * (1 - param_range), r * (1 + param_range), 50)
            param_label = "Risk-free Rate"
            base_value = r
            
            prices = [self.pricing_engine.price_option('black_scholes', S, K, T, rate, sigma, option_type) 
                     for rate in param_values]
        else:
            raise ValueError("param must be 'spot', 'volatility', 'time', or 'rate'")
        
        # Create figure
        color = self.colors['call'] if option_type == 'call' else self.colors['put']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=param_values,
            y=prices,
            mode='lines',
            name=f'{option_type.title()} Price',
            line=dict(color=color, width=3),
            hovertemplate=f'{param_label}: %{{x:.4f}}<br>Price: $%{{y:.2f}}<extra></extra>'
        ))
        
        # Mark current value
        current_price = self.pricing_engine.price_option('black_scholes', S, K, T, r, sigma, option_type)
        fig.add_trace(go.Scatter(
            x=[base_value],
            y=[current_price],
            mode='markers',
            name='Current',
            marker=dict(color='yellow', size=10, symbol='star'),
            hovertemplate=f'Current {param_label}: %{{x:.4f}}<br>Current Price: $%{{y:.2f}}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{option_type.title()} Option Price Sensitivity to {param.title()}',
            xaxis_title=param_label,
            yaxis_title="Option Price ($)",
            template=self.theme,
            showlegend=True
        )
        
        return fig
    
    def plot_greeks_dashboard(self, S: float, K: float, T: float, r: float,
                            sigma: float, option_type: str = 'call') -> go.Figure:
        """
        Create comprehensive Greeks dashboard
        
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
        go.Figure
            Greeks dashboard with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Greeks Summary'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Generate spot range for Greeks calculation
        spot_range = np.linspace(S * 0.8, S * 1.2, 50)
        
        # Calculate Greeks for each spot price
        deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
        
        for spot in spot_range:
            greeks = self.greeks_calc.calculate_greeks(spot, K, T, r, sigma, option_type)
            deltas.append(greeks.delta)
            gammas.append(greeks.gamma)
            thetas.append(greeks.theta)
            vegas.append(greeks.vega)
            rhos.append(greeks.rho)
        
        # Plot Delta
        fig.add_trace(
            go.Scatter(x=spot_range, y=deltas, name='Delta', 
                      line=dict(color=self.colors['call'], width=2)),
            row=1, col=1
        )
        
        # Plot Gamma
        fig.add_trace(
            go.Scatter(x=spot_range, y=gammas, name='Gamma',
                      line=dict(color=self.colors['put'], width=2)),
            row=1, col=2
        )
        
        # Plot Theta
        fig.add_trace(
            go.Scatter(x=spot_range, y=thetas, name='Theta',
                      line=dict(color=self.colors['neutral'], width=2)),
            row=1, col=3
        )
        
        # Plot Vega
        fig.add_trace(
            go.Scatter(x=spot_range, y=vegas, name='Vega',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # Plot Rho
        fig.add_trace(
            go.Scatter(x=spot_range, y=rhos, name='Rho',
                      line=dict(color='cyan', width=2)),
            row=2, col=2
        )
        
        # Current Greeks summary (bar chart)
        current_greeks = self.greeks_calc.calculate_greeks(S, K, T, r, sigma, option_type)
        greeks_names = ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
        greeks_values = [current_greeks.delta, current_greeks.gamma * 100,  # Scale gamma
                        current_greeks.theta, current_greeks.vega, current_greeks.rho]
        
        fig.add_trace(
            go.Bar(x=greeks_names, y=greeks_values, name='Current Greeks',
                  marker_color=[self.colors['call'], self.colors['put'], 
                               self.colors['neutral'], 'purple', 'cyan']),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f'{option_type.title()} Option Greeks Dashboard (S=${S}, K=${K})',
            template=self.theme,
            height=700,
            showlegend=False
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 4):
                if not (i == 2 and j == 3):  # Skip bar chart
                    fig.update_xaxes(title_text="Spot Price ($)", row=i, col=j)
        
        return fig
    
    def plot_volatility_surface(self, symbol: Optional[str] = None,
                              surface_data: Optional[Dict] = None) -> go.Figure:
        """
        Create 3D volatility surface plot
        
        Parameters:
        -----------
        symbol : str, optional
            Stock symbol for live data
        surface_data : Dict, optional
            Pre-computed surface data
            
        Returns:
        --------
        go.Figure
            3D volatility surface
        """
        if surface_data is None:
            if symbol:
                # Try to get live data
                try:
                    market_surface = self.market_data.build_volatility_surface_from_market(symbol)
                    surface_data = market_surface
                except Exception:
                    # Fallback to sample data
                    warnings.warn("Using sample volatility surface - live data unavailable")
                    vol_surface = create_sample_surface()
                    vol_surface.build_surface()
                    surface_data = {
                        'surface_data': [
                            {'strike': point.strike, 'expiry': point.expiry, 'volatility': point.volatility}
                            for point in vol_surface.data_points
                        ],
                        'spot_price': 100
                    }
            else:
                # Use sample data
                vol_surface = create_sample_surface()
                vol_surface.build_surface()
                surface_data = {
                    'surface_data': [
                        {'strike': point.strike, 'expiry': point.expiry, 'volatility': point.volatility}
                        for point in vol_surface.data_points
                    ],
                    'spot_price': 100
                }
        
        # Extract data
        df = pd.DataFrame(surface_data['surface_data'])
        spot_price = surface_data.get('spot_price', 100)
        
        if df.empty:
            raise ValueError("No volatility surface data available")
        
        # Convert to moneyness
        df['moneyness'] = df['strike'] / spot_price
        
        # Create pivot table for surface
        pivot_table = df.pivot_table(values='volatility', index='expiry', columns='moneyness', fill_value=np.nan)
        
        # Create 3D surface
        fig = go.Figure(data=[go.Surface(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            colorbar=dict(title="Implied Volatility"),
            hovertemplate='Moneyness: %{x:.2f}<br>Expiry: %{y:.2f}y<br>IV: %{z:.1%}<extra></extra>'
        )])
        
        # Update layout
        title_text = f"Volatility Surface - {symbol}" if symbol else "Volatility Surface"
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Time to Expiry (years)",
                zaxis_title="Implied Volatility",
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='gray'),
                yaxis=dict(gridcolor='gray'),
                zaxis=dict(gridcolor='gray')
            ),
            template=self.theme,
            height=600
        )
        
        return fig
    
    def plot_pnl_analysis(self, S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str = 'call',
                         position_size: int = 1, premium_paid: float = None) -> go.Figure:
        """
        Create P&L analysis for option position
        
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
        position_size : int
            Number of contracts (positive for long, negative for short)
        premium_paid : float, optional
            Premium paid per contract
            
        Returns:
        --------
        go.Figure
            P&L analysis chart
        """
        if premium_paid is None:
            premium_paid = self.pricing_engine.price_option('black_scholes', S, K, T, r, sigma, option_type)
        
        # Generate spot range
        spot_range = np.linspace(S * 0.7, S * 1.3, 100)
        
        # Calculate P&L for different scenarios
        scenarios = {
            'At Expiry': 0,
            '1 Week Left': 1/52,
            '1 Month Left': 1/12,
            'Current': T
        }
        
        fig = go.Figure()
        colors = ['red', 'orange', 'yellow', 'green']
        
        for i, (scenario_name, time_left) in enumerate(scenarios.items()):
            pnls = []
            
            for spot in spot_range:
                if time_left == 0:
                    # At expiry - intrinsic value only
                    if option_type == 'call':
                        option_value = max(spot - K, 0)
                    else:
                        option_value = max(K - spot, 0)
                else:
                    # Calculate option value with time left
                    option_value = self.pricing_engine.price_option(
                        'black_scholes', spot, K, time_left, r, sigma, option_type
                    )
                
                # Calculate P&L
                pnl = position_size * (option_value - premium_paid)
                pnls.append(pnl)
            
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=pnls,
                mode='lines',
                name=scenario_name,
                line=dict(color=colors[i], width=2),
                hovertemplate=f'{scenario_name}<br>Spot: $%{{x:.2f}}<br>P&L: $%{{y:.2f}}<extra></extra>'
            ))
        
        # Add breakeven line
        fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
        
        # Add current spot line
        fig.add_vline(x=S, line_dash="dot", line_color="cyan", opacity=0.7,
                     annotation_text=f"Current: ${S:.0f}")
        
        # Update layout
        position_desc = f"Long {abs(position_size)}" if position_size > 0 else f"Short {abs(position_size)}"
        title_text = f"P&L Analysis - {position_desc} {option_type.title()} ${K:.0f} (Premium: ${premium_paid:.2f})"
        
        fig.update_layout(
            title=title_text,
            xaxis_title="Underlying Price ($)",
            yaxis_title="Profit/Loss ($)",
            template=self.theme,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, symbol: str) -> Dict[str, go.Figure]:
        """
        Create comprehensive dashboard for a symbol
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        Dict[str, go.Figure]
            Dictionary of all dashboard figures
        """
        try:
            # Get market data
            market_env = self.market_data.get_market_environment(symbol)
            S = market_env.spot_price
            r = market_env.risk_free_rate
            
            # Use standard parameters for demo
            K = S * 1.05  # 5% OTM
            T = 0.25      # 3 months
            sigma = 0.25  # 25% vol
            
            # Calculate option price
            call_price = price_european_option(S, K, T, r, sigma, 'call')
            put_price = price_european_option(S, K, T, r, sigma, 'put')
            
            # Create all dashboard components
            dashboard = {
                'call_payoff': self.plot_option_payoff(K, call_price, 'call'),
                'put_payoff': self.plot_option_payoff(K, put_price, 'put'),
                'spot_sensitivity': self.plot_price_sensitivity(S, K, T, r, sigma, 'call', 'spot'),
                'vol_sensitivity': self.plot_price_sensitivity(S, K, T, r, sigma, 'call', 'volatility'),
                'greeks_dashboard': self.plot_greeks_dashboard(S, K, T, r, sigma, 'call'),
                'volatility_surface': self.plot_volatility_surface(symbol),
                'call_pnl': self.plot_pnl_analysis(S, K, T, r, sigma, 'call', 1, call_price),
                'put_pnl': self.plot_pnl_analysis(S, K, T, r, sigma, 'put', 1, put_price)
            }
            
            return dashboard
            
        except Exception as e:
            raise ValueError(f"Error creating dashboard for {symbol}: {str(e)}")

# Convenience functions
def quick_payoff_plot(strike: float, premium: float, option_type: str = 'call') -> go.Figure:
    """Quick function to create payoff diagram"""
    dashboard = DerivativesDashboard()
    return dashboard.plot_option_payoff(strike, premium, option_type)

def quick_greeks_plot(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> go.Figure:
    """Quick function to create Greeks dashboard"""
    dashboard = DerivativesDashboard()
    return dashboard.plot_greeks_dashboard(S, K, T, r, sigma, option_type)

# Example usage and testing
if __name__ == "__main__":
    print("🚀 DERIVFLOW-FINANCE Visualization Dashboard")
    print("=" * 60)
    
    # Initialize dashboard
    dashboard = DerivativesDashboard()
    
    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.25
    call_price = price_european_option(S, K, T, r, sigma, 'call')
    
    print(f"📊 Creating visualizations for CALL option:")
    print(f"   Spot: ${S}, Strike: ${K}, Premium: ${call_price:.2f}")
    print("-" * 50)
    
    try:
        # Test 1: Payoff diagram
        print("1. ✅ Payoff diagram: Ready")
        payoff_fig = dashboard.plot_option_payoff(K, call_price, 'call')
        
        # Test 2: Price sensitivity
        print("2. ✅ Price sensitivity: Ready")
        sensitivity_fig = dashboard.plot_price_sensitivity(S, K, T, r, sigma, 'call', 'spot')
        
        # Test 3: Greeks dashboard
        print("3. ✅ Greeks dashboard: Ready")
        greeks_fig = dashboard.plot_greeks_dashboard(S, K, T, r, sigma, 'call')
        
        # Test 4: Volatility surface
        print("4. ✅ Volatility surface: Ready")
        vol_surface_fig = dashboard.plot_volatility_surface()
        
        # Test 5: P&L analysis
        print("5. ✅ P&L analysis: Ready")
        pnl_fig = dashboard.plot_pnl_analysis(S, K, T, r, sigma, 'call', 1, call_price)
        
        print("\n🎉 DASHBOARD STATUS:")
        print("-" * 30)
        print("✅ Payoff diagrams: Working")
        print("✅ Sensitivity analysis: Working")
        print("✅ Greeks visualization: Working")
        print("✅ Volatility surfaces: Working")
        print("✅ P&L analysis: Working")
        print("✅ Interactive plots: Ready")
        
        print(f"\n🚀 VISUALIZATION MODULE COMPLETE!")
        print("📊 Professional derivatives analytics dashboard ready!")
        print("💡 Use .show() on any figure to display in browser")
        
        # Example: Show one plot
        print(f"\n📈 Example: Displaying payoff diagram...")
        # payoff_fig.show()  # Uncomment to display
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("💡 Check dependencies: plotly, pandas, numpy")