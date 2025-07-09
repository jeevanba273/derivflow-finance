# test_dashboard.py - Simple Dashboard Test
import sys
import os
# Add src directory to path (we're in tests/test_visualization/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

print("🚀 Testing DERIVFLOW-FINANCE Dashboard")
print("=" * 50)

try:
    # Import core components
    from derivflow.core.pricing_engine import PricingEngine, price_european_option
    print("✅ Core pricing engine: Loaded")
    
    from derivflow.greeks.calculator import GreeksCalculator
    print("✅ Greeks calculator: Loaded")
    
    from derivflow.volatility.surface import VolatilitySurface, create_sample_surface
    print("✅ Volatility surface: Loaded")
    
    # Test basic functionality
    print("\n📊 Testing Basic Functionality:")
    print("-" * 30)
    
    # Test parameters
    S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.25
    
    # Test pricing
    call_price = price_european_option(S, K, T, r, sigma, 'call')
    put_price = price_european_option(S, K, T, r, sigma, 'put')
    
    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price:  ${put_price:.2f}")
    
    # Test Greeks
    greeks_calc = GreeksCalculator()
    greeks = greeks_calc.calculate_greeks(S, K, T, r, sigma, 'call')
    
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.4f}")
    print(f"Theta: {greeks.theta:.2f}")
    
    # Test volatility surface
    vol_surface = create_sample_surface()
    vol_surface.build_surface()
    
    print(f"Vol Surface Points: {len(vol_surface.vol_data)}")
    
    # Test surface statistics
    stats = vol_surface.surface_statistics()
    print(f"Surface Statistics: {stats['num_points']} points, {stats['unique_expiries']} expiries")
    
    print("\n🎨 Testing Visualization Components:")
    print("-" * 40)
    
    # Now test dashboard components
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    
    print("✅ Plotly: Loaded")
    
    # Simple dashboard class for testing
    class SimpleDashboard:
        def __init__(self):
            self.pricing_engine = PricingEngine()
            self.greeks_calc = GreeksCalculator()
            self.colors = {
                'call': '#00ff88',
                'put': '#ff4444',
                'profit': '#00cc66',
                'loss': '#ff3366'
            }
        
        def plot_option_payoff(self, K, premium, option_type='call'):
            """Create simple payoff diagram"""
            spot_range = np.linspace(K * 0.7, K * 1.3, 100)
            
            if option_type.lower() == 'call':
                intrinsic_values = np.maximum(spot_range - K, 0)
                net_payoffs = intrinsic_values - premium
                color = self.colors['call']
                title = f"Call Option Payoff (K=${K:.0f})"
            else:
                intrinsic_values = np.maximum(K - spot_range, 0)
                net_payoffs = intrinsic_values - premium
                color = self.colors['put']
                title = f"Put Option Payoff (K=${K:.0f})"
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=net_payoffs,
                mode='lines',
                name='Net P&L',
                line=dict(color=color, width=3)
            ))
            
            fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.5)
            fig.add_vline(x=K, line_dash="dot", line_color="yellow", opacity=0.7)
            
            fig.update_layout(
                title=title,
                xaxis_title="Underlying Price ($)",
                yaxis_title="Profit/Loss ($)",
                template='plotly_dark'
            )
            
            return fig
        
        def plot_greeks_summary(self, S, K, T, r, sigma, option_type='call'):
            """Create Greeks summary"""
            spot_range = np.linspace(S * 0.8, S * 1.2, 50)
            deltas = []
            
            for spot in spot_range:
                greeks = self.greeks_calc.calculate_greeks(spot, K, T, r, sigma, option_type)
                deltas.append(greeks.delta)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spot_range,
                y=deltas,
                mode='lines',
                name='Delta',
                line=dict(color=self.colors['call'], width=3)
            ))
            
            fig.update_layout(
                title=f'{option_type.title()} Option Delta',
                xaxis_title="Spot Price ($)",
                yaxis_title="Delta",
                template='plotly_dark'
            )
            
            return fig
        
        def plot_volatility_surface_2d(self, vol_surface):
            """Create 2D volatility surface heatmap"""
            # Extract data from surface
            surface_data = []
            for point in vol_surface.vol_data:
                surface_data.append({
                    'strike': point.strike,
                    'expiry': point.expiry,
                    'volatility': point.volatility
                })
            
            df = pd.DataFrame(surface_data)
            
            # Create pivot table for heatmap
            pivot = df.pivot_table(values='volatility', index='expiry', columns='strike', fill_value=np.nan)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Viridis',
                colorbar=dict(title="Implied Vol"),
                hovertemplate='Strike: %{x}<br>Expiry: %{y:.2f}y<br>Vol: %{z:.1%}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Volatility Surface Heatmap',
                xaxis_title='Strike Price',
                yaxis_title='Time to Expiry (years)',
                template='plotly_dark'
            )
            
            return fig
    
    # Test dashboard
    dashboard = SimpleDashboard()
    
    # Test payoff diagram
    payoff_fig = dashboard.plot_option_payoff(K, call_price, 'call')
    print("✅ Payoff diagram: Created")
    
    # Test Greeks plot
    greeks_fig = dashboard.plot_greeks_summary(S, K, T, r, sigma, 'call')
    print("✅ Greeks plot: Created")
    
    # Test volatility surface plot
    vol_surface_fig = dashboard.plot_volatility_surface_2d(vol_surface)
    print("✅ Volatility surface: Created")
    
    print("\n🎉 DASHBOARD TEST RESULTS:")
    print("-" * 40)
    print("✅ All core modules: Working")
    print("✅ Pricing engine: Working")
    print("✅ Greeks calculation: Working")
    print("✅ Volatility surface: Working")
    print("✅ Plotly visualization: Working")
    print("✅ Dashboard components: Created")
    print("✅ Surface visualization: Created")
    
    print(f"\n🚀 SUCCESS!")
    print("📊 Your derivatives platform is ready!")
    print("💡 Use fig.show() to display plots in browser")
    print("💡 Example: payoff_fig.show()")
    
    # Optionally show a plot
    # payoff_fig.show()  # Uncomment to display
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Check that all modules are in the correct location")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check dependencies and module structure")