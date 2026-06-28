import sys
sys.path.append('src')

# Ensure non-ASCII (Greek) symbols print on any console encoding.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("DERIVFLOW-FINANCE FINAL PLATFORM TEST")
print("=" * 60)

# Test all imports
from derivflow.core import PricingEngine, price_european_option
from derivflow.greeks import GreeksCalculator
from derivflow.utils import AdvancedMarketData
from derivflow.volatility import create_sample_surface
from derivflow.exotic import AsianOptions, BarrierOptions
from derivflow.portfolio import PortfolioRiskAnalyzer

print("All modules imported successfully!")

# Test package info
from derivflow import get_package_info, demo_derivflow
info = get_package_info()
print(f"Package: {info['name']} v{info['version']}")
print(f"Author: {info['author']}")
print(f"Email: {info['email']}")

print("\nRunning comprehensive demo...")
print("=" * 60)
demo_derivflow()