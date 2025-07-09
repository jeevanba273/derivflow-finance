"""
DERIVFLOW-FINANCE: PyPI Package Setup Configuration
==================================================

Professional setup.py for PyPI publishing with all required metadata.
"""

from setuptools import setup, find_packages

# Read long description from README
def read_long_description():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "DERIVFLOW-FINANCE: Advanced Derivatives Analytics Platform"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
    except FileNotFoundError:
        return []


setup(
    name="derivflow-finance",
    use_scm_version=True,  # <-- Use Git tag as version
    author="Jeevan B A",
    author_email="jeevanba273@gmail.com",
    description="Advanced derivatives analytics platform for quantitative finance",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/jeevanba273/derivflow-finance",
    project_urls={
        "Bug Tracker": "https://github.com/jeevanba273/derivflow-finance/issues",
        "Documentation": "https://github.com/jeevanba273/derivflow-finance/wiki",
        "Source Code": "https://github.com/jeevanba273/derivflow-finance",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "numba>=0.56.0",
        "yfinance>=0.1.70",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
        "testing": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-xdist>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "derivflow-demo=derivflow:demo_derivflow",
            "derivflow-info=derivflow:get_package_info",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "derivatives", "finance", "quantitative", "options", "pricing",
        "risk", "analytics", "monte-carlo", "black-scholes", "greeks",
        "volatility", "stochastic", "exotic-options", "portfolio",
        "var", "heston", "barrier-options", "asian-options",
        "financial-engineering", "quant", "trading", "wall-street", "fintech"
    ],
    license="MIT",
    platforms=["any"],
)
