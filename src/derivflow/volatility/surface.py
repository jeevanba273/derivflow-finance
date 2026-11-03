"""
Implied volatility surfaces from (strike, expiry, volatility) quotes.

Methods: 'rbf' (thin-plate radial basis function on (ln K, T), exact at the
quotes), 'linear' (piecewise-linear on a Delaunay triangulation of the same
coordinates) and 'svi' (raw SVI of Gatheral (2004) per expiry, total variance
linear in T between expiries). Expiries are in years and volatilities are
annualised. None of the methods enforces absence of arbitrage;
:meth:`VolatilitySurface.check_arbitrage` tests a smile for static
arbitrage across strikes.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate


@dataclass
class VolatilityPoint:
    """Single volatility data point"""

    strike: float
    expiry: float
    volatility: float
    market_price: float | None = None
    bid: float | None = None
    ask: float | None = None


@dataclass
class VolatilitySurfaceResult:
    """Result container for volatility surface operations"""

    strikes: np.ndarray
    expiries: np.ndarray
    volatilities: np.ndarray
    interpolated_vol: float | None = None
    smile_params: dict | None = None


class VolatilitySurface:
    """
    Implied volatility surface built from (strike, expiry, volatility) quotes.

    Methods: 'rbf' (thin-plate radial basis function, exact at the quotes),
    'linear' (Delaunay piecewise-linear) and 'svi' (raw SVI smile per expiry,
    total variance interpolated linearly in expiry). No method enforces
    absence of arbitrage; use :meth:`check_arbitrage` to test a smile.
    """

    def __init__(self, method: str = "rbf", reference_price: float | None = None):
        """
        Initialize volatility surface engine

        Parameters:
        -----------
        method : str
            'rbf' (default), 'linear' or 'svi'; see :meth:`build_surface`.
        reference_price : float, optional
            Spot or forward used to define log-moneyness ln(K / reference) for
            SVI fits. If omitted, each expiry uses its median strike.
        """
        self.method = method
        self.reference_price = reference_price
        self.vol_data = []
        self.surface_fitted = False
        self.interpolator = None
        self._nearest = None
        self._scale = (1.0, 1.0)
        self.smile_params = {}

    def add_volatility_point(
        self, strike: float, expiry: float, volatility: float, market_price: float | None = None
    ) -> None:
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
            strike=strike, expiry=expiry, volatility=volatility, market_price=market_price
        )
        self.vol_data.append(vol_point)
        self.surface_fitted = False

    def load_market_data(self, data: Any) -> int:
        """
        Load quotes in bulk and return how many were added.

        Parameters:
        -----------
        data : pandas.DataFrame or list of dict
            Rows with 'strike', 'expiry' (years) and 'volatility', and
            optionally 'market_price'. Other keys (e.g. 'option_type' from
            :func:`~derivflow.volatility.implied_vol.implied_vol_from_chain` or
            'volume' from the market data client) are ignored. Rows whose
            volatility is missing, NaN or not positive are skipped.
        """
        if hasattr(data, "to_dict") and hasattr(data, "columns"):  # pandas DataFrame
            rows = data.to_dict("records")
        elif isinstance(data, list):
            rows = data
        else:
            raise TypeError("data must be a pandas DataFrame or a list of dicts")
        added = 0
        for row in rows:
            vol = row.get("volatility")
            if vol is None or not np.isfinite(vol) or vol <= 0:
                continue
            price = row.get("market_price")
            self.add_volatility_point(
                strike=float(row["strike"]),
                expiry=float(row["expiry"]),
                volatility=float(vol),
                market_price=None if price is None else float(price),
            )
            added += 1
        return added

    def _prepare_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for surface fitting"""
        if not self.vol_data:
            raise ValueError("No volatility data available")

        strikes = np.array([point.strike for point in self.vol_data])
        expiries = np.array([point.expiry for point in self.vol_data])
        volatilities = np.array([point.volatility for point in self.vol_data])

        return strikes, expiries, volatilities

    def _scaled_points(self, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
        """Coordinates (ln K, T), each divided by its spread in the data, so that
        strike and expiry carry comparable weight in the interpolation."""
        return np.column_stack(
            (np.log(strikes) / self._scale[0], np.asarray(expiries, dtype=float) / self._scale[1])
        )

    def build_surface_interpolated(self) -> None:
        """Build the 'linear' or 'rbf' surface on (ln K, T).

        - 'linear': piecewise-linear interpolation on a Delaunay triangulation
          (``scipy.interpolate.LinearNDInterpolator``); outside the convex hull
          of the data the nearest quote is used.
        - 'rbf': thin-plate-spline radial basis function interpolation with no
          smoothing (``scipy.interpolate.RBFInterpolator``), so the surface
          passes through every quote.

        Several quotes at the same (strike, expiry), e.g. a call and a put,
        are replaced by their average volatility before interpolating.
        """
        strikes, expiries, volatilities = self._prepare_data_arrays()
        keys, inverse = np.unique(np.column_stack((strikes, expiries)), axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        volatilities = np.bincount(inverse, weights=volatilities) / np.bincount(inverse)
        strikes, expiries = keys[:, 0], keys[:, 1]
        log_k = np.log(strikes)
        self._scale = (
            float(np.ptp(log_k)) or 1.0,
            float(np.ptp(expiries)) or 1.0,
        )
        points = self._scaled_points(strikes, expiries)
        if self.method == "linear":
            self.interpolator = interpolate.LinearNDInterpolator(points, volatilities)
            self._nearest = interpolate.NearestNDInterpolator(points, volatilities)
        else:
            self.interpolator = interpolate.RBFInterpolator(
                points, volatilities, kernel="thin_plate_spline", smoothing=0.0
            )
        self.surface_fitted = True

    @staticmethod
    def _svi_total_variance(
        k: np.ndarray | float, a: float, b: float, sigma: float, rho: float, m: float
    ) -> np.ndarray | float:
        """Raw SVI total implied variance (Gatheral, 2004):

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),

        with k = ln(K / reference strike) and w = implied_vol^2 * T.
        """
        km = np.asarray(k) - m
        return a + b * (rho * km + np.sqrt(km**2 + sigma**2))

    def _calibrate_svi_slice(
        self, strikes: np.ndarray, volatilities: np.ndarray, expiry: float
    ) -> dict[str, float] | None:
        """Fit raw SVI to one expiry by least squares in total variance.

        The reference strike is the median strike of the slice (or the
        ``reference_price`` given to the constructor); it is stored with the
        parameters so that :meth:`interpolate` uses the same log-moneyness.
        """
        from scipy.optimize import least_squares

        ref = float(self.reference_price) if self.reference_price else float(np.median(strikes))
        k = np.log(strikes / ref)
        w_market = volatilities**2 * expiry
        w_max = float(np.max(w_market))

        def residuals(x: np.ndarray) -> np.ndarray:
            a, b, sigma, rho, m = x
            return self._svi_total_variance(k, a, b, sigma, rho, m) - w_market

        def jacobian(x: np.ndarray) -> np.ndarray:
            _, b, sigma, rho, m = x
            km = k - m
            root = np.sqrt(km**2 + sigma**2)
            return np.column_stack(
                (np.ones_like(k), rho * km + root, b * sigma / root, b * km, -b * (rho + km / root))
            )

        lower = [-w_max, 0.0, 1e-4, -0.999, float(k.min()) - 1.0]
        upper = [w_max, 10.0, 5.0, 0.999, float(k.max()) + 1.0]
        best = None
        k_min = float(k[np.argmin(w_market)])
        for rho0 in (-0.5, 0.0, 0.5):
            for sigma0 in (0.05, 0.3):
                x0 = [0.5 * float(np.min(w_market)), 0.1, sigma0, rho0, k_min]
                x0 = np.clip(x0, lower, upper)
                fit = least_squares(
                    residuals,
                    x0,
                    jac=jacobian,
                    bounds=(lower, upper),
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                    max_nfev=2000,
                )
                if np.all(np.isfinite(fit.x)) and (best is None or fit.cost < best.cost):
                    best = fit
        if best is None:
            return None
        if not best.success:
            rmse = float(np.sqrt(2.0 * best.cost / k.size))
            warnings.warn(
                f"SVI fit for expiry {expiry:g} did not converge ({best.message}); using the "
                f"best parameters found (total-variance RMSE {rmse:.2e})",
                RuntimeWarning,
                stacklevel=3,
            )
        params = dict(zip(["a", "b", "sigma", "rho", "m"], map(float, best.x), strict=True))
        params["reference_strike"] = ref
        return params

    def build_surface_svi(self) -> None:
        """Fit an SVI smile to every expiry that has at least five quotes.

        Expiries with fewer quotes than the five SVI parameters are skipped
        with a warning. A fit that stops before meeting its tolerances keeps
        the best parameters found and warns (RuntimeWarning) with its RMSE.
        """
        strikes, expiries, volatilities = self._prepare_data_arrays()

        self.smile_params = {}
        for expiry in np.unique(expiries):
            mask = expiries == expiry
            if mask.sum() < 5:
                warnings.warn(
                    f"expiry {expiry:g} has fewer than 5 quotes and gets no SVI fit",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            svi_params = self._calibrate_svi_slice(strikes[mask], volatilities[mask], float(expiry))
            if svi_params:
                self.smile_params[float(expiry)] = svi_params

        self.surface_fitted = True

    def _svi_slice_variance(self, strike: float, expiry: float) -> float:
        params = dict(self.smile_params[expiry])
        ref = params.pop("reference_strike")
        return float(self._svi_total_variance(np.log(strike / ref), **params))

    def _interpolate_svi(self, strike: float, expiry: float) -> float:
        """Total variance from the fitted slices, linear in T at fixed strike.

        Outside the fitted expiry range the implied volatility of the nearest
        slice is held constant.
        """
        if not self.smile_params:
            raise ValueError("No SVI parameters available (each expiry needs >= 5 quotes)")
        if expiry <= 0:
            raise ValueError("expiry must be positive")
        ts = sorted(self.smile_params)
        if expiry <= ts[0]:
            w = self._svi_slice_variance(strike, ts[0]) * expiry / ts[0]
        elif expiry >= ts[-1]:
            w = self._svi_slice_variance(strike, ts[-1]) * expiry / ts[-1]
        else:
            hi = next(n for n, t in enumerate(ts) if t >= expiry)
            t1, t2 = ts[hi - 1], ts[hi]
            w1 = self._svi_slice_variance(strike, t1)
            w2 = self._svi_slice_variance(strike, t2)
            w = w1 + (w2 - w1) * (expiry - t1) / (t2 - t1)
        return float(np.sqrt(max(w, 0.0) / expiry))

    def build_surface(self, method: str | None = None) -> None:
        """
        Build volatility surface using specified method

        Parameters:
        -----------
        method : str, optional
            Override the constructor's method: 'rbf', 'linear' or 'svi'.
            'cubic_spline' is a deprecated alias for 'rbf' (it never was a
            cubic spline).
        """
        if method:
            self.method = method
        if self.method == "cubic_spline":
            warnings.warn(
                "method='cubic_spline' is deprecated; it is a radial basis function "
                "surface, use method='rbf'",
                DeprecationWarning,
                stacklevel=2,
            )
            self.method = "rbf"

        if not self.vol_data:
            raise ValueError("No volatility data to fit surface")

        if self.method in ("rbf", "linear"):
            self.build_surface_interpolated()
        elif self.method == "svi":
            self.build_surface_svi()
        else:
            raise ValueError(f"Unknown method: {self.method!r} (use 'rbf', 'linear' or 'svi')")

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

        if self.method == "svi":
            return self._interpolate_svi(strike, expiry)

        point = self._scaled_points(np.array([strike], dtype=float), np.array([expiry]))
        vol = float(np.asarray(self.interpolator(point)).ravel()[0])
        if self.method == "linear" and np.isnan(vol):
            vol = float(np.asarray(self._nearest(point)).ravel()[0])
        if not vol > 0:
            raise ValueError(
                f"Interpolated volatility {vol:.6g} at K={strike}, T={expiry} is not positive "
                "(RBF extrapolation far from the data); query inside the data range or use "
                "method='linear'"
            )
        return vol

    def get_smile(
        self,
        expiry: float,
        strike_range: tuple[float, float] | None = None,
        num_points: int = 50,
    ) -> dict[str, np.ndarray]:
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
            strikes_data = [
                point.strike for point in self.vol_data if abs(point.expiry - expiry) < 0.1
            ]
            if not strikes_data:
                raise ValueError(f"No data found near expiry {expiry}")

            min_strike = min(strikes_data) * 0.8
            max_strike = max(strikes_data) * 1.2
        else:
            min_strike, max_strike = strike_range

        # Sample the smile
        strikes = np.linspace(min_strike, max_strike, num_points)
        volatilities = np.array([self.interpolate(k, expiry) for k in strikes])

        return {"strikes": strikes, "volatilities": volatilities, "expiry": expiry}

    def check_arbitrage(
        self,
        strikes: np.ndarray,
        volatilities: np.ndarray,
        expiry: float,
        spot: float,
        r: float = 0.0,
        q: float = 0.0,
        tol: float = 1e-10,
    ) -> dict[str, Any]:
        """
        Static-arbitrage check of one smile, on the call prices it implies.

        Calls C(K) are priced with Black-Scholes-Merton at each quote's
        volatility. With strikes sorted, the smile is free of static arbitrage
        across strikes when

        - vertical spreads: -e^{-rT} <= (C(K2) - C(K1)) / (K2 - K1) <= 0, and
        - butterflies: these slopes are non-decreasing in K (C convex in K).

        Calendar arbitrage involves several expiries and is not checked here.

        Returns:
        --------
        dict
            'vertical_spread_arbitrage', 'butterfly_arbitrage',
            'arbitrage_free' (bools) and 'call_prices' (array, sorted strikes).
        """
        from ..core.pricing_engine import BlackScholesAnalytical

        K = np.asarray(strikes, dtype=float)
        vols = np.asarray(volatilities, dtype=float)
        if K.shape != vols.shape or K.ndim != 1 or K.size < 2:
            raise ValueError("strikes and volatilities must be 1-D arrays of equal length >= 2")
        order = np.argsort(K)
        K, vols = K[order], vols[order]
        bs = BlackScholesAnalytical()
        calls = np.array(
            [bs.price(spot, k, expiry, r, v, "call", q) for k, v in zip(K, vols, strict=True)]
        )
        slopes = np.diff(calls) / np.diff(K)
        vertical = bool(np.any(slopes > tol) or np.any(slopes < -np.exp(-r * expiry) - tol))
        butterfly = bool(np.any(np.diff(slopes) < -tol)) if slopes.size >= 2 else False
        return {
            "vertical_spread_arbitrage": vertical,
            "butterfly_arbitrage": butterfly,
            "arbitrage_free": not (vertical or butterfly),
            "call_prices": calls,
        }

    def surface_statistics(self) -> dict[str, float]:
        """Calculate surface quality statistics"""
        if not self.vol_data:
            return {}

        volatilities = [point.volatility for point in self.vol_data]

        return {
            "num_points": len(self.vol_data),
            "min_volatility": min(volatilities),
            "max_volatility": max(volatilities),
            "mean_volatility": np.mean(volatilities),
            "vol_of_vol": np.std(volatilities),
            "unique_expiries": len(set(point.expiry for point in self.vol_data)),
            "unique_strikes": len(set(point.strike for point in self.vol_data)),
        }

    def plot_surface(self, figsize: tuple[int, int] = (12, 8)) -> None:
        """Plot 3D volatility surface (requires matplotlib, extra ``[plot]``)"""
        import matplotlib.pyplot as plt

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
                except ValueError:
                    vol_grid[i, j] = np.nan

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot surface
        surf = ax.plot_surface(K_grid, T_grid, vol_grid, cmap="viridis", alpha=0.8)

        # Plot original data points
        ax.scatter(strikes, expiries, volatilities, color="red", s=50, alpha=0.8)

        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to Expiry")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("Volatility Surface")

        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()


# Convenience functions
def create_sample_surface() -> VolatilitySurface:
    """Create sample volatility surface for testing"""
    surface = VolatilitySurface()

    # Illustrative quotes: skewed smile, upward term structure, small noise
    rng = np.random.default_rng(42)

    spot = 100
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    expiries = np.array([0.083, 0.25, 0.5, 1.0])  # 1M, 3M, 6M, 1Y

    for expiry in expiries:
        for strike in strikes:
            moneyness = np.log(strike / spot)
            base_vol = 0.2 + 0.1 * expiry  # Positive term structure
            skew = -0.1 * moneyness  # Negative skew
            smile = 0.05 * moneyness**2  # Smile curvature
            noise = rng.normal(0, 0.01)  # Small noise

            vol = base_vol + skew + smile + noise
            vol = max(vol, 0.05)  # Minimum vol

            surface.add_volatility_point(strike, expiry, vol)

    return surface
