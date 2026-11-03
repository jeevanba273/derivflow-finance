"""Volatility surface: sample construction, grid shape, and interpolation."""
import pytest


from derivflow.volatility.surface import create_sample_surface


def test_sample_surface_grid():
    surface = create_sample_surface()
    assert len(surface.vol_data) == 36
    strikes = {p.strike for p in surface.vol_data}
    expiries = {p.expiry for p in surface.vol_data}
    assert len(strikes) == 9
    assert len(expiries) == 4


def test_build_and_interpolate():
    surface = create_sample_surface()
    surface.build_surface()
    vol = surface.interpolate(100, 0.25)
    assert 0.05 < vol < 1.0


def test_surface_statistics():
    surface = create_sample_surface()
    stats = surface.surface_statistics()
    assert stats["num_points"] == 36
    assert 0 < stats["min_volatility"] <= stats["mean_volatility"] <= stats["max_volatility"]


def _svi_total_variance(k, a, b, rho, m, s):
    import numpy as np

    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))


def _svi_surface():
    """Surface whose slices are exact SVI smiles in log-moneyness ln(K / 100)."""
    import numpy as np

    from derivflow.volatility.surface import VolatilitySurface

    params = {0.25: (0.008, 0.04, -0.4, 0.02, 0.15), 1.0: (0.03, 0.08, -0.35, 0.03, 0.2)}
    surface = VolatilitySurface(method="svi")
    for T, p in params.items():
        for K in np.linspace(70, 130, 13):
            w = _svi_total_variance(np.log(K / 100.0), *p)
            surface.add_volatility_point(K, T, float(np.sqrt(w / T)))
    return surface, params


def test_svi_slice_reproduces_its_input_smile():
    import numpy as np

    surface, params = _svi_surface()
    surface.build_surface()
    for T, p in params.items():
        for K in (72.5, 88.0, 100.0, 117.0, 129.0):  # mostly between the input strikes
            expected = np.sqrt(_svi_total_variance(np.log(K / 100.0), *p) / T)
            assert surface.interpolate(K, T) == pytest.approx(expected, abs=1e-6)


def test_svi_interpolates_total_variance_between_expiries():
    import numpy as np

    surface, params = _svi_surface()
    surface.build_surface()
    K, T = 95.0, 0.5
    w1 = _svi_total_variance(np.log(K / 100.0), *params[0.25])
    w2 = _svi_total_variance(np.log(K / 100.0), *params[1.0])
    w = w1 + (w2 - w1) * (T - 0.25) / (1.0 - 0.25)
    assert surface.interpolate(K, T) == pytest.approx(np.sqrt(w / T), abs=1e-6)
