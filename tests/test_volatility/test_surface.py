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


@pytest.mark.parametrize("method", ["rbf", "linear"])
def test_interpolating_methods_reproduce_the_input_points(method):
    surface = create_sample_surface()
    surface.build_surface(method=method)
    for p in surface.vol_data:
        assert surface.interpolate(p.strike, p.expiry) == pytest.approx(p.volatility, abs=1e-8)


def test_linear_interpolation_between_two_strikes():
    from derivflow.volatility.surface import VolatilitySurface

    surface = VolatilitySurface(method="linear")
    for K, v in ((90, 0.30), (110, 0.20)):
        for T in (0.5, 1.0):
            surface.add_volatility_point(K, T, v)
    surface.build_surface()
    # Interpolation is in (ln K, T); at the log-midpoint the weights are equal.
    import numpy as np

    k_mid = float(np.exp(0.5 * (np.log(90) + np.log(110))))
    assert surface.interpolate(k_mid, 0.75) == pytest.approx(0.25, abs=1e-12)


def test_cubic_spline_name_is_a_deprecated_alias_for_rbf():
    surface = create_sample_surface()
    with pytest.warns(DeprecationWarning):
        surface.build_surface(method="cubic_spline")
    assert surface.method == "rbf"


def test_building_a_surface_prints_nothing(capsys):
    surface = create_sample_surface()
    surface.build_surface(method="rbf")
    surface.build_surface(method="svi")
    surface.load_market_data([{"strike": 100, "expiry": 2.0, "volatility": 0.2}])
    assert capsys.readouterr().out == ""
