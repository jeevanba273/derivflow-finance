"""Volatility surface: SVI against a known slice, error paths and plotting."""

import warnings

import numpy as np
import pytest

from derivflow_finance.volatility.surface import VolatilitySurface, create_sample_surface

SVI = {"a": 0.01, "b": 0.1, "sigma": 0.2, "rho": -0.4, "m": 0.05}


def _svi_vol(K, T, ref=100.0):
    k = np.log(np.asarray(K, dtype=float) / ref)
    km = k - SVI["m"]
    w = SVI["a"] + SVI["b"] * (SVI["rho"] * km + np.sqrt(km**2 + SVI["sigma"] ** 2))
    return np.sqrt(w / T)


def test_svi_recovers_a_known_slice_between_the_quotes():
    """Quotes generated from raw SVI parameters: the fitted slice reproduces
    the smile at strikes that were not quoted."""
    T = 0.5
    surface = VolatilitySurface("svi", reference_price=100.0)
    for K in np.linspace(70, 130, 13):
        surface.add_volatility_point(K, T, float(_svi_vol(K, T)))
    surface.build_surface()
    for K in (72.5, 88.8, 97.3, 101.1, 113.0, 127.9):
        assert surface.interpolate(K, T) == pytest.approx(float(_svi_vol(K, T)), abs=1e-7)


def test_building_without_data_or_with_an_unknown_method_raises():
    with pytest.raises(ValueError, match="No volatility data"):
        VolatilitySurface().build_surface()
    surface = create_sample_surface()
    with pytest.raises(ValueError, match="Unknown method"):
        surface.build_surface("kriging")
    with pytest.raises(ValueError, match="No volatility data"):
        VolatilitySurface()._prepare_data_arrays()


def test_svi_needs_five_quotes_per_expiry_and_a_positive_expiry():
    surface = VolatilitySurface("svi")
    for K in (90, 100, 110):
        surface.add_volatility_point(K, 0.5, 0.2)
    with pytest.warns(UserWarning, match="fewer than 5 quotes"):
        surface.build_surface()
    with pytest.raises(ValueError, match="No SVI parameters"):
        surface.interpolate(100, 0.5)

    fitted = create_sample_surface()
    fitted.method = "svi"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # noisy sample quotes
        fitted.build_surface()
    with pytest.raises(ValueError, match="expiry must be positive"):
        fitted.interpolate(100, 0.0)


def test_rbf_reproduces_a_linear_smile_and_rejects_negative_extrapolation():
    """The thin-plate spline on (ln K, T) has a linear polynomial term, so a
    smile linear in ln K is reproduced exactly between the quotes; far
    outside them it extrapolates that line below zero, which is an error."""

    def vol(K):
        return 0.5 - 0.5 * np.log(K / 80.0)

    surface = VolatilitySurface("rbf")
    for T in (0.5, 1.0):
        for K in range(80, 121, 5):
            surface.add_volatility_point(K, T, float(vol(K)))
    surface.build_surface()
    for K, T in [(100.0, 0.75), (87.5, 0.6), (118.0, 0.9)]:
        assert surface.interpolate(K, T) == pytest.approx(float(vol(K)), abs=1e-9)
    with pytest.raises(ValueError, match="not positive"):
        surface.interpolate(400, 0.75)


def test_linear_surface_uses_the_nearest_quote_outside_the_data():
    surface = VolatilitySurface("linear")
    for K, v in [(90, 0.25), (100, 0.2), (110, 0.22)]:
        for T in (0.5, 1.0):
            surface.add_volatility_point(K, T, v + 0.01 * T)
    surface.build_surface()
    assert surface.interpolate(150, 1.0) == pytest.approx(0.23)
    assert surface.interpolate(60, 0.5) == pytest.approx(0.255)


def test_load_market_data_input_types():
    surface = VolatilitySurface()
    with pytest.raises(TypeError, match="DataFrame or a list"):
        surface.load_market_data({"strike": 100, "expiry": 1.0, "volatility": 0.2})
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"strike": [90.0, 100.0, 110.0], "expiry": [0.5] * 3, "volatility": [0.25, np.nan, 0.21]}
    )
    assert surface.load_market_data(frame) == 2


def test_get_smile_ranges_and_errors():
    surface = create_sample_surface()
    smile = surface.get_smile(0.5, strike_range=(90.0, 110.0), num_points=5)
    assert np.allclose(smile["strikes"], [90, 95, 100, 105, 110])
    assert np.all(smile["volatilities"] > 0)
    empty = VolatilitySurface()
    with pytest.raises(ValueError, match="No volatility data"):
        empty.get_smile(0.5)


def test_check_arbitrage_validates_its_inputs():
    surface = VolatilitySurface()
    with pytest.raises(ValueError, match="1-D arrays of equal length"):
        surface.check_arbitrage([100.0], [0.2], 0.5, 100.0)
    with pytest.raises(ValueError, match="1-D arrays of equal length"):
        surface.check_arbitrage([90.0, 100.0], [0.2], 0.5, 100.0)


def test_statistics_of_an_empty_surface():
    assert VolatilitySurface().surface_statistics() == {}


def test_plot_surface_draws_the_fitted_surface_and_the_quotes(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    create_sample_surface().plot_surface(figsize=(6, 4))
    assert len(shown) == 1
    ax = shown[0].axes[0]
    assert ax.get_title() == "Volatility Surface" and ax.name == "3d"
    plt.close("all")
