"""Volatility surface: sample construction, grid shape, and interpolation."""

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
    assert stats['num_points'] == 36
    assert 0 < stats['min_volatility'] <= stats['mean_volatility'] <= stats['max_volatility']
