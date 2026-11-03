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


def _check(vols, strikes=None):
    import numpy as np

    from derivflow.volatility.surface import VolatilitySurface

    strikes = np.linspace(80, 120, 9) if strikes is None else np.asarray(strikes, float)
    return VolatilitySurface().check_arbitrage(
        strikes, np.asarray(vols, float), expiry=1.0, spot=100.0, r=0.05
    )


def test_flat_smile_is_arbitrage_free():
    res = _check([0.2] * 9)
    assert res["arbitrage_free"]
    assert not res["butterfly_arbitrage"] and not res["vertical_spread_arbitrage"]


def test_mild_frown_is_arbitrage_free():
    """A concave smile in volatility can still give convex call prices; the
    check works on prices, not on the curvature of volatility."""
    import numpy as np

    k = np.log(np.linspace(80, 120, 9) / 100.0)
    res = _check(0.2 - 0.05 * k**2)
    assert res["arbitrage_free"]


def test_volatility_spike_is_butterfly_arbitrage():
    vols = [0.2] * 9
    vols[4] = 0.35  # K = 100 call priced far above its neighbours
    res = _check(vols)
    assert res["butterfly_arbitrage"]
    assert not res["arbitrage_free"]


def test_call_price_rising_with_strike_is_vertical_spread_arbitrage():
    res = _check([0.10, 0.80], strikes=[100.0, 102.0])
    assert res["vertical_spread_arbitrage"]
    assert not res["arbitrage_free"]


def test_load_market_data_accepts_chain_rows_and_skips_unsolved_quotes():
    """Rows from implied_vol_from_chain (with option_type and possibly NaN
    volatility) load directly; quotes without a volatility are skipped."""
    from derivflow import black_scholes_price, implied_vol_from_chain
    from derivflow.volatility.surface import VolatilitySurface

    chain = [
        {
            "strike": k,
            "expiry": 0.5,
            "option_type": "call",
            "market_price": black_scholes_price(100.0, k, 0.5, 0.03, 0.25, "call"),
        }
        for k in (90.0, 100.0, 110.0)
    ]
    chain.append({"strike": 60.0, "expiry": 0.5, "option_type": "call", "market_price": 1.0})
    rows = implied_vol_from_chain(chain, S=100.0, r=0.03)
    surface = VolatilitySurface()
    surface.load_market_data([{**row, "volume": 5, "open_interest": 7} for row in rows])
    assert len(surface.vol_data) == 3
    assert all(p.volatility == pytest.approx(0.25, abs=1e-9) for p in surface.vol_data)


@pytest.mark.parametrize("method", ["rbf", "linear"])
def test_duplicate_quotes_at_one_point_are_averaged(method):
    """A call and a put at the same strike and expiry give two quotes for one
    point; the surface uses their average instead of failing."""
    from derivflow.volatility.surface import VolatilitySurface

    surface = VolatilitySurface(method=method)
    for K in (90.0, 100.0, 110.0):
        for T in (0.1, 0.5):
            surface.add_volatility_point(K, T, 0.20)
            surface.add_volatility_point(K, T, 0.22)
    surface.build_surface()
    assert surface.interpolate(100.0, 0.1) == pytest.approx(0.21, abs=1e-9)


def test_svi_fits_every_expiry_with_noisy_quotes():
    """No expiry may be dropped silently because the optimiser stopped early."""
    import warnings

    import numpy as np

    from derivflow.volatility.surface import VolatilitySurface

    rng = np.random.default_rng(1)
    strikes = np.linspace(80, 120, 8)
    k = np.log(strikes / 100.0)
    for _ in range(15):
        surface = VolatilitySurface("svi", reference_price=100.0)
        for T in (0.25, 1.0):
            vols = 0.2 + 0.02 * T - 0.1 * k + 0.25 * k * k + rng.normal(0, 0.003, k.size)
            for K, v in zip(strikes, vols, strict=True):
                surface.add_volatility_point(K, T, v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            surface.build_surface()
        assert sorted(surface.smile_params) == [0.25, 1.0]
        assert surface.interpolate(100.0, 1.0) == pytest.approx(0.22, abs=0.01)


def test_svi_warns_about_expiries_with_too_few_quotes():

    surface, _ = _svi_surface()
    for K in (90.0, 100.0, 110.0):
        surface.add_volatility_point(K, 2.0, 0.2)
    with pytest.warns(UserWarning, match="fewer than 5"):
        surface.build_surface()
    assert 2.0 not in surface.smile_params
