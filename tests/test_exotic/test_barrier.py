"""Barrier options: in/out parity, analytical-vs-MC, and basic bounds."""

import pytest

from derivflow.exotic.barrier_options import BarrierOptions

barrier = BarrierOptions()


def _price(btype, H, option_type='call', method='analytical'):
    return barrier.price(100, 100, H, 1.0, 0.05, 0.2, btype, option_type, method).price


def test_down_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    di = _price('down_and_in', 90)
    do = _price('down_and_out', 90)
    assert di + do == pytest.approx(vanilla, abs=tol['analytic'])


def test_up_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    ui = _price('up_and_in', 110)
    uo = _price('up_and_out', 110)
    assert ui + uo == pytest.approx(vanilla, abs=tol['analytic'])


def test_knockout_cheaper_than_vanilla(bs_ref):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    assert _price('down_and_out', 90) <= vanilla + 1e-9
    assert _price('up_and_out', 110) <= vanilla + 1e-9


def test_survival_probability_in_range():
    res = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, 'down_and_out', 'call')
    assert 0.0 <= res.probability_survival <= 1.0


@pytest.mark.slow
def test_analytical_vs_monte_carlo():
    # Discrete-monitoring MC carries a known bias vs continuous analytical, so
    # we use a loose relative tolerance rather than a std-error-tight bound.
    analytic = _price('down_and_out', 90, method='analytical')
    mc = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, 'down_and_out', 'call',
                       'monte_carlo').price
    assert mc == pytest.approx(analytic, rel=0.1)


def test_already_breached_knock_in_equals_vanilla(bs_ref):
    """Regression: a barrier already breached at t=0 must give knock-in == vanilla
    and knock-out == 0 (never knock-in > vanilla)."""
    import warnings
    # _price() helper prices at sigma=0.2, so the vanilla reference must match.
    van_call = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    van_put = bs_ref(100, 100, 1.0, 0.05, 0.2, 'put')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # out-of-spec placement warns by design
        # Down barrier above spot -> already breached.
        assert _price('down_and_in', 105) == pytest.approx(van_call, abs=1e-9)
        assert _price('down_and_out', 105) == pytest.approx(0.0, abs=1e-12)
        # Up barrier below spot -> already breached (put).
        assert _price('up_and_in', 95, option_type='put') == pytest.approx(van_put, abs=1e-9)
        assert _price('up_and_out', 95, option_type='put') == pytest.approx(0.0, abs=1e-12)


def test_breached_knock_in_never_exceeds_vanilla(bs_ref):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for H in (101, 105, 110, 130):
            assert _price('down_and_in', H) <= vanilla + 1e-9
