"""Barrier options: in/out parity, analytical-vs-MC, and basic bounds."""

import pytest

from derivflow.exotic.barrier_options import BarrierOptions

barrier = BarrierOptions()


def _price(btype, H, option_type="call", method="analytical"):
    return barrier.price(100, 100, H, 1.0, 0.05, 0.2, btype, option_type, method).price


def test_down_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, "call")
    di = _price("down_and_in", 90)
    do = _price("down_and_out", 90)
    assert di + do == pytest.approx(vanilla, abs=tol["analytic"])


def test_up_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, "call")
    ui = _price("up_and_in", 110)
    uo = _price("up_and_out", 110)
    assert ui + uo == pytest.approx(vanilla, abs=tol["analytic"])


def test_knockout_cheaper_than_vanilla(bs_ref):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, "call")
    assert _price("down_and_out", 90) <= vanilla + 1e-9
    assert _price("up_and_out", 110) <= vanilla + 1e-9


def test_survival_probability_in_range():
    res = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, "down_and_out", "call")
    assert 0.0 <= res.probability_survival <= 1.0


@pytest.mark.slow
def test_analytical_vs_monte_carlo():
    # Discrete-monitoring MC carries a known bias vs continuous analytical, so
    # we use a loose relative tolerance rather than a std-error-tight bound.
    analytic = _price("down_and_out", 90, method="analytical")
    mc = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, "down_and_out", "call", "monte_carlo").price
    assert mc == pytest.approx(analytic, rel=0.1)


def test_already_breached_knock_in_equals_vanilla(bs_ref):
    """Regression: a barrier already breached at t=0 must give knock-in == vanilla
    and knock-out == 0 (never knock-in > vanilla)."""
    import warnings

    # _price() helper prices at sigma=0.2, so the vanilla reference must match.
    van_call = bs_ref(100, 100, 1.0, 0.05, 0.2, "call")
    van_put = bs_ref(100, 100, 1.0, 0.05, 0.2, "put")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # out-of-spec placement warns by design
        # Down barrier above spot -> already breached.
        assert _price("down_and_in", 105) == pytest.approx(van_call, abs=1e-9)
        assert _price("down_and_out", 105) == pytest.approx(0.0, abs=1e-12)
        # Up barrier below spot -> already breached (put).
        assert _price("up_and_in", 95, option_type="put") == pytest.approx(van_put, abs=1e-9)
        assert _price("up_and_out", 95, option_type="put") == pytest.approx(0.0, abs=1e-12)


def test_breached_knock_in_never_exceeds_vanilla(bs_ref):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, "call")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for H in (101, 105, 110, 130):
            assert _price("down_and_in", H) <= vanilla + 1e-9


def _no_touch_probability(S, H, T, r, sigma):
    """Risk-neutral probability that GBM started at S does not touch H before T.

    Reflection principle for X_t = mu t + sigma W_t with mu = r - sigma^2 / 2
    (e.g. Shreve, Stochastic Calculus for Finance II, section 7.2).
    """
    import numpy as np
    from scipy.stats import norm

    mu = r - 0.5 * sigma**2
    st = sigma * np.sqrt(T)
    if H > S:  # running maximum stays below b = ln(H/S) > 0
        b = np.log(H / S)
        return norm.cdf((b - mu * T) / st) - np.exp(2 * mu * b / sigma**2) * norm.cdf(
            (-b - mu * T) / st
        )
    a = np.log(S / H)  # running minimum stays above -a < 0
    return norm.cdf((a + mu * T) / st) - np.exp(-2 * mu * a / sigma**2) * norm.cdf(
        (-a + mu * T) / st
    )


@pytest.mark.parametrize(
    ("barrier_type", "H"),
    [("down_and_out", 90), ("down_and_in", 80), ("up_and_out", 110), ("up_and_in", 130)],
)
def test_survival_probability_matches_reflection_principle(barrier_type, H):
    res = barrier.price(100, 100, H, 1.0, 0.05, 0.2, barrier_type, "call")
    assert res.probability_survival == pytest.approx(
        _no_touch_probability(100, H, 1.0, 0.05, 0.2), abs=1e-12
    )


@pytest.mark.parametrize("H", [90.0, 110.0])
def test_survival_probability_matches_monte_carlo(H):
    """Discretely monitored no-touch frequency vs the continuous formula evaluated
    at the Broadie-Glasserman-Kou (1997) shifted barrier H * exp(+-0.5826 sigma sqrt(dt))."""
    import numpy as np

    S, T, r, sigma, steps, paths = 100.0, 1.0, 0.05, 0.2, 250, 200_000
    dt = T / steps
    rng = np.random.default_rng(2024)
    log_s = np.zeros(paths)
    touched = np.zeros(paths, dtype=bool)
    log_h = np.log(H / S)
    for _ in range(steps):
        log_s += (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(paths)
        touched |= (log_s >= log_h) if H > S else (log_s <= log_h)
    p_mc = 1.0 - touched.mean()
    se = np.sqrt(p_mc * (1 - p_mc) / paths)
    shift = np.exp(0.5826 * sigma * np.sqrt(dt))
    H_eff = H * shift if H > S else H / shift
    bt = "up_and_out" if H > S else "down_and_out"
    p_formula = barrier.price(S, 100.0, H_eff, T, r, sigma, bt, "call").probability_survival
    assert abs(p_mc - p_formula) < 4 * se + 2e-3



def test_monte_carlo_survival_probability_is_the_same_quantity_for_in_and_out():
    """probability_survival is P(barrier not touched) for knock-in and knock-out alike."""
    out = BarrierOptions(random_seed=11).monte_carlo_price(
        100, 100, 90, 1.0, 0.05, 0.2, "down_and_out", "call", num_sims=20_000, num_steps=50
    )
    knock_in = BarrierOptions(random_seed=11).monte_carlo_price(
        100, 100, 90, 1.0, 0.05, 0.2, "down_and_in", "call", num_sims=20_000, num_steps=50
    )
    assert knock_in.probability_survival == pytest.approx(out.probability_survival, abs=1e-12)


def test_price_barrier_option_convenience_function():
    from derivflow.exotic import price_barrier_option

    assert price_barrier_option(100, 100, 90, 1.0, 0.05, 0.2, "down_and_out") == pytest.approx(
        _price("down_and_out", 90), abs=1e-14
    )


def test_at_expiry_a_spot_beyond_the_barrier_counts_as_touched():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Down barrier at 90 and spot 85 at expiry: knocked out / knocked in.
        assert barrier.price(85, 80, 90, 0.0, 0.05, 0.2, "down_and_out", "call").price == 0.0
        assert barrier.price(85, 80, 90, 0.0, 0.05, 0.2, "down_and_in", "call").price == 5.0
    # Spot above the down barrier: knock-out pays, knock-in does not.
    assert barrier.price(105, 100, 90, 0.0, 0.05, 0.2, "down_and_out", "call").price == 5.0
    assert barrier.price(105, 100, 90, 0.0, 0.05, 0.2, "down_and_in", "call").price == 0.0
