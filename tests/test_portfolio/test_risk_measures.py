"""VaR and expected shortfall on distributions with known quantiles.

Losses are reported as positive numbers. References: the normal and Student-t
closed forms (McNeil, Frey and Embrechts, Quantitative Risk Management, 2nd
ed., 2015, section 2.3); the t expected shortfall is also checked against
direct numerical integration of the tail.
"""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, t

from derivflow.portfolio.risk_measures import (
    expected_shortfall,
    normal_es,
    normal_var,
    student_t_es,
    student_t_var,
    value_at_risk,
)


@pytest.mark.parametrize("c", [0.95, 0.99, 0.999])
def test_normal_closed_forms(c):
    assert normal_var(2.0, c) == pytest.approx(2.0 * norm.ppf(c), rel=1e-14)
    assert normal_es(2.0, c) == pytest.approx(2.0 * norm.pdf(norm.ppf(c)) / (1 - c), rel=1e-14)
    # A positive expected P&L lowers both measures one for one.
    assert normal_var(2.0, c, mu=0.5) == pytest.approx(normal_var(2.0, c) - 0.5, rel=1e-14)


@pytest.mark.parametrize("df", [3, 5, 30])
@pytest.mark.parametrize("c", [0.95, 0.99])
def test_student_t_es_matches_numerical_tail_integral(df, c):
    q = t.ppf(c, df)
    tail_mean, _ = integrate.quad(lambda x: x * t.pdf(x, df), q, np.inf)
    assert student_t_var(1.5, df, c) == pytest.approx(1.5 * q, rel=1e-12)
    assert student_t_es(1.5, df, c) == pytest.approx(1.5 * tail_mean / (1 - c), rel=1e-9)


def test_student_t_es_needs_finite_mean():
    with pytest.raises(ValueError):
        student_t_es(1.0, 1.0, 0.99)


@pytest.mark.parametrize("c", [0.95, 0.99])
def test_empirical_estimators_on_normal_sample(c):
    rng = np.random.default_rng(123)
    pnl = rng.normal(0.0, 1.0, 2_000_000)
    assert value_at_risk(pnl, c) == pytest.approx(norm.ppf(c), rel=5e-3)
    assert expected_shortfall(pnl, c) == pytest.approx(normal_es(1.0, c), rel=5e-3)


@pytest.mark.parametrize("c", [0.95, 0.99])
def test_empirical_estimators_on_student_t_sample(c):
    rng = np.random.default_rng(321)
    df = 4
    pnl = t.rvs(df, size=2_000_000, random_state=rng)
    assert value_at_risk(pnl, c) == pytest.approx(student_t_var(1.0, df, c), rel=1e-2)
    assert expected_shortfall(pnl, c) == pytest.approx(student_t_es(1.0, df, c), rel=2e-2)


def test_empirical_estimators_small_example():
    pnl = np.array([-10.0, -4.0, -1.0, 0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0])
    # 80% VaR is minus the 20% quantile (numpy's default linear interpolation).
    assert value_at_risk(pnl, 0.8) == pytest.approx(-np.quantile(pnl, 0.2))
    # Expected shortfall averages the P&L at or below -VaR.
    var = value_at_risk(pnl, 0.8)
    assert expected_shortfall(pnl, 0.8) == pytest.approx(-pnl[pnl <= -var].mean())


def test_rejects_bad_confidence():
    with pytest.raises(ValueError):
        value_at_risk([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        normal_var(1.0, 0.0)
