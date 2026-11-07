"""Expiry year fractions, input checks and the lazily loaded optional names."""

import builtins
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

import derivflow_finance
from derivflow_finance._dates import new_york, year_fraction_to_expiry
from derivflow_finance._validation import check_live_inputs, check_market_inputs, check_option_type

try:
    NY = new_york()
except Exception:  # pragma: no cover - no IANA time-zone data
    NY = None
needs_tz = pytest.mark.skipif(NY is None, reason="needs the America/New_York time zone (tzdata)")


@needs_tz
def test_one_day_to_the_close_is_one_365th_of_a_year():
    now = datetime(2026, 12, 17, 16, 0, tzinfo=NY)
    assert year_fraction_to_expiry("2026-12-18", now) == pytest.approx(1 / 365, abs=1e-15)


@needs_tz
def test_daylight_saving_change_counts_elapsed_hours():
    # Clocks go back at 02:00 on 2026-11-01: 40 wall-clock hours are 41 real hours.
    now = datetime(2026, 11, 1, 0, 0, tzinfo=NY)
    assert year_fraction_to_expiry(date(2026, 11, 2), now) == pytest.approx(41 / 24 / 365)


@needs_tz
def test_expiry_and_now_in_any_accepted_form():
    aware = datetime(2026, 6, 1, 9, 30, tzinfo=NY)
    naive = datetime(2026, 6, 1, 9, 30)
    utc = aware.astimezone(timezone.utc)
    expected = year_fraction_to_expiry("2026-06-19", aware)
    assert year_fraction_to_expiry(date(2026, 6, 19), naive) == pytest.approx(expected)
    assert year_fraction_to_expiry(datetime(2026, 6, 19, 8, 0), utc) == pytest.approx(expected)
    assert year_fraction_to_expiry("2026-05-29", aware) == 0.0  # already expired


@needs_tz
def test_now_defaults_to_the_current_time():
    tomorrow = (datetime.now(NY) + timedelta(days=2)).date()
    fraction = year_fraction_to_expiry(tomorrow)
    assert 1 / 365 < fraction < 3 / 365


@pytest.mark.parametrize("bad", ["100", None, float("nan"), -1.0, 0.0])
def test_positive_inputs_reject_non_numbers_and_non_positive_values(bad):
    with pytest.raises(ValueError, match="S must be positive"):
        check_market_inputs(bad, 100.0, 1.0, 0.2)


def test_time_and_option_type_checks():
    with pytest.raises(ValueError, match="finite time"):
        check_market_inputs(100.0, 100.0, "1y", 0.2)
    check_market_inputs(100.0, 100.0, 0.0, -1.0)  # expired: sigma is not needed
    with pytest.raises(ValueError, match="T must be positive"):
        check_live_inputs(100.0, 100.0, 0.0, 0.2)
    assert check_option_type("PUT") == "put"
    with pytest.raises(ValueError, match="option_type"):
        check_option_type("straddle")


def test_missing_optional_dependency_names_the_extra(monkeypatch):
    real_import = builtins.__import__

    def no_plotly(name, *args, **kwargs):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("No module named 'plotly'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "derivflow_finance.visualization.dashboard", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_plotly)
    with pytest.raises(ImportError, match=r"derivflow-finance\[plot\]"):
        _ = derivflow_finance.DerivativesDashboard
