"""Time to expiry for listed options (one day-count convention for the package)."""

from __future__ import annotations

from datetime import date, datetime, time, tzinfo
from functools import lru_cache

EXPIRY_CLOSE = time(16, 0)  # US equity options stop trading at 16:00 New York time
DAYS_PER_YEAR = 365.0  # ACT/365 fixed


@lru_cache(maxsize=1)
def new_york() -> tzinfo:
    """America/New_York time zone (on Windows this needs the ``tzdata`` package,
    which the ``[data]`` extra installs)."""
    from zoneinfo import ZoneInfo

    return ZoneInfo("America/New_York")


def year_fraction_to_expiry(expiry: str | date, now: datetime | None = None) -> float:
    """Years from ``now`` to 16:00 New York time on the expiry date, ACT/365.

    ``expiry`` is a ``datetime.date`` or a 'YYYY-MM-DD' string. ``now``
    defaults to the current time; a naive ``now`` is taken as New York time.
    Returns 0.0 for options past expiry.
    """
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
    if isinstance(expiry, datetime):
        expiry = expiry.date()
    tz = new_york()
    if now is None:
        now = datetime.now(tz)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    close = datetime.combine(expiry, EXPIRY_CLOSE, tzinfo=tz)
    seconds = (close - now).total_seconds()
    return max(seconds, 0.0) / (DAYS_PER_YEAR * 86400.0)
