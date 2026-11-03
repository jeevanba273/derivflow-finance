"""
Black-Scholes-Merton implied volatility.

Method
------
The quote is first converted to its out-of-the-money equivalent with
put-call parity (an in-the-money call becomes the put with the same strike),
because the out-of-the-money price is the option's time value and carries all
the information about volatility. The volatility is then found by a
safeguarded Newton iteration ("rtsafe", Press et al., Numerical Recipes,
section 9.4) on the logarithm of that price, which stays well scaled deep in
the wings: Newton steps start from the Manaster and Koehler (1982) point
sigma0 = sqrt(2 abs(ln(F/K)) / T), a bracket [lo, hi] is kept from the sign of
the error, and bisection replaces any Newton step that leaves the bracket or
fails to halve the previous step. Iteration stops when the change in
volatility, or the bracket width, is below ``tol`` (absolute, default 1e-12).

No solution exists when the price is outside the no-arbitrage bounds
(intrinsic value of the forward, discounted, and the discounted spot for a
call or strike for a put). The volatility is not identifiable when the time
value is below 1e-12 of the option price: it is then lost in floating-point
rounding. In both cases :func:`implied_volatility` raises ``ValueError`` (or
returns NaN with ``on_fail='nan'``). Just above that threshold, and generally
where vega is tiny relative to the spot, the volatility is determined only to
a few significant digits.

Conventions: T in years, r and q continuously compounded annual rates, the
result is an annualised volatility.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SIGMA_CAP = 100.0  # 10,000% volatility: anything above is treated as no solution
_IDENTIFIABLE = 1e-12  # minimum time value as a fraction of the quoted price


def _log_otm(
    F: float, K: float, T: float, df: float, sigma: float, call: bool
) -> tuple[float, float]:
    """ln of the out-of-the-money Black-Scholes-Merton price and its derivative
    in sigma (vega / price). Returns (-inf, inf) if the price underflows."""
    st = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    d2 = d1 - st
    if call:
        price = df * (F * 0.5 * math.erfc(-d1 / _SQRT2) - K * 0.5 * math.erfc(-d2 / _SQRT2))
    else:
        price = df * (K * 0.5 * math.erfc(d2 / _SQRT2) - F * 0.5 * math.erfc(d1 / _SQRT2))
    if price <= 0.0:
        return -math.inf, math.inf
    vega = df * F * math.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI * math.sqrt(T)
    return math.log(price), vega / price


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-12,
    max_iter: int = 100,
    initial_guess: float | None = None,
    on_fail: str = "raise",
) -> float:
    """
    Implied volatility of one European option under Black-Scholes-Merton.

    Parameters:
    -----------
    price : float
        Observed option price.
    S, K : float
        Spot and strike (positive).
    T : float
        Time to expiry in years (positive).
    r, q : float
        Continuously compounded risk-free rate and dividend yield.
    option_type : str
        'call' or 'put'.
    tol : float
        Absolute tolerance on the volatility.
    max_iter : int
        Maximum number of iterations.
    initial_guess : float, optional
        Starting volatility; default is the Manaster-Koehler point.
    on_fail : str
        'raise' (default) raises ``ValueError`` with the reason when no
        volatility reproduces the price; 'nan' returns ``nan`` instead.

    Returns:
    --------
    float
        Annualised implied volatility.
    """
    if on_fail not in ("raise", "nan"):
        raise ValueError("on_fail must be 'raise' or 'nan'")

    def fail(msg: str) -> float:
        if on_fail == "raise":
            raise ValueError(msg)
        return float("nan")

    ot = str(option_type).lower()
    if ot not in ("call", "put"):
        return fail("option_type must be 'call' or 'put'")
    if not (S > 0 and math.isfinite(S)):
        return fail(f"S must be positive and finite, got {S}")
    if not (K > 0 and math.isfinite(K)):
        return fail(f"K must be positive and finite, got {K}")
    if not (T > 0 and math.isfinite(T)):
        return fail(f"T must be positive (years), got {T}")
    if price is None or not math.isfinite(price):
        return fail(f"price must be finite, got {price}")

    df = math.exp(-r * T)
    F = S * math.exp((r - q) * T)
    is_call = ot == "call"
    lower = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if price >= upper:
        return fail(f"price {price} is at or above the upper bound {upper:.6g} (no solution)")
    if price <= lower:
        return fail(
            f"price {price} is at or below the discounted intrinsic value {lower:.6g} "
            "(no positive volatility reproduces it)"
        )

    otm_call = K >= F
    time_value = price - lower if is_call != otm_call else price
    if time_value <= _IDENTIFIABLE * price:
        return fail(
            f"time value {time_value:.3e} is below {_IDENTIFIABLE:g} of the price {price:.6g}: "
            "the volatility is not identifiable in double precision"
        )

    target = math.log(time_value)
    lo, hi = 0.0, 1.0
    while _log_otm(F, K, T, df, hi, otm_call)[0] < target:
        lo, hi = hi, 2.0 * hi
        if hi > _SIGMA_CAP:
            return fail(f"implied volatility above {_SIGMA_CAP:g} (no solution in range)")

    if initial_guess is not None:
        sigma = float(initial_guess)
    else:
        k = abs(math.log(F / K))
        sigma = (
            math.sqrt(2.0 * k / T) if k > 0 else time_value / (df * F) * math.sqrt(2 * math.pi / T)
        )
    if not lo < sigma < hi:
        sigma = 0.5 * (lo + hi)

    dx_old = dx = hi - lo
    for _ in range(max_iter):
        f, slope = _log_otm(F, K, T, df, sigma, otm_call)
        f -= target
        if f == 0.0:
            return sigma
        if f < 0:
            lo = sigma
        else:
            hi = sigma
        step = f / slope if math.isfinite(f) and slope > 0 else math.nan
        candidate = sigma - step
        newton_ok = lo < candidate < hi and abs(2.0 * step) <= abs(dx_old)
        dx_old = dx
        new = candidate if newton_ok else 0.5 * (lo + hi)
        dx = new - sigma
        scale = tol * max(1.0, new)
        if abs(dx) <= scale or hi - lo <= scale:
            return new
        sigma = new
    return fail(f"no convergence in {max_iter} iterations")


def _solve_log_otm_vectorized(
    tv: np.ndarray,
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    df: np.ndarray,
    call: np.ndarray,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Array version of the rtsafe iteration in :func:`implied_volatility`."""
    target = np.log(tv)
    sqrt_t = np.sqrt(T)
    log_fk = np.log(F / K)

    def log_otm(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st = sig * sqrt_t
        d1 = (log_fk + 0.5 * st * st) / st
        d2 = d1 - st
        price = np.where(
            call,
            df * (F * ndtr(d1) - K * ndtr(d2)),
            df * (K * ndtr(-d2) - F * ndtr(-d1)),
        )
        vega = df * F * np.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI * sqrt_t
        positive = price > 0
        safe = np.where(positive, price, 1.0)
        return np.where(positive, np.log(safe), -np.inf), np.where(positive, vega / safe, np.inf)

    n = tv.size
    out = np.full(n, np.nan)
    with np.errstate(all="ignore"):
        lo = np.zeros(n)
        hi = np.ones(n)
        while True:
            short = log_otm(hi)[0] < target
            if not short.any() or hi.max() > _SIGMA_CAP:
                break
            lo = np.where(short, hi, lo)
            hi = np.where(short, 2.0 * hi, hi)
        done = log_otm(hi)[0] < target  # no solution below the cap
        k = np.abs(log_fk)
        guess = np.where(k > 0, np.sqrt(2.0 * k / T), tv / (df * F) * np.sqrt(2 * np.pi / T))
        sigma = np.where((guess > lo) & (guess < hi), guess, 0.5 * (lo + hi))
        dx_old = hi - lo
        dx = dx_old.copy()
        for _ in range(max_iter):
            f, slope = log_otm(sigma)
            f = f - target
            exact = (~done) & (f == 0)
            out[exact] = sigma[exact]
            done |= exact
            lo = np.where(f < 0, sigma, lo)
            hi = np.where(f > 0, sigma, hi)
            step = f / slope
            candidate = sigma - step
            newton_ok = (
                np.isfinite(step)
                & (candidate > lo)
                & (candidate < hi)
                & (np.abs(2.0 * step) <= np.abs(dx_old))
            )
            dx_old = dx
            new = np.where(newton_ok, candidate, 0.5 * (lo + hi))
            dx = new - sigma
            scale = tol * np.maximum(1.0, new)
            conv = (~done) & ((np.abs(dx) <= scale) | (hi - lo <= scale))
            out[conv] = new[conv]
            done |= conv
            if done.all():
                break
            sigma = np.where(done, sigma, new)
    return out


def implied_volatility_vectorized(
    prices: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: Any = "call",
    q: ArrayLike = 0.0,
    tol: float = 1e-12,
    max_iter: int = 100,
    on_fail: str = "nan",
) -> np.ndarray:
    """
    Vectorised version of :func:`implied_volatility` (same algorithm, NumPy
    arrays, all elements iterated together).

    Inputs broadcast against each other; ``option_type`` may be a string or an
    array of 'call'/'put'. Elements without a solution are NaN (or, with
    ``on_fail='raise'``, the first failure raises ``ValueError``).
    """
    p, S_, K_, T_, r_, q_ = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (prices, S, K, T, r, q))
    )
    ot = np.broadcast_to(np.char.lower(np.asarray(option_type, dtype=str)), p.shape)
    if not np.all(np.isin(ot, ("call", "put"))):
        raise ValueError("option_type must be 'call' or 'put'")
    p, S_, K_, T_, r_, q_, ot = (np.array(a).ravel() for a in (p, S_, K_, T_, r_, q_, ot))
    out = np.full(p.shape, np.nan)

    with np.errstate(all="ignore"):
        valid = (S_ > 0) & (K_ > 0) & (T_ > 0) & np.isfinite(p) & np.isfinite(S_ * K_ * T_)
        df = np.exp(-r_ * T_)
        F = S_ * np.exp((r_ - q_) * T_)
        is_call = ot == "call"
        lower = np.where(is_call, df * np.maximum(F - K_, 0.0), df * np.maximum(K_ - F, 0.0))
        upper = np.where(is_call, df * F, df * K_)
        valid &= (p > lower) & (p < upper)
        otm_call = K_ >= F
        tv = np.where(is_call != otm_call, p - lower, p)
        valid &= tv > _IDENTIFIABLE * p

    idx = np.flatnonzero(valid)
    if idx.size:
        out[idx] = _solve_log_otm_vectorized(
            tv[idx], F[idx], K_[idx], T_[idx], df[idx], otm_call[idx], tol, max_iter
        )

    if on_fail == "raise" and np.isnan(out).any():
        n = int(np.flatnonzero(np.isnan(out))[0])
        implied_volatility(p[n], S_[n], K_[n], T_[n], r_[n], ot[n], q_[n], tol, max_iter)
        raise ValueError(f"no implied volatility for element {n}")
    return out.reshape(np.broadcast_shapes(*(np.shape(x) for x in (prices, S, K, T, r, q))))


def implied_vol_from_chain(
    chain: Any, S: float, r: float, q: float = 0.0, price_field: str = "mid", **kwargs
) -> list[dict]:
    """
    Compute implied volatilities for a whole option chain.

    Accepts a :class:`~derivflow.utils.market_data.MarketDataResult`, a pandas
    ``DataFrame``, or a list of dicts. Tabular inputs must provide ``strike``,
    ``expiry`` (time to expiry in YEARS), ``option_type`` and a price column
    (``market_price``/``last_price``/``bid``/``ask`` or an explicit
    ``price_field``). The returned rows mirror the shape produced by
    :meth:`AdvancedMarketData.build_volatility_surface_from_market` so they can
    be fed straight into :meth:`VolatilitySurface.load_market_data`.

    Parameters:
    -----------
    chain : MarketDataResult | pandas.DataFrame | list[dict]
        Option chain to invert.
    S : float
        Spot price.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield (default 0.0).
    price_field : str
        Which price to invert when a row has no 'market_price': 'mid'
        (default; (bid + ask) / 2, or the last price without a two-sided
        quote), 'last', 'bid' or 'ask'.
    kwargs
        Keyword arguments: passed to :func:`implied_volatility`; ``on_fail``
        defaults to 'nan' here, so unsolvable quotes give NaN.

    Returns:
    --------
    list[dict]
        One row per option with keys: ``strike``, ``expiry`` (years),
        ``volatility`` (solved IV), ``market_price``, ``option_type``.
    """
    if price_field not in ("mid", "last", "bid", "ask"):
        raise ValueError("price_field must be 'mid', 'last', 'bid' or 'ask'")
    kwargs.setdefault("on_fail", "nan")
    rows = _normalize_chain(chain, price_field)
    results: list[dict] = []
    for row in rows:
        iv = implied_volatility(
            row["market_price"],
            S,
            row["strike"],
            row["expiry"],
            r,
            option_type=row["option_type"],
            q=q,
            **kwargs,
        )
        results.append(
            {
                "strike": row["strike"],
                "expiry": row["expiry"],
                "volatility": iv,
                "market_price": row["market_price"],
                "option_type": row["option_type"],
            }
        )
    return results


def _mid_price(last_price, bid, ask):
    """Mid of bid and ask when both are positive, else the last price."""
    try:
        if bid and ask and bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
    except TypeError:
        pass
    return last_price


def _pick_price(price_field: str, last_price: Any, bid: Any, ask: Any) -> Any:
    if price_field == "last":
        return last_price
    if price_field == "bid":
        return bid
    if price_field == "ask":
        return ask
    return _mid_price(last_price, bid, ask)


def _normalize_chain(chain: Any, price_field: str) -> list[dict]:
    """Normalize supported chain inputs to a list of {strike, expiry, market_price, option_type}."""
    # MarketDataResult (duck-typed via options_data attribute)
    if hasattr(chain, "options_data"):
        rows = []
        for opt in chain.options_data:
            expiry_years = _expiry_to_years(opt.expiry)
            if expiry_years is None or expiry_years <= 0:
                continue
            mp = _pick_price(price_field, opt.last_price, opt.bid, opt.ask)
            if mp is None or mp <= 0:
                continue
            rows.append(
                {
                    "strike": opt.strike,
                    "expiry": expiry_years,
                    "market_price": mp,
                    "option_type": opt.option_type,
                }
            )
        return rows

    if hasattr(chain, "to_dict") and hasattr(chain, "columns"):  # pandas DataFrame
        chain = chain.to_dict("records")

    # list of dicts
    rows = []
    for rec in chain:
        mp = rec.get("market_price")
        if mp is None:
            mp = _pick_price(price_field, rec.get("last_price"), rec.get("bid"), rec.get("ask"))
        if mp is None or mp <= 0:
            continue
        rows.append(
            {
                "strike": rec["strike"],
                "expiry": rec["expiry"],
                "market_price": mp,
                "option_type": rec.get("option_type", "call"),
            }
        )
    return rows


def _expiry_to_years(expiry: Any) -> float | None:
    """Years to expiry: numeric values are taken as years; 'YYYY-MM-DD' strings
    and dates use ACT/365 to 16:00 New York time (see derivflow._dates)."""
    if isinstance(expiry, (int, float)):
        return float(expiry)
    try:
        from .._dates import year_fraction_to_expiry

        return year_fraction_to_expiry(expiry)
    except (ValueError, TypeError):
        return None
