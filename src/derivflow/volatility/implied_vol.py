"""
DERIVFLOW-FINANCE: Implied Volatility Solver
============================================

Recover the Black-Scholes(-Merton) implied volatility from an observed option
price. This is the entry point for almost every real-world options workflow:
market quotes give prices, and models need volatilities.

The solver uses a fast Newton-Raphson iteration (seeded with the
Brenner-Subrahmanyam at-the-money approximation) and falls back to a robust
Brent bracketing method when Newton struggles (deep in/out-of-the-money quotes,
near-zero vega, or prices close to the no-arbitrage bounds).

Continuous dividend yield ``q`` is supported throughout, consistent with the
generalized Black-Scholes-Merton pricing in :mod:`derivflow.core.pricing_engine`.
"""

from typing import Optional, Union, List, Dict, Any

import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq

from ..core.pricing_engine import BlackScholesAnalytical

# Solver configuration constants
_SIGMA_MIN = 1e-6      # lower bracket for volatility
_SIGMA_MAX = 5.0       # upper bracket (500% vol) for volatility
_VEGA_FLOOR = 1e-10    # below this vega, Newton steps are unreliable

_bs = BlackScholesAnalytical()


def _vega_raw(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Raw Black-Scholes vega (per unit volatility, NOT scaled per 1%)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float,
                         option_type: str, q: float):
    """Return (lower, upper) no-arbitrage price bounds for a European option."""
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option_type.lower() == 'call':
        return max(disc_S - disc_K, 0.0), disc_S
    else:
        return max(disc_K - disc_S, 0.0), disc_K


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       option_type: str = 'call', q: float = 0.0,
                       tol: float = 1e-8, max_iter: int = 100,
                       initial_guess: Optional[float] = None,
                       on_fail: str = 'nan') -> float:
    """
    Solve for the Black-Scholes-Merton implied volatility of a single option.

    Parameters:
    -----------
    price : float
        Observed (market) option price.
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry (in years).
    r : float
        Risk-free rate.
    option_type : str
        'call' or 'put'.
    q : float
        Continuous dividend yield (default 0.0).
    tol : float
        Absolute price tolerance for convergence.
    max_iter : int
        Maximum iterations for Newton / Brent.
    initial_guess : float, optional
        Starting volatility for Newton-Raphson. If None, the
        Brenner-Subrahmanyam approximation is used.
    on_fail : str
        'nan' (default) returns ``np.nan`` when no solution exists or the price
        is outside the no-arbitrage bounds; 'raise' raises ``ValueError``.

    Returns:
    --------
    float
        Implied volatility, or ``np.nan`` (when ``on_fail='nan'``) if it cannot
        be recovered.
    """
    def _fail(msg: str) -> float:
        if on_fail == 'raise':
            raise ValueError(msg)
        return np.nan

    # Degenerate / expired option: volatility is undefined.
    if T <= 0 or S <= 0 or K <= 0 or price is None or not np.isfinite(price):
        return _fail("Implied volatility undefined for non-positive T/S/K or invalid price")

    # No-arbitrage bounds (allow a tiny numerical tolerance at the edges).
    lower, upper = _no_arbitrage_bounds(S, K, T, r, option_type, q)
    eps = max(tol, 1e-12)
    if price < lower - eps or price > upper + eps:
        return _fail(
            f"Price {price} is outside no-arbitrage bounds [{lower:.6f}, {upper:.6f}]"
        )

    # Brenner-Subrahmanyam at-the-money approximation as the initial guess.
    if initial_guess is None:
        sigma = np.sqrt(2 * np.pi / T) * price / S
    else:
        sigma = initial_guess
    sigma = float(np.clip(sigma, 1e-4, _SIGMA_MAX))

    # Newton-Raphson refinement.
    for _ in range(max_iter):
        model_price = _bs.price(S, K, T, r, sigma, option_type, q)
        diff = model_price - price
        if abs(diff) < tol:
            return sigma
        vega = _vega_raw(S, K, T, r, sigma, q)
        if vega < _VEGA_FLOOR or not np.isfinite(vega):
            break
        sigma_new = sigma - diff / vega
        if not np.isfinite(sigma_new) or sigma_new <= _SIGMA_MIN or sigma_new >= _SIGMA_MAX:
            break
        sigma = sigma_new

    # Brent fallback for robustness.
    def objective(s: float) -> float:
        return _bs.price(S, K, T, r, s, option_type, q) - price

    try:
        return float(brentq(objective, _SIGMA_MIN, _SIGMA_MAX, xtol=tol, maxiter=max_iter))
    except (ValueError, RuntimeError):
        return _fail("Newton and Brent both failed to converge to an implied volatility")


def implied_volatility_vectorized(prices, S, K, T, r, option_type: str = 'call',
                                  q: float = 0.0, **kwargs) -> np.ndarray:
    """
    Vectorized implied-volatility solver over array-like inputs.

    Any of ``prices``, ``S``, ``K``, ``T``, ``r``, ``q`` may be scalars or
    array-likes; they are broadcast to a common shape and solved element-wise.
    ``option_type`` may be a single string or an array of strings.

    Returns an ``np.ndarray`` of implied volatilities (``np.nan`` where a value
    cannot be recovered). Remaining keyword arguments are forwarded to
    :func:`implied_volatility` (e.g. ``tol``, ``max_iter``, ``on_fail``).
    """
    kwargs.setdefault('on_fail', 'nan')  # never raise inside a vectorized sweep by default
    prices_a, S_a, K_a, T_a, r_a, q_a = np.broadcast_arrays(
        np.asarray(prices, dtype=float), np.asarray(S, dtype=float),
        np.asarray(K, dtype=float), np.asarray(T, dtype=float),
        np.asarray(r, dtype=float), np.asarray(q, dtype=float)
    )
    ot_a = np.broadcast_to(np.asarray(option_type, dtype=object), prices_a.shape)

    out = np.empty(prices_a.shape, dtype=float)
    for idx in np.ndindex(prices_a.shape):
        out[idx] = implied_volatility(
            float(prices_a[idx]), float(S_a[idx]), float(K_a[idx]),
            float(T_a[idx]), float(r_a[idx]),
            option_type=str(ot_a[idx]), q=float(q_a[idx]), **kwargs
        )
    return out


def implied_vol_from_chain(chain: Any, S: float, r: float, q: float = 0.0,
                           price_field: str = 'mid', **kwargs) -> List[Dict]:
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
        Which price to invert: 'mid' (default, (bid+ask)/2 with last_price
        fallback), 'last', 'bid', or 'ask'. Ignored for explicit dict/DataFrame
        rows that already carry a 'market_price'.

    Returns:
    --------
    list[dict]
        One row per option with keys: ``strike``, ``expiry`` (years),
        ``volatility`` (solved IV), ``market_price``, ``option_type``.
    """
    rows = _normalize_chain(chain, price_field)
    results: List[Dict] = []
    for row in rows:
        iv = implied_volatility(
            row['market_price'], S, row['strike'], row['expiry'], r,
            option_type=row['option_type'], q=q, on_fail='nan', **kwargs
        )
        results.append({
            'strike': row['strike'],
            'expiry': row['expiry'],
            'volatility': iv,
            'market_price': row['market_price'],
            'option_type': row['option_type'],
        })
    return results


def _mid_price(last_price, bid, ask):
    """Best available price: mid of bid/ask, else last price."""
    try:
        if bid and ask and bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
    except TypeError:
        pass
    return last_price


def _normalize_chain(chain: Any, price_field: str) -> List[Dict]:
    """Normalize supported chain inputs to a list of {strike, expiry, market_price, option_type}."""
    # MarketDataResult (duck-typed via options_data attribute)
    if hasattr(chain, 'options_data'):
        rows = []
        for opt in chain.options_data:
            expiry_years = _expiry_to_years(opt.expiry)
            if expiry_years is None or expiry_years <= 0:
                continue
            if price_field == 'last':
                mp = opt.last_price
            elif price_field == 'bid':
                mp = opt.bid
            elif price_field == 'ask':
                mp = opt.ask
            else:
                mp = _mid_price(opt.last_price, opt.bid, opt.ask)
            if mp is None or mp <= 0:
                continue
            rows.append({'strike': opt.strike, 'expiry': expiry_years,
                         'market_price': mp, 'option_type': opt.option_type})
        return rows

    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(chain, pd.DataFrame):
            chain = chain.to_dict('records')
    except ImportError:  # pragma: no cover
        pass

    # list of dicts
    rows = []
    for rec in chain:
        mp = rec.get('market_price')
        if mp is None:
            mp = _mid_price(rec.get('last_price'), rec.get('bid'), rec.get('ask'))
        if mp is None or mp <= 0:
            continue
        rows.append({
            'strike': rec['strike'],
            'expiry': rec['expiry'],
            'market_price': mp,
            'option_type': rec.get('option_type', 'call'),
        })
    return rows


def _expiry_to_years(expiry) -> Optional[float]:
    """Convert an expiry (already-numeric years, or a YYYY-MM-DD date string) to years."""
    if isinstance(expiry, (int, float)):
        return float(expiry)
    try:
        from datetime import datetime
        exp = datetime.strptime(str(expiry), '%Y-%m-%d')
        days = (exp - datetime.now()).days
        return max(days, 0) / 365.0
    except (ValueError, TypeError):
        return None
