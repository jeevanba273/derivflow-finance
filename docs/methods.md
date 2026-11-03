# Methods

This page states the model behind each function, its assumptions and conventions, and where
results should not be trusted. Formulas are written for a continuous dividend yield q;
Haug's cost of carry is b = r - q.

## Conventions

- Time to expiry T in years; for listed options `derivflow._dates.year_fraction_to_expiry`
  uses ACT/365 to 16:00 New York time on the expiry date.
- Rates r and q are continuously compounded and flat (no term structure).
- Volatility is annualised; one volatility point is 0.01.
- VaR horizons are trading days with 252 per year.
- Greeks: theta and color per calendar day, vega, rho, vanna and zomma per point, volga per
  point squared ([Greeks](user-guide/greeks.md)).

## Black-Scholes-Merton

Risk-neutral dynamics dS = (r - q) S dt + sigma S dW with constant r, q and sigma; European
exercise. Prices and the ten Greeks use the closed forms in Haug (2007), chapters 1 and 2.
Color is the calendar-time derivative of gamma,
Gamma (q + (r - q) d1 / (sigma sqrt(T)) + (1 - d1 d2) / (2T)).

Limitations: constant volatility (no smile), no discrete dividends, no early exercise.

## Implied volatility

The quote is mapped to its out-of-the-money equivalent through put-call parity and solved by a
safeguarded Newton iteration on the log price (Press et al., Numerical Recipes, section 9.4,
"rtsafe"), from the Manaster and Koehler (1982) starting point. No solution exists outside
the bounds max(F - K, 0) e^{-rT} < C < S e^{-qT} (and the put analogue). When the time value is
below 1e-12 of the price, the volatility cannot be resolved in double precision and the solver
raises instead of returning a number. Jaeckel (2015), "Let's Be Rational", is the reference
algorithm used in the cross-checks.

## Binomial tree

Cox, Ross and Rubinstein (1979): u = e^{sigma sqrt(dt)}, d = 1/u,
p = (e^{(r - q) dt} - d) / (u - d). American exercise compares continuation and intrinsic value
at every node. The European error is O(1/n) and oscillates with the parity of n. A
risk-neutral probability outside (0, 1) (possible with few steps and large rates) raises.

## Monte Carlo

Terminal prices are sampled exactly from the lognormal law. Antithetic variates and the
terminal-price control variate follow Glasserman, Monte Carlo Methods in Financial Engineering
(2003), sections 4.1 and 4.2. The control-variate coefficient is estimated from the same
sample, which adds a bias of order 1/n that is negligible at the path counts used here.

## Barrier options

Continuous monitoring, no dividends and no rebate: the Merton (1973) and Reiner and Rubinstein
(1991) closed forms in the A, B, C, D notation of Haug (2007), section 4.17.1. The no-touch
probability comes from the reflection principle for drifted Brownian motion. The Monte Carlo
pricer monitors the barrier at its time steps; Broadie, Glasserman and Kou (1997) give the
continuity correction exp(+-0.5826 sigma sqrt(dt)) that relates the two.

Limitations: no rebates, no double barriers, no dividend yield, no partial-time windows.

## Asian options

Average-price options with n equally spaced fixings at t_i = i T / n, today excluded.
Geometric averages are lognormal, so the price is a Black-type formula with
mean ln S + (r - sigma^2/2) T (n+1)/(2n) and variance sigma^2 T (n+1)(2n+1)/(6n^2); the
continuous limit is Kemna and Vorst (1990). Arithmetic averages are priced by Monte Carlo with
the geometric option as a control variate. No dividend yield and no average-strike options.

## Heston

Heston (1993) dynamics with variance v. Prices use the single-integral representation of
Lewis (2001) with the characteristic function in the form of Albrecher, Mayer, Schoutens and
Tistaert (2007), "The little Heston trap", integrated with adaptive quadrature. Simulation uses
Andersen's (2008) quadratic-exponential scheme or full-truncation Euler (Lord, Koekkoek and van
Dijk, 2010). Calibration is bounded least squares on prices.

Limitations: no jumps; parameters fitted to one date can be unstable; very short maturities
and extreme parameters make the integrand oscillate, so check prices there against a second
method.

## Volatility surfaces

Raw SVI (Gatheral, 2004) per expiry, fitted by least squares in total variance, with linear
interpolation of total variance in T at fixed strike. The RBF and linear surfaces interpolate
quotes on (ln K, T). No method imposes absence of arbitrage. `check_arbitrage` tests a single
smile for vertical-spread and butterfly arbitrage on Black-Scholes call prices; it does not
test calendar arbitrage.

## Portfolio risk

Delta-normal VaR and expected shortfall (J.P. Morgan/Reuters, RiskMetrics Technical Document,
1996), component VaR by Euler allocation (Jorion, Value at Risk, chapter 7), Monte Carlo VaR by
full Black-Scholes-Merton revaluation under jointly lognormal underlyings, and closed-form
normal and Student-t VaR and expected shortfall (McNeil, Frey and Embrechts, Quantitative Risk
Management, 2015, section 2.3).

Limitations: volatilities and correlations are inputs; no volatility, rate or dividend risk in
the VaR; no fat tails in the portfolio simulation; scenario analysis moves spots only.

## References

- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. Wilmott Magazine.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11(3).
- Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy 81(3).
- Broadie, M., Glasserman, P. and Kou, S. (1997). A continuity correction for discrete barrier options. Mathematical Finance 7(4).
- Cox, J., Ross, S. and Rubinstein, M. (1979). Option pricing: a simplified approach. Journal of Financial Economics 7(3).
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization. Global Derivatives conference.
- Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering. Springer.
- Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas, 2nd ed. McGraw-Hill.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. Review of Financial Studies 6(2).
- Hull, J. Options, Futures, and Other Derivatives. Pearson.
- Jaeckel, P. (2015). Let's be rational. Wilmott Magazine.
- Jorion, P. Value at Risk, 3rd ed. McGraw-Hill.
- Kemna, A. and Vorst, A. (1990). A pricing method for options based on average asset values. Journal of Banking and Finance 14(1).
- Lewis, A. (2001). A simple option formula for general jump-diffusion and other exponential Levy processes. Working paper.
- Lord, R., Koekkoek, R. and van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models. Quantitative Finance 10(2).
- Manaster, S. and Koehler, G. (1982). The calculation of implied variances from the Black-Scholes model. Journal of Finance 37(1).
- McNeil, A., Frey, R. and Embrechts, P. (2015). Quantitative Risk Management, 2nd ed. Princeton University Press.
- Merton, R. (1973). Theory of rational option pricing. Bell Journal of Economics 4(1).
- Reiner, E. and Rubinstein, M. (1991). Breaking down the barriers. Risk 4(8).
