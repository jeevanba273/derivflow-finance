# Mathematical Background

This page summarises the mathematics behind the models implemented in
DERIVFLOW-FINANCE. It is a concise reference, not a textbook treatment.

## Black-Scholes-Merton with dividends

Under the risk-neutral measure, the underlying follows a geometric Brownian
motion with continuous dividend yield $q$:

$$dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t.$$

The price of a European call is

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),$$

and the corresponding put is

$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),$$

where $N(\cdot)$ is the standard normal CDF and

$$d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}},
\qquad d_2 = d_1 - \sigma \sqrt{T}.$$

Setting $q = 0$ recovers the classic Black-Scholes formula.

## The Greeks

The Greeks are partial derivatives of the option value $V$ with respect to its
inputs. The first-order analytical Greeks (for a dividend-paying underlying) are:

- **Delta** $\Delta = \dfrac{\partial V}{\partial S}$, with
  $\Delta_{\text{call}} = e^{-qT} N(d_1)$ and
  $\Delta_{\text{put}} = e^{-qT}(N(d_1) - 1)$.
- **Gamma** $\Gamma = \dfrac{\partial^2 V}{\partial S^2}
  = \dfrac{e^{-qT}\,\phi(d_1)}{S \sigma \sqrt{T}}$, where $\phi$ is the normal PDF.
- **Vega** $\nu = \dfrac{\partial V}{\partial \sigma}
  = S e^{-qT} \phi(d_1) \sqrt{T}$.
- **Theta** $\Theta = \dfrac{\partial V}{\partial t}$ (time decay).
- **Rho** $\rho = \dfrac{\partial V}{\partial r}$.

Second- and third-order sensitivities (Volga, Vanna, Speed, Zomma, Color) are
also provided analytically.

## Barrier options

Barrier options knock in or out when the underlying touches a barrier level $H$.
DERIVFLOW prices the four single-barrier European variants
(up/down $\times$ in/out) in closed form using the **reflection principle** and
method of images. The implementation follows the standard
**Reiner-Rubinstein** decomposition into building blocks $A$, $B$, $C$, $D$
parameterised by $\phi$ (call $=+1$, put $=-1$) and $\eta$ (down barrier $=+1$,
up barrier $=-1$). The reflection term carries the factor $(H/S)^{2\mu}$ with
$\mu = (r - \tfrac{1}{2}\sigma^2)/\sigma^2$, which reflects paths across the
barrier to enforce the knock condition. In-out parity links each knock-in price
to the corresponding knock-out price plus the vanilla option.

## Asian options (geometric closed form)

For a **geometric-average** Asian option, the average of a lognormal process is
itself lognormal, so a Black-Scholes-style closed form exists. With continuous
averaging the effective volatility and drift are reduced:

$$\sigma_G = \frac{\sigma}{\sqrt{3}},
\qquad r_G = \frac{r - \tfrac{1}{2}\sigma^2}{2} + \frac{\sigma^2}{6}.$$

Substituting $\sigma_G$ and $r_G$ into the Black-Scholes formula prices the
geometric Asian option. **Arithmetic-average** Asians have no closed form and are
priced by Monte Carlo, using the geometric Asian as a control variate to reduce
variance.

## Heston stochastic volatility

The Heston model lets variance $V_t$ follow its own mean-reverting diffusion,
correlated with the asset:

$$dS_t = r S_t \, dt + \sqrt{V_t}\, S_t \, dW_1,$$
$$dV_t = \kappa(\theta - V_t)\, dt + \sigma \sqrt{V_t}\, dW_2,
\qquad dW_1\, dW_2 = \rho\, dt.$$

Here $\kappa$ is the mean-reversion speed, $\theta$ the long-run variance,
$\sigma$ the volatility of volatility, and $\rho$ the correlation (the negative
$\rho$ producing the leverage effect and skew). The **Feller condition**

$$2 \kappa \theta > \sigma^2$$

ensures the variance process stays strictly positive. DERIVFLOW prices Heston
options via Fourier inversion of the characteristic function and via
Euler-Maruyama Monte Carlo, and can calibrate parameters to market quotes.

## Implied volatility

Implied volatility is the value of $\sigma$ that reproduces an observed market
price under Black-Scholes-Merton. DERIVFLOW solves

$$\text{BS}(\sigma) - P_{\text{market}} = 0$$

with a fast **Newton-Raphson** iteration

$$\sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - P_{\text{market}}}{\nu(\sigma_n)},$$

seeded by the Brenner-Subrahmanyam approximation. When Newton struggles
(near-zero vega, deep in/out-of-the-money quotes, or prices near the
no-arbitrage bounds) it falls back to the robust **Brent** bracketing method.

## Value at Risk

Two VaR methodologies are implemented:

- **Parametric (delta-normal) VaR** assumes portfolio returns are normal. At
  confidence level $c$,

  $$\text{VaR} = V\, \sigma_p \sqrt{h}\; z_c,$$

  where $V$ is portfolio value, $\sigma_p$ its volatility, $h$ the horizon, and
  $z_c = N^{-1}(c)$. Expected shortfall uses the conditional tail expectation.

- **Monte Carlo VaR** simulates correlated shocks to each position, reprices the
  portfolio, and reads VaR as the appropriate quantile of the simulated P&L
  distribution; expected shortfall is the mean loss beyond that quantile.
