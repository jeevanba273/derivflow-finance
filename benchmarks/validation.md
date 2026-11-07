# DerivFlow Finance validation results

Command: `python benchmarks/validation_report.py` (derivflow-finance 0.4.0, QuantLib 1.44, numpy 2.5.4)

## Published reference values

| case | reference | DerivFlow Finance | matches to the quoted digits |
|---|---|---|---|
| Hull: call, S=42 K=40 T=0.5 r=10% sigma=20% | 4.76 | 4.759422 | yes |
| Hull: put, same inputs | 0.81 | 0.808599 | yes |
| Hull: American put, 5-step tree, S=K=50 r=10% sigma=40% T=5/12 | 4.49 | 4.488459 | yes |
| Haug: call, S=60 K=65 T=0.25 r=8% sigma=30% | 2.1334 | 2.133368 | yes |
| Haug: put with q, S=100 K=95 T=0.5 r=10% q=5% sigma=20% | 2.4648 | 2.464788 | yes |
| Haug: call delta, b=0, S=105 K=100 T=0.5 r=10% sigma=36% | 0.5946 | 0.594629 | yes |
| Haug: put delta, same inputs | -0.3566 | -0.356601 | yes |
| Haug: gamma, S=55 K=60 T=0.75 r=b=10% sigma=30% | 0.0278 | 0.027821 | yes |
| Haug: put theta per year, S=430 K=405 T=1/12 r=7% q=5% sigma=20% | -31.1924 | -31.192350 | yes |
| Haug: call rho per unit rate, S=72 K=75 T=1 r=b=9% sigma=19% | 38.7325 | 38.732505 | yes |
| Lewis: Heston calls K=80..120 (5 strikes), max abs difference | 15+ digits | 1.1e-12 | yes |

Put-call parity C - P = S e^(-qT) - K e^(-rT): max abs error 5.7e-14 over 324 parameter sets.

## QuantLib

Maximum absolute difference; maturities are whole days on Actual/365 Fixed.

| quantity | QuantLib engine | cases | max abs difference |
|---|---|---|---|
| European price | AnalyticEuropeanEngine | 144 | 3.9e-14 |
| European delta (QuantLib units) | AnalyticEuropeanEngine | 144 | 2.2e-15 |
| European gamma (QuantLib units) | AnalyticEuropeanEngine | 144 | 6.9e-17 |
| European vega (QuantLib units) | AnalyticEuropeanEngine | 144 | 4.6e-14 |
| European theta (QuantLib units) | AnalyticEuropeanEngine | 144 | 1.8e-13 |
| European rho (QuantLib units) | AnalyticEuropeanEngine | 144 | 1.1e-13 |
| American option, 2000-step tree | FdBlackScholesVanillaEngine 2000x2000 | 5 | 5.8e-04 |
| Barrier, closed form | AnalyticBarrierEngine | 48 | 2.1e-14 |
| Geometric Asian, continuous | AnalyticContinuousGeometricAveragePriceAsianEngine | 6 | 1.4e-14 |
| Geometric Asian, 12 fixings | AnalyticDiscreteGeometricAveragePriceAsianEngine | 6 | 1.4e-14 |
| Arithmetic Asian, 12 fixings, Monte Carlo (200k paths each) | MCDiscreteArithmeticAPEngine | 1 | 2.5e-04 (0.2 combined s.e.) |
| Heston, analytic (3 parameter sets incl. one violating Feller) | AnalyticHestonEngine | 27 | 2.1e-10 |

## Monte Carlo confidence intervals

ATM European call (S=K=100, T=1, r=5%, sigma=20%), 20,000 draws, seeds 0 to 399. z = (estimate - closed form) / standard error should have mean 0 and standard deviation 1.

| method | 95% CI coverage | mean z | std of z |
|---|---|---|---|
| plain | 95.0% | -0.088 | 0.976 |
| antithetic | 94.0% | -0.036 | 1.006 |
| control variate | 94.0% | -0.032 | 1.022 |

## py_vollib

| quantity | cases | max abs difference |
|---|---|---|
| Implied volatility vs Let's Be Rational (quotes with vega > 1e-6 S) | 128 | 4.1e-12 |
