# derivflow

derivflow is a Python library for pricing equity-style options and measuring the risk of
small option books. It covers the Black-Scholes-Merton closed form and Greeks, implied
volatility, Cox-Ross-Rubinstein trees with early exercise, Monte Carlo with variance
reduction, single-barrier and Asian options, the Heston model, volatility surfaces and
portfolio VaR. It is meant for students, researchers and practitioners who want readable
NumPy/SciPy code whose results are checked against QuantLib, py_vollib and textbook values.
It is not a trading system and does not handle term structures, calendars or day-count
conventions beyond ACT/365.

```{toctree}
:maxdepth: 2
:caption: Using derivflow

getting-started
user-guide/index
api/index
```

```{toctree}
:maxdepth: 1
:caption: Background

methods
validation
benchmarks
changelog
contributing
```

For research and education; not investment advice.
