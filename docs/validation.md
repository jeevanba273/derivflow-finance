# Validation

The tables below were produced by

```bash
pip install -e ".[validation]"
python benchmarks/validation_report.py
```

They report measured differences. The same comparisons are asserted with tolerances in the
test suite: `tests/test_references.py` (published values and put-call parity),
`tests/test_crosscheck_quantlib.py` and `tests/test_crosscheck_vollib.py` (skipped when the
reference library is not installed), `tests/test_greeks/test_finite_difference.py` (every Greek
against central differences), and the Monte Carlo tests, which require the closed-form price
to lie within the reported confidence interval.

Two comparisons are not exact by construction: the American tree against QuantLib's finite
difference engine compares two different discretisations, and the arithmetic Asian option
compares two Monte Carlo estimates, so its difference is shown in combined standard errors.

```{include} ../benchmarks/validation.md
:start-line: 2
```
