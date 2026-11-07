# Binomial trees

`BinomialTree(steps)` is the Cox-Ross-Rubinstein (1979) tree: u = exp(sigma sqrt(dt)),
d = 1/u and risk-neutral probability p = (exp((r - q) dt) - d) / (u - d). With
`american=True` each node takes the larger of the continuation and the exercise value.

## Hull's American put

Hull's five-step example (S = K = 50, r = 10%, sigma = 40%, five months) is worth 4.49.

```python
from derivflow_finance import BinomialTree, black_scholes_price

print(round(BinomialTree(5).price(50, 50, 5 / 12, 0.10, 0.40, "put", american=True), 2))
```

```text
4.49
```

## Convergence

The European tree price converges to Black-Scholes-Merton with an error of order 1/steps that
oscillates between odd and even step counts.

```python
bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
for steps in (50, 200, 800, 3200):
    tree = BinomialTree(steps)
    eu = tree.price(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    am = tree.price(100.0, 100.0, 1.0, 0.05, 0.2, "put", american=True)
    print(f"{steps:5d} steps: European error {eu - bs:+.6f}, American {am:.6f}")
```

```text
   50 steps: European error -0.039892, American 6.073728
  200 steps: European error -0.009992, American 6.086383
  800 steps: European error -0.002499, American 6.089402
 3200 steps: European error -0.000625, American 6.090133
```

## Early exercise

Without dividends an American call is never exercised early, so it equals the European call;
with a dividend yield it can be worth more.

```python
tree = BinomialTree(500)
for q in (0.0, 0.08):
    am = tree.price(100.0, 90.0, 1.0, 0.03, 0.2, "call", american=True, q=q)
    eu = tree.price(100.0, 90.0, 1.0, 0.03, 0.2, "call", q=q)
    print(f"q={q:.2f}: American {am:.6f}, European {eu:.6f}")
```

```text
q=0.00: American 15.428434, European 15.428434
q=0.08: American 11.381408, European 9.910723
```

The backward induction is vectorised over the nodes of each time step. A 2000-step American
price takes about 28 ms on the benchmark machine, against about 15 ms for QuantLib's
C++ tree; see [benchmarks](../benchmarks.md). Against QuantLib's finite-difference engine the
2000-step tree differs by at most 6e-4 on the five validation cases.
