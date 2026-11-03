# Benchmarks

## Method

`benchmarks/run_benchmarks.py` times with `time.perf_counter`. Implementations compared in one
table run in the same loop: after one warm-up call each, every repetition runs each of them
once, in an order that rotates between repetitions, 15 repetitions in all. Background load
therefore affects them alike. Tables give the median and the [min, max] range of rates or
times, and the per-repetition ratio of each implementation's time to derivflow's (below 1
means faster than derivflow). All implementations get identical inputs, single-threaded, in one
Python process. The script also records the total CPU load during the run and the busiest other
processes (with psutil).

The baselines are QuantLib (C++ with Python bindings), py_vollib (Python implementation of
Jaeckel's "Let's Be Rational") and the Black-Scholes-Merton formula written directly in
NumPy/SciPy, in two forms: with `scipy.special.ndtr`, the same normal CDF derivflow uses, and
in the textbook form with `scipy.stats.norm.cdf`.

Not every comparison is like-for-like, and the notes under each table say how:

- The plain `ndtr` formula does less work than `black_scholes_price`: no input checks, no
  broadcasting of six inputs, no expiry handling and no call/put branch.
- QuantLib is driven from Python one option at a time, so its rows include the cost of the
  Python bindings and of QuantLib's observer updates.
- derivflow computes ten Greeks per call, QuantLib five in the benchmark.
- QuantLib's "crr" tree uses a different up-probability from derivflow's textbook CRR tree, so
  the two give different prices at the same step count.
- Heston: derivflow integrates with SciPy's adaptive quadrature (QUADPACK), QuantLib with
  adaptive Gauss-Lobatto at a relative tolerance of 1e-12.

Where derivflow is slower, the tables show it: the plain `ndtr` formula takes about 0.71 of
derivflow's time at a million options; QuantLib's tree about half at 2000 steps and about a
tenth at 100 steps; its Heston engine about half per price.

The published run was made with OneDrive syncing in the background (total CPU 16-31%, median
19%, recorded at the top of the results). Absolute rates may be lower than on an idle machine;
the ratios are measured side by side within each repetition.

Reproduce with:

```bash
pip install -e ".[validation]"
python benchmarks/run_benchmarks.py
```

The numbers depend on the machine, the Python build, the library versions recorded in the first
table and the background load recorded next to them.

## Results

```{include} ../benchmarks/results.md
:start-line: 2
```
