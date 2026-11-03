# Benchmarks

## Method

`benchmarks/run_benchmarks.py` times each operation with `time.perf_counter`: one warm-up
run, then seven repetitions, reported as the median and the [min, max] range. All
implementations get identical inputs on the same machine, single-threaded, in one Python
process. The baselines are QuantLib (C++ with Python bindings), py_vollib (Python
implementation of Jaeckel's "Let's Be Rational") and a textbook NumPy/SciPy formula.

Not every comparison is like-for-like, and the notes under each table say how:

- QuantLib is driven from Python one option at a time, so its rows include the cost of the
  Python bindings and of QuantLib's observer updates; QuantLib called from C++ is faster.
- derivflow computes ten Greeks per call, QuantLib five in the benchmark.
- QuantLib's "crr" tree uses a different up-probability from derivflow's textbook CRR tree, so
  the two give different prices at the same step count.
- Heston: derivflow integrates with SciPy's adaptive quadrature, QuantLib with Gauss-Laguerre.

Where derivflow is slower, the table shows it: QuantLib's tree is about twice as fast at 2000
steps and its Heston engine about twice as fast per price.

Reproduce with:

```bash
pip install -e ".[validation]"
python benchmarks/run_benchmarks.py
```

The numbers depend on the machine, the Python build and the library versions recorded in the
first table.

## Results

```{include} ../benchmarks/results.md
:start-line: 2
```
