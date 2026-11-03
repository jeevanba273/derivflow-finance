# Benchmarks

## Method

`benchmarks/run_benchmarks.py` times each operation with `time.perf_counter`: one warm-up
run, then seven repetitions, reported as the median and the [min, max] range. All
implementations get identical inputs on the same machine, single-threaded, in one Python
process. The baselines are QuantLib (C++ with Python bindings), py_vollib (Python
implementation of Jaeckel's "Let's Be Rational") and a textbook NumPy/SciPy formula.

Not every comparison is like-for-like, and the notes under each table say how:

- QuantLib is driven from Python one option at a time, so its rows include the cost of the
  Python bindings and of QuantLib's observer updates.
- derivflow computes ten Greeks per call, QuantLib five in the benchmark.
- QuantLib's "crr" tree uses a different up-probability from derivflow's textbook CRR tree, so
  the two give different prices at the same step count.
- Heston: derivflow integrates with SciPy's adaptive quadrature (QUADPACK), QuantLib with
  adaptive Gauss-Lobatto at a relative tolerance of 1e-12.

Where derivflow is slower, the tables show it: QuantLib's tree takes less than half the time
at 2000 steps and its Heston engine a little over half the time per price.

Reproduce with:

```bash
pip install -e ".[validation]"
python benchmarks/run_benchmarks.py
```

The numbers depend on the machine, the Python build and the library versions recorded in the
first table, and on what else the machine is doing: throughput drops when other work runs at the
same time, so compare runs made on an idle machine.

## Results

```{include} ../benchmarks/results.md
:start-line: 2
```
