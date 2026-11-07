"""Batches of simulated paths, so that path-dependent Monte Carlo uses bounded
memory whatever the number of paths.

The generator fills arrays in row order, so drawing the paths batch by batch
consumes the same random numbers in the same order as drawing them at once,
and every per-path result (cumulative sums, averages, barrier checks) is
computed along one row; batched results are identical to unbatched ones.
"""

from __future__ import annotations

from collections.abc import Iterator

PATH_BUDGET = 2_000_000
"""Values (paths x time steps) per batch: about 16 MB for each float64 array."""


def path_batches(num_paths: int, num_steps: int) -> Iterator[tuple[int, int]]:
    """Yield (start, stop) row ranges covering ``num_paths`` paths, each with
    at most ``PATH_BUDGET`` values and at least one path."""
    size = max(1, PATH_BUDGET // max(1, num_steps))
    for start in range(0, num_paths, size):
        yield start, min(start + size, num_paths)
