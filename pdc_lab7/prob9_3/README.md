# Problem 9.3 — Bitonic Split Properties (C/OpenMP)

Generates the three bitonic input types (a: increasing-decreasing,
b: increasing-to-pivot-then-decreasing, c: cyclic shift), performs the
bitonic split in parallel, and verifies both properties.

## Build / Run
```
make
OMP_NUM_THREADS=4 ./bitonic_split 16     # n must be a power of 2 (16, 32, 64...)
```

## What it checks
For split into `s1[i]=min(s[i],s[i+n/2])`, `s2[i]=max(...)`:
1. **s1 and s2 are bitonic** — `is_bitonic()` counts cyclic direction changes
   (a bitonic ring has ≤ 2).
2. **max(s1) ≤ min(s2)** — every low-half element ≤ every high-half element.

## OpenMP parallelism
- The `n/2` independent compare operations run under `#pragma omp parallel for`.
- The property-2 check uses `reduction(max:...)` and `reduction(min:...)`.

## Parallel efficiency note
The split is embarrassingly parallel: `n/2` independent comparisons, no
dependencies, so ideal speedup is limited only by memory bandwidth and the
small problem size. For n=16–64 the OpenMP overhead dominates; the parallel
form pays off when n is large (thousands+), where the O(n) compare work is
spread across cores with linear speedup.

Sample output for n=16 is in `sample_output.txt`.
