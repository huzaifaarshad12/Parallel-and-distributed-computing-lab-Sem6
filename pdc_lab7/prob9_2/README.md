# Problem 9.2 — Partial Ordering of Blocks (OpenMP)

Relation: `A ⪯ B` iff every element of A ≤ every element of B. For sorted
blocks this reduces to `max(A) ≤ min(B)` → `A[k-1] ≤ B[0]`.

## Build / Run
```
make
OMP_NUM_THREADS=4 ./partial_order 100 20      # m=100 blocks, size k=20
```

## How OpenMP is used
- **Reflexivity**: `#pragma omp parallel for` with `reduction(&&:reflexive)` —
  each thread checks a disjoint set of blocks; results combined by logical AND.
- **Antisymmetry**: parallel `for` over the outer index of all `i<j` pairs,
  `schedule(dynamic)` to balance the triangular loop, `reduction(&&:...)`.
- **Transitivity**: parallel over `a`; inner `b,c` loops scan pairs. The
  reduction collapses each thread's verdict.

No critical sections are needed because each property uses a logical-AND
reduction over a read-only block array — there is no shared write contention,
which also avoids false sharing.

## Expected output
All three properties report `HOLDS` (true by construction). Sample:
```
Reflexive    : HOLDS
Antisymmetric: HOLDS
Transitive   : HOLDS
```
