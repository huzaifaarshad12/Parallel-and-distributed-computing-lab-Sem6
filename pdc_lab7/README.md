# Parallel & Distributed Computing — Lab 07 (Chapter 9, Problems 9.1–9.10)

Code archive. One directory per problem; each has source, a `Makefile`, and a
`README.md` with build/run/results. The theory parts are submitted separately
(handwritten, per the instructor's instruction).

## Requirements
- MPI (`mpicc`/`mpirun`) — OpenMPI or MPICH (problems 9.1, 9.4–9.10)
- GCC with OpenMP (`-fopenmp`) (problems 9.2, 9.3)
- Python 3 + matplotlib + numpy (only for 9.9's efficiency plot)

## Build everything
```
for d in prob9_* ; do (cd "$d" && make) ; done
```

## Note on running MPI on one machine
Several problems request more processes than you have physical cores
(e.g. 16 or 32). Add `--oversubscribe`:
```
mpirun --oversubscribe -np 16 ./bitonic_hypercube
```
If your environment runs MPI as root, also add `--allow-run-as-root`.

## Problem index
| Dir        | Topic                                   | Model        |
|------------|-----------------------------------------|--------------|
| prob9_1    | Compare-split: merge-split vs pairwise  | MPI (2 proc) |
| prob9_2    | Partial ordering of blocks              | OpenMP       |
| prob9_3    | Bitonic split properties (3 types)      | C/OpenMP     |
| prob9_4    | Bitonic sort with n/2 processes         | MPI          |
| prob9_5    | Hypercube bitonic merge                  | MPI          |
| prob9_6    | Full hypercube bitonic sort (Alg 9.1)   | MPI          |
| prob9_7    | Mesh bitonic comm-cost simulation       | MPI          |
| prob9_8    | Block-based bitonic sort                | MPI          |
| prob9_9    | Bitonic sort on a ring + isoefficiency  | MPI + Python |
| prob9_10   | Block-based odd-even transposition sort | MPI          |

All practical programs were compiled and verified to PASS / produce SORTED
output. Replace the sample timing tables in each README with measurements from
your own machine for the report.
