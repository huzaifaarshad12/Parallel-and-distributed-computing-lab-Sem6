# Problem 9.8 — Block-Based Bitonic Sort (MPI)

p processes, each holding n/p elements. The bitonic network runs over the p
blocks; each compare-exchange becomes a compare-split (exchange block, merge,
keep half). Blocks stay locally sorted.

## Build / Run
```
make
mpirun --oversubscribe -np 4 ./block_bitonic 32      # p=4, n=32
make run                                             # all p in {2,4,8}, n in {16,32,64}
```

## Measured
| n  | p | k=n/p | compare-splits | comm bytes | sorted |
|----|---|------:|---------------:|-----------:|:------:|
| 16 | 2 | 8     | 2              | 64         | yes |
| 32 | 2 | 16    | 2              | 128        | yes |
| 64 | 2 | 32    | 2              | 256        | yes |
| 16 | 4 | 4     | 12             | 192        | yes |
| 32 | 4 | 8     | 12             | 384        | yes |
| 64 | 4 | 16    | 12             | 768        | yes |
| 16 | 8 | 2     | 48             | 384        | yes |
| 32 | 8 | 4     | 48             | 768        | yes |
| 64 | 8 | 8     | 48             | 1536       | yes |

## Comparison with element-wise version
- Block version compare-splits depend only on **p**: `p·log²p·(approx)/2`
  → 2, 12, 48 for p = 2, 4, 8.
- The element-wise version needs **n** processes and `n·log₂n·(log₂n+1)/2`
  compare-exchanges (e.g. 160 for n=16, 1344 for n=64) — far more network
  operations.
- Trade-off: the block version does more *local* work per step (an O(k) merge)
  but vastly fewer *network* operations and uses p << n processors, which is
  the realistic regime.

See handwritten theory for the inductive correctness proof over phases.
