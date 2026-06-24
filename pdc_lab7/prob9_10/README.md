# Problem 9.10 — Block-Based Odd-Even Transposition Sort (MPI)

p processes in a linear array, each holding n/p elements. Runs p phases
alternating even/odd neighbour compare-splits. Lower rank keeps low half.

## Build / Run
```
make
mpirun --oversubscribe -np 4 ./oddeven 32
make run                                     # all p in {2,4,8}, n in {16,32,64}
```

## Measured
| n  | p | k  | phases | messages | comm bytes | sorted |
|----|---|---:|-------:|---------:|-----------:|:------:|
| 16 | 2 | 8  | 2      | 2        | 64         | yes |
| 32 | 2 | 16 | 2      | 2        | 128        | yes |
| 64 | 2 | 32 | 2      | 2        | 256        | yes |
| 16 | 4 | 4  | 4      | 12       | 192        | yes |
| 32 | 4 | 8  | 4      | 12       | 384        | yes |
| 64 | 4 | 16 | 4      | 12       | 768        | yes |
| 16 | 8 | 2  | 8      | 56       | 448        | yes |
| 32 | 8 | 4  | 8      | 56       | 896        | yes |
| 64 | 8 | 8  | 8      | 56       | 1792       | yes |

## Comparison with block bitonic (Problem 9.8)
| p | odd-even msgs | bitonic msgs |
|---|--------------:|-------------:|
| 2 | 2             | 2            |
| 4 | 12            | 12           |
| 8 | 56            | 48           |

- For small p the two are similar. As p grows, **odd-even needs p phases**
  (O(p) compare-splits) while **block bitonic needs O(log²p)** — so bitonic
  wins at larger p (56 vs 48 messages at p=8, widening further beyond).
- Odd-even is simpler (only nearest-neighbour communication on a linear array),
  which can help on a physical linear/mesh network where bitonic's long-range
  partners cost extra hops. So the "faster" method depends on the topology:
  bitonic by operation count, odd-even by locality.

See handwritten theory for the invariant-based correctness proof (after each
phase the block sequence moves strictly closer to globally sorted).
