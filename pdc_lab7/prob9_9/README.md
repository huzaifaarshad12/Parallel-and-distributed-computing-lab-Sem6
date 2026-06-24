# Problem 9.9 — Bitonic Sort on a Ring (MPI)

`ring_bitonic.c` — n=16, compares total ring-hop cost of three label→position
mappings (Gray-code optimal, consecutive, random) using a periodic 1-D
Cartesian topology.
`block_ring.c` — block version on the ring; fix p, vary k = n/p for weak scaling.
`plot_isoefficiency.py` — drives weak-scaling runs and plots efficiency vs p.

## Build / Run
```
make
mpirun --oversubscribe -np 16 ./ring_bitonic
mpirun --oversubscribe -np 4 ./block_ring 8
python3 plot_isoefficiency.py        # produces efficiency_vs_p.png
```
> If your launcher runs as root, add `--allow-run-as-root` to the mpirun calls
> (and inside the Python script's subprocess list).

## Mapping comparison (n=16, measured)
| mapping        | messages | total ring hops |
|----------------|---------:|----------------:|
| optimal (Gray) | 80       | 176             |
| consecutive    | 80       | 208             |
| random         | 80       | 382             |

The Gray-code mapping minimises hops because hypercube neighbours
(labels differing in one bit) land on adjacent or near-adjacent ring positions.
Random scatters partners around the ring, roughly doubling the cost.

## Weak-scaling / isoefficiency (sample, k=2048 per process)
| p  | n     | Tp (s)   | E     |
|----|-------|---------:|------:|
| 2  | 4096  | 0.000233 | ~1.07 |
| 4  | 8192  | 0.000784 | 0.383 |
| 8  | 16384 | 0.002020 | 0.147 |
| 16 | 32768 | 0.006370 | 0.047 |

Efficiency falls as p grows (see `efficiency_vs_p.png`): the ring's diameter is
Θ(p), so communication cost grows much faster than on a hypercube.

## Theory summary (full derivation handwritten)
- Best mapping: embed the bitonic indices via Gray code so single-bit-difference
  partners are ring-adjacent; parallel time is dominated by the long-distance
  compare-exchanges that still cost up to Θ(p) hops, giving ring parallel time
  Θ(p) per such stage.
- Largest cost-optimal p: on a ring the communication overhead forces
  p = Θ(log n) (so that the Θ(p·…) communication does not dominate the
  Θ(n log n) sequential work) — see handwritten analysis for the exact bound.
- Isoefficiency: substantially worse than the hypercube's Θ(p log²p); the ring
  diameter pushes it toward an exponential-in-p problem-size growth to hold E
  fixed.
