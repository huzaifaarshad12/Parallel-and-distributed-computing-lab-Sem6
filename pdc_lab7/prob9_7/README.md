# Problem 9.7 — Mesh Bitonic Sort: Communication-Cost Simulation (MPI)

Simulates (does NOT sort) the bitonic-sort communication pattern over a
sqrt(n) x sqrt(n) mesh for three mappings, totalling hop counts and applying
the store-and-forward (SAF) and cut-through (CT) cost models.

## Build / Run
```
make
mpirun --oversubscribe -np 4 ./mesh_cost 16
mpirun --oversubscribe -np 4 ./mesh_cost 64
mpirun --oversubscribe -np 4 ./mesh_cost 256
```
(`t_s = 1.0`, `t_w = 0.1` set in the source; change to match your formulas.)

## Mappings
- **row-major**: index = hypercube label, (row,col) = (idx/c, idx%c)
- **snakelike**: as row-major but odd rows are reversed
- **shuffled**: bit-reversal (perfect shuffle) of the label before row-major

## Cost models
- SAF: each message costs `t_s + t_w * hops` (Manhattan distance)
- CT : each message costs `t_s + t_w` (per the problem's simplification)

## Measured (t_s=1, t_w=0.1)
| n   | mapping   | msgs | totalHops | SAF     | CT      |
|-----|-----------|-----:|----------:|--------:|--------:|
| 16  | row-major | 80   | 112       | 91.20   | 88.00   |
| 16  | snakelike | 80   | 144       | 94.40   | 88.00   |
| 16  | shuffled  | 80   | 128       | 92.80   | 88.00   |
| 64  | row-major | 672  | 1376      | 809.60  | 739.20  |
| 64  | snakelike | 672  | 1760      | 848.00  | 739.20  |
| 64  | shuffled  | 672  | 1760      | 848.00  | 739.20  |
| 256 | row-major | 4608 | 14336     | 6041.60 | 5068.80 |
| 256 | snakelike | 4608 | 18432     | 6451.20 | 5068.80 |
| 256 | shuffled  | 4608 | 20224     | 6630.40 | 5068.80 |

## Validating the theory
- **CT is identical across mappings** at each n, because cut-through cost
  depends only on message count, not distance — matching the theoretical
  result that cut-through removes the per-hop distance penalty.
- **SAF depends on the mapping** through total hop count. The dominant SAF term
  scales like Θ(sqrt(n)·log²n) per the textbook (each of the
  Θ(log²n) stages can require messages of length up to Θ(sqrt(n)) hops on the
  mesh). The simulated hop totals grow accordingly as n increases.
- Row-major keeps low-dimension partners physically close, so it has the
  fewest hops here; snakelike/shuffled trade locality differently.

See the handwritten theory for the exact leading constants of the
SAF runtime under each mapping and the O(.) expressions.
