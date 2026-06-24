# Problem 9.4 — Bitonic Sort with n/2 Processes (MPI)

`bitonic_np2.c` — p = n/2 processes, each holds a block of **2** elements;
every compare-exchange becomes a **compare-split** between 2-element blocks.
`bitonic_n.c` — baseline: n processes, 1 element each (standard).

## Build / Run
```
make
mpirun --oversubscribe -np 8  ./bitonic_np2     # n=16, p=8
mpirun --oversubscribe -np 16 ./bitonic_n       # n=16 baseline
mpirun --oversubscribe -np 16 ./bitonic_np2     # n=32, p=16
mpirun --oversubscribe -np 32 ./bitonic_n       # n=32 baseline
```

## Measured (oversubscribed on one node — illustrative)
| n  | version       | p  | k | time (s)  | messages |
|----|---------------|----|---|-----------|----------|
| 16 | n/2-proc block| 8  | 2 | 0.000263  | 48       |
| 16 | n-proc base   | 16 | 1 | 0.002966  | 160      |
| 32 | n/2-proc block| 16 | 2 | 0.000745  | 160      |
| 32 | n-proc base   | 32 | 1 | 0.002483  | 480      |

Speedup S = T_base / T_block; efficiency E = S / p. Fill these from your own
runs — the block version uses one fewer "dimension" of compare steps and half
the processes, so it issues markedly fewer messages.

## Why it still sorts
The bitonic network's phase/stage structure is identical. Replacing each
single-element compare-exchange with a block compare-split preserves the
0-1 principle: a compare-split correctly separates the low/high halves of two
already-sorted blocks, which is exactly what the network needs at each node.
See the handwritten theory for the full inductive argument.
