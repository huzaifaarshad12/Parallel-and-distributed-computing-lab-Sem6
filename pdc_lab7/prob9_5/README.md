# Problem 9.5 — Hypercube Bitonic Merge (MPI)

One bitonic merge of size 2^k on a k-dimensional hypercube. Each process holds
one integer; neighbours along dimension j are `rank XOR (1<<j)`. The merge does
compare-exchange along dimensions k-1, k-2, ..., 0.

## Build / Run
```
make
mpirun --oversubscribe -np 8 ./hypercube_merge      # k=3
```

## Output
Each rank prints its sequence of communication partners (one per dimension)
and its final element; rank 0 prints the merged array and a PASS/FAIL.
Sample for k=3 is in `sample_output_k3.txt`.

## Hypercube diagram (k=3, 8 nodes)
Nodes are 0..7 (3-bit labels). Edges connect labels differing in one bit:

```
        110(6)---------111(7)
        /|             /|
   100(4)----------101(5)
       | |            | |
       |010(2)--------|011(3)
       |/             |/
   000(0)----------001(1)
```
Merge dimension order (edges used, in sequence):
- dim 2 (bit 2): 0-4, 1-5, 2-6, 3-7
- dim 1 (bit 1): 0-2, 1-3, 4-6, 5-7
- dim 0 (bit 0): 0-1, 2-3, 4-5, 6-7

## Why a size-2^k merge lives on a k-cube
A bitonic merge of 2^k elements needs exactly k compare-exchange stages, one
per dimension, and at each stage every element pairs with the partner that
differs in exactly one bit. Those are precisely the edges of a k-dimensional
hypercube, so the 2^k elements map one-to-one onto the 2^k nodes of a k-cube,
and disjoint size-2^k subsequences map onto disjoint k-dimensional subcubes.
