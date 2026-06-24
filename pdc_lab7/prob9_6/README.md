# Problem 9.6 — Full Hypercube Bitonic Sort, Algorithm 9.1 (MPI)

n = 2^d processes, one element each. Runs the textbook bitonic sort:
`for i in 0..d-1 { for j in i..0 { compare-exchange with rank XOR (1<<j) } }`.
Sort direction at a node is set by the bitwise condition (bit i+1 of the rank);
which side keeps min/max is set by bit j of the rank.

## Build / Run
```
make
mpirun --oversubscribe -np 8  ./bitonic_hypercube       # n=8
mpirun --oversubscribe -np 16 ./bitonic_hypercube        # n=16
mpirun --oversubscribe -np 32 ./bitonic_hypercube        # n=32
mpirun --oversubscribe -np 8  ./bitonic_hypercube 1      # debug: print per-phase
```

## Debug / "screenshot" deliverable
`sample_output_debug.txt` shows the element distribution after each phase for
n=8. You can see the bitonic property emerge: after phase 1 the array holds
two size-4 bitonic runs (one ascending `25 53 742 799`, one descending
`816 679 292 283`), and phase 2 merges them into a fully sorted array.

## How the code follows Algorithm 9.1
- Phase index `i` builds bitonic sequences of length 2^(i+1).
- Stage index `j` (from i down to 0) selects the compare-exchange distance 2^j.
- Partner = `rank XOR (1<<j)` is the hypercube neighbour along dimension j.
- `ascending = (bit (i+1) of rank == 0)` chooses the sort window direction.
- `keep_min` combines that window with the low/high side (bit j) so the
  correct neighbour retains the smaller element.
- Verified by gathering at rank 0 and checking ascending order; all of
  n = 8, 16, 32 report PASS.
