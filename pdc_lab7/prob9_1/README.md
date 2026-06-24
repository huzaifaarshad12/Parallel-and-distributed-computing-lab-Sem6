# Problem 9.1 — Compare-Split: Merge-Split vs Pairwise Exchange

Two MPI programs (2 processes each) that perform one compare-split between P0 and P1.

## Files
- `methodA_mergesplit.c` — standard merge-split: exchange whole block once, merge, keep half.
- `methodB_pairwise.c`   — pairwise exchange: send one element at a time (per the hint).
- `Makefile`

## Build
```
make            # builds methodA and methodB
```
Requires an MPI compiler (`mpicc`) and launcher (`mpirun`), e.g. OpenMPI or MPICH.

## Run
```
mpirun -np 2 ./methodA 1000
mpirun -np 2 ./methodB 1000
```
On a single multi-core machine where only 1 slot is detected, add `--oversubscribe`:
```
mpirun --oversubscribe -np 2 ./methodA 1000
```
Or just run the whole sweep:
```
make run        # set NPFLAGS=--oversubscribe in the Makefile if needed
```

## Output format
```
METHOD_A  k=1000  time=0.000035 s  messages=2  bytes=8000  PASS
```
`messages` and `bytes` are summed across both processes. `PASS` means the
compare-split property holds (the two halves are correctly separated).

## Measured results (example run, 2 procs oversubscribed on one node)
| k     | A time (s) | A msgs | A bytes | B time (s) | B msgs | B bytes |
|-------|-----------:|-------:|--------:|-----------:|-------:|--------:|
| 10    | 0.000016   | 2      | 80      | 0.000023   | 18     | 72      |
| 100   | 0.000014   | 2      | 800     | 0.000157   | 189    | 756     |
| 1000  | 0.000035   | 2      | 8000    | 0.001079   | 1527   | 6108    |
| 10000 | 0.000315   | 2      | 80000   | 0.010473   | 14784  | 59136   |

(Your numbers will differ by machine; re-run to fill your report.)

## Discussion
Method A sends a constant **2 messages** regardless of k; Method B sends O(k)
tiny messages. Even though Method B transfers *fewer bytes* (the early-stop
condition triggers before all k elements are exchanged), it is far slower
because **per-message latency dominates** — each element pays a full
startup cost t_s. This is the classic lesson: aggregate communication into
few large messages.

**MIMD vs SIMD:** Method B is MIMD-friendly — each process runs its own
control flow, decides independently when to stop, and the two processes do
*different* work (one sends, one compares/replies). It maps poorly to SIMD,
where all PEs execute the same instruction in lockstep and fine-grained
data-dependent branching (the early stop) is wasteful. Method A's bulk
exchange + identical local merge is comparatively more SIMD-amenable.
