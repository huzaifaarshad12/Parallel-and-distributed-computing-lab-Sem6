#!/usr/bin/env python3
"""Problem 9.9 - weak-scaling / isoefficiency helper.

Runs block_ring under weak scaling (n and p grow together) and plots efficiency
vs p. Efficiency E = T1 / (p * Tp), where T1 is the 1-process time for the same
total n. We approximate T1 with a sequential sort timing.

Usage:  python3 plot_isoefficiency.py
Requires: mpirun, the compiled ./block_ring, matplotlib, numpy.
"""
import subprocess, re, time, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_block_ring(p, k):
    out = subprocess.run(
        ["mpirun", "--oversubscribe", "-np", str(p), "./block_ring", str(k)],
        capture_output=True, text=True)
    m = re.search(r"time=([0-9.]+)", out.stdout)
    return float(m.group(1)) if m else None

def seq_sort_time(n):
    a = [random.randint(0, 100000) for _ in range(n)]
    t0 = time.perf_counter(); a.sort(); t1 = time.perf_counter()
    return t1 - t0

# Weak scaling: keep k (work per process) fixed, grow p.
k = 2048
ps = [2, 4, 8, 16]
eff, times = [], []
for p in ps:
    n = p * k
    tp = run_block_ring(p, k)
    t1 = seq_sort_time(n)               # sequential baseline for same n
    e = t1 / (p * tp) if tp and tp > 0 else 0.0
    times.append(tp); eff.append(e)
    print(f"p={p:2d} n={n:6d} Tp={tp:.6f}s T1={t1:.6f}s  E={e:.3f}")

plt.figure(figsize=(6,4))
plt.plot(ps, eff, "o-", label="measured efficiency")
plt.xlabel("number of processes p")
plt.ylabel("efficiency E")
plt.title("Ring block bitonic: weak-scaling efficiency")
plt.ylim(0, max(1.0, max(eff)*1.1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("efficiency_vs_p.png", dpi=120)
print("saved efficiency_vs_p.png")
