#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# benchmark_mpi.sh -- Phase 3 MPI + HYBRID performance sweep.
#
# Records EVERY run's per-stage and total timings to results/benchmark_mpi.csv,
# then prints speedup / efficiency summaries using the best (minimum) total time
# per configuration -- the standard way to report parallel timings (the minimum
# is least polluted by OS-scheduling jitter and background load).
#
#   Speedup    S(p) = T(1) / T(p)     (how many times faster with p workers)
#   Efficiency E(p) = S(p) / p         (fraction of ideal linear speedup)
#
# MPI baseline T(1) = the MPI binary on 1 process.
# Hybrid is reported by total worker count (procs x threads) against the MPI
# 1-process time, so MPI and hybrid sit on the same scale.
#
# NOTE on cores: this machine has 12 logical CPUs. We keep total parallelism
# (procs x threads) <= 12 so we measure real scaling, not oversubscription.
#
# Run from the project root, inside WSL:   ./scripts/benchmark_mpi.sh
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

DATASETS=(10k 50k full)     # 1k is too small to time meaningfully
MPI_PROCS=(1 2 4 8)         # MPI process counts
HYBRID=("2x2" "2x4" "4x2" "2x6" "4x3" "6x2")  # processes x threads (product <= 12)
MINWORDS=5
REPS=5
CSV=results/benchmark_mpi.csv
MPIRUN="mpirun --allow-run-as-root --oversubscribe"
mkdir -p results

# Fresh detailed CSV (per-run, per-stage).
echo "version,dataset,dataset_size,procs,threads,rep,load_ms,clean_ms,filter_ms,dedup_ms,tokenize_ms,total_ms" > "$CSV"

# Extract one stage timing (ms) from a run's stdout.
stage() { awk -v k="$1" '$1==k {print $3; exit}'; }
get_size() { awk '/Total documents/ {print $NF; exit}'; }

declare -A best   # best["ds,key"] = min total_ms

run_cfg() {  # version dataset procs threads label_key
    local version=$1 ds=$2 np=$3 th=$4 key=$5
    local in="data/agnews/agnews_${ds}.txt"
    local bin out size load cln flt ddp tok tot
    local mintot=1000000000
    [[ $version == mpi ]] && bin=./build/mpi_pipeline || bin=./build/hybrid_pipeline
    for ((r=1; r<=REPS; r++)); do
        out=$($MPIRUN -np "$np" "$bin" "$in" "results/_bench_mpi.txt" "$MINWORDS" "$th" 2>/dev/null)
        size=$(echo "$out" | get_size)
        load=$(echo "$out" | stage load)
        cln=$(echo "$out"  | stage clean)
        flt=$(echo "$out"  | stage filter)
        ddp=$(echo "$out"  | stage dedup)
        tok=$(echo "$out"  | stage tokenize)
        tot=$(echo "$out"  | stage TOTAL)
        echo "$version,$ds,$size,$np,$th,$r,$load,$cln,$flt,$ddp,$tok,$tot" >> "$CSV"
        awk -v a="$tot" -v b="$mintot" 'BEGIN{exit !(a<b)}' && mintot=$tot
    done
    best["$ds,$key"]=$mintot
    printf "    %-12s (%sp x %st): best total = %9.3f ms\n" "$version" "$np" "$th" "$mintot"
}

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "==== dataset $ds ===="
    echo "  -- MPI --"
    for np in "${MPI_PROCS[@]}"; do run_cfg mpi "$ds" "$np" 1 "mpi$np"; done
    echo "  -- HYBRID --"
    for combo in "${HYBRID[@]}"; do
        np=${combo%x*}; th=${combo#*x}
        run_cfg hybrid "$ds" "$np" "$th" "hyb$combo"
    done
done

# ----- speedup / efficiency summary (baseline = MPI 1 process) -------------
echo ""
echo "================ MPI SPEEDUP  (S = T1 / Tp,  baseline = mpi 1 proc) ================"
printf "%-8s" "dataset"; for np in "${MPI_PROCS[@]}"; do printf "%10s" "p=$np"; done; echo
for ds in "${DATASETS[@]}"; do
    t1=${best["$ds,mpi1"]}
    printf "%-8s" "$ds"
    for np in "${MPI_PROCS[@]}"; do
        awk -v t1="$t1" -v tp="${best["$ds,mpi$np"]}" 'BEGIN{printf "%9.2fx", t1/tp}'
    done; echo
done

echo ""
echo "================ MPI EFFICIENCY  (E = S / p) ================"
printf "%-8s" "dataset"; for np in "${MPI_PROCS[@]}"; do printf "%10s" "p=$np"; done; echo
for ds in "${DATASETS[@]}"; do
    t1=${best["$ds,mpi1"]}
    printf "%-8s" "$ds"
    for np in "${MPI_PROCS[@]}"; do
        awk -v t1="$t1" -v tp="${best["$ds,mpi$np"]}" -v p="$np" 'BEGIN{printf "%9.0f%%", 100*(t1/tp)/p}'
    done; echo
done

echo ""
echo "================ HYBRID SPEEDUP  (baseline = mpi 1 proc, by total workers) ================"
printf "%-8s" "dataset"; for c in "${HYBRID[@]}"; do printf "%10s" "$c"; done; echo
for ds in "${DATASETS[@]}"; do
    t1=${best["$ds,mpi1"]}
    printf "%-8s" "$ds"
    for c in "${HYBRID[@]}"; do
        awk -v t1="$t1" -v tp="${best["$ds,hyb$c"]}" 'BEGIN{printf "%9.2fx", t1/tp}'
    done; echo
done

echo ""
echo "Detailed per-run timings written to $CSV"
