#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# verify_mpi.sh -- proves the MPI and HYBRID builds match the sequential baseline.
#
# For every dataset size it:
#   1. produces the sequential reference output,
#   2. runs the MPI build at several process counts,
#   3. runs the HYBRID build at several (processes x threads) combinations,
#   4. asserts each cleaned output file is BYTE-IDENTICAL to the reference
#      (sha256 compare),
#   5. asserts the five correctness counters match.
# Exit code is 0 only if every check passes.
#
# Run from the project root, inside WSL:   ./scripts/verify_mpi.sh
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

DATASETS=(1k 10k 50k full)
PROCS=(1 2 4 8)             # MPI process counts to verify
HYBRID=("2x2" "2x4" "4x3")  # hybrid  processes x threads
MINWORDS=5
TMP=results/_verify_mpi
MPIRUN="mpirun --allow-run-as-root --oversubscribe"
mkdir -p "$TMP"

# Pull the five counters out of a run's stdout: "total removed dups kept tokens".
counters() {
    awk '
      /Total documents/      {t=$NF}
      /Removed \(low quality\)/ {r=$NF}
      /Duplicates removed/   {d=$NF}
      /Kept documents/       {k=$NF}
      /Total tokens/         {n=$NF}
      END {print t, r, d, k, n}'
}

all_ok=1
for ds in "${DATASETS[@]}"; do
    in="data/agnews/agnews_${ds}.txt"
    ref="$TMP/seq_${ds}.txt"
    echo ""
    echo "==== dataset $ds ===="

    seq_out=$(./build/seq_pipeline "$in" "$ref" "$MINWORDS")
    seq_c=$(echo "$seq_out" | counters)
    ref_hash=$(sha256sum "$ref" | awk '{print $1}')
    echo "  seq: [$seq_c]"

    # ---- MPI ----
    for np in "${PROCS[@]}"; do
        out_file="$TMP/mpi_${ds}_p${np}.txt"
        run_out=$($MPIRUN -np "$np" ./build/mpi_pipeline "$in" "$out_file" "$MINWORDS" 2>/dev/null)
        c=$(echo "$run_out" | counters)
        h=$(sha256sum "$out_file" | awk '{print $1}')
        if [[ "$c" == "$seq_c" && "$h" == "$ref_hash" ]]; then
            printf "  OK   mpi    -np %-2s : bytes+counters match\n" "$np"
        else
            all_ok=0
            printf "  FAIL mpi    -np %-2s : counters[%s] vs [%s]  bytesMatch=%s\n" \
                   "$np" "$c" "$seq_c" "$([[ $h == "$ref_hash" ]] && echo yes || echo no)"
        fi
    done

    # ---- HYBRID ----
    for combo in "${HYBRID[@]}"; do
        np=${combo%x*}; th=${combo#*x}
        out_file="$TMP/hyb_${ds}_${combo}.txt"
        run_out=$($MPIRUN -np "$np" ./build/hybrid_pipeline "$in" "$out_file" "$MINWORDS" "$th" 2>/dev/null)
        c=$(echo "$run_out" | counters)
        h=$(sha256sum "$out_file" | awk '{print $1}')
        if [[ "$c" == "$seq_c" && "$h" == "$ref_hash" ]]; then
            printf "  OK   hybrid %sp x %st : bytes+counters match\n" "$np" "$th"
        else
            all_ok=0
            printf "  FAIL hybrid %sp x %st : counters[%s] vs [%s]  bytesMatch=%s\n" \
                   "$np" "$th" "$c" "$seq_c" "$([[ $h == "$ref_hash" ]] && echo yes || echo no)"
        fi
    done
done

echo ""
if [[ $all_ok -eq 1 ]]; then
    echo "ALL CHECKS PASSED: MPI and HYBRID == Sequential on every dataset/config."
    exit 0
else
    echo "VERIFICATION FAILED."
    exit 1
fi
