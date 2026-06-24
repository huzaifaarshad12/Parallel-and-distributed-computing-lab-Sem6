// ---------------------------------------------------------------------------
// main_hybrid.cpp
// ---------------------------------------------------------------------------
// PHASE 3 DELIVERABLE (part 2): the HYBRID MPI + OpenMP version.
//
// Two levels of parallelism at once:
//   * MPI    distributes contiguous document batches ACROSS processes (the
//            same master-worker scatter/gather design as main_mpi.cpp), and
//   * OpenMP parallelises the per-document stage loops WITHIN each process,
//            across that machine's cores (the same loops as Phase 2).
//
// This is exactly the pure-MPI program from src/common/mpi_pipeline.hpp, but
// compiled WITH -fopenmp, which "switches on" the "#pragma omp parallel for"
// directives inside the shared header. Because the logic is literally the same
// source, the output remains byte-identical to the sequential baseline for any
// combination of processes and threads -- only the timing changes.
//
// Why hybrid? On a cluster you launch ONE MPI process per node (avoiding
// cross-node memory traffic) and let OpenMP use all cores inside the node. It
// combines MPI's data distribution with OpenMP's low-overhead shared-memory
// threading.
//
// Build (under WSL / Linux, OpenMPI):
//   mpicxx -O2 -std=c++17 -fopenmp src/hybrid/main_hybrid.cpp src/common/*.cpp -o build/hybrid_pipeline
//
// Run (e.g. 2 processes x 4 threads each = 8-way parallel):
//   OMP_NUM_THREADS=4 mpirun -np 2 build/hybrid_pipeline data/agnews/agnews_full.txt results/clean_full.txt 5
//   hybrid_pipeline <input> <output> [min_words] [threads] [csv_path]
//   ([threads] overrides OMP_NUM_THREADS for this process.)
// ---------------------------------------------------------------------------

#include "../common/mpi_pipeline.hpp"

int main(int argc, char** argv) {
    return run_pipeline_mpi(argc, argv, "HYBRID");
}
