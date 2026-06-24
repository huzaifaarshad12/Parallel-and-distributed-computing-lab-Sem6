// ---------------------------------------------------------------------------
// main_mpi.cpp
// ---------------------------------------------------------------------------
// PHASE 3 DELIVERABLE (part 1): the distributed-memory (MPI) version.
//
// A master-worker pipeline: rank 0 reads the dataset and SCATTERS contiguous
// document batches to the worker ranks; every rank runs the SAME src/common/
// stage functions on its batch; rank 0 GATHERS the survivors and performs the
// final, serial, id-ordered deduplication so the output stays byte-identical to
// the sequential baseline. Only the timing changes.
//
// The entire implementation lives in src/common/mpi_pipeline.hpp so that this
// pure-MPI version and the hybrid (MPI+OpenMP) version share ONE source of
// truth and therefore cannot diverge. This file is compiled WITHOUT -fopenmp,
// so the "#pragma omp" directives in the shared header are ignored and each
// process runs its batch serially.
//
// Build (under WSL / Linux, OpenMPI):
//   mpicxx -O2 -std=c++17 src/mpi/main_mpi.cpp src/common/*.cpp -o build/mpi_pipeline
//
// Run:
//   mpirun -np 4 build/mpi_pipeline data/agnews/agnews_10k.txt results/clean_10k.txt 5
//   mpi_pipeline <input> <output> [min_words] [threads(ignored)] [csv_path]
// ---------------------------------------------------------------------------

#include "../common/mpi_pipeline.hpp"

int main(int argc, char** argv) {
    return run_pipeline_mpi(argc, argv, "MPI");
}
