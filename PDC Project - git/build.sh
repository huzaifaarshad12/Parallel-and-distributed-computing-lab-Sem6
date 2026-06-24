#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# build.sh  --  one-command build for Linux / WSL (g++ + OpenMPI)
# ---------------------------------------------------------------------------
# Builds all four targets of the pipeline:
#   build/seq_pipeline      Phase 1  (sequential)        g++
#   build/omp_pipeline      Phase 2  (OpenMP)            g++  -fopenmp
#   build/mpi_pipeline      Phase 3  (MPI)               mpicxx
#   build/hybrid_pipeline   Phase 3  (MPI + OpenMP)      mpicxx -fopenmp
#
# The Windows counterpart is build.ps1 (MinGW g++). This script is for the
# Phase 3 toolchain, which lives under WSL because Windows MinGW has no mpicxx.
#
# Usage (from the project root, inside WSL):
#     ./build.sh
#
# Prerequisites (Ubuntu/WSL, once):
#     sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

CXXFLAGS="-O2 -std=c++17 -Wall -Wextra"
COMMON="src/common/1text_cleaning.cpp src/common/2quality_filter.cpp \
        src/common/3deduplication.cpp src/common/4tokenizer.cpp \
        src/common/io_utils.cpp"

mkdir -p build results

echo "Building SEQUENTIAL target..."
# shellcheck disable=SC2086
g++ $CXXFLAGS src/sequential/main_sequential.cpp $COMMON -o build/seq_pipeline
echo "OK  -> build/seq_pipeline"

echo "Building OPENMP target..."
# shellcheck disable=SC2086
g++ $CXXFLAGS -fopenmp src/openmp/main_openmp.cpp $COMMON -o build/omp_pipeline
echo "OK  -> build/omp_pipeline"

echo "Building MPI target..."
# -Wno-unknown-pragmas: the shared header carries "#pragma omp" directives that
# are intentionally ignored here (no -fopenmp) -- that is exactly what makes
# this the pure-MPI build, so the warning is noise.
# shellcheck disable=SC2086
mpicxx $CXXFLAGS -Wno-unknown-pragmas src/mpi/main_mpi.cpp $COMMON -o build/mpi_pipeline
echo "OK  -> build/mpi_pipeline"

echo "Building HYBRID (MPI + OpenMP) target..."
# shellcheck disable=SC2086
mpicxx $CXXFLAGS -fopenmp src/hybrid/main_hybrid.cpp $COMMON -o build/hybrid_pipeline
echo "OK  -> build/hybrid_pipeline"

echo "All targets built into build/"
