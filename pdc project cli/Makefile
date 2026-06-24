# ===========================================================================
# Makefile  --  for users who have `make` (mingw32-make on Windows, or Linux)
# ===========================================================================
# Phase 1 provides `sequential`; Phase 2 adds `openmp`; Phase 3 adds `mpi` and
# `hybrid`. All targets reuse the same $(COMMON) objects.
#
#   make            # builds sequential + openmp
#   make mpi        # builds the MPI target      (needs mpicxx; use WSL on Windows)
#   make hybrid     # builds the MPI+OpenMP target
#   make phase3     # builds mpi + hybrid
#   make run        # builds + runs sequential on the 10k AG News dataset
#   make run-mpi    # builds + runs MPI on 10k with 4 processes
#   make clean      # removes build artifacts
#
# NOTE: `mpi`/`hybrid` need `mpicxx` (OpenMPI). Windows MinGW has none, so build
# those under WSL:  sudo apt-get install build-essential openmpi-bin libopenmpi-dev
# ===========================================================================

CXX       := g++
MPICXX    := mpicxx
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra
BUILD_DIR := build

COMMON := src/common/1text_cleaning.cpp \
          src/common/2quality_filter.cpp \
          src/common/3deduplication.cpp \
          src/common/4tokenizer.cpp \
          src/common/io_utils.cpp

# On Windows the binary is .exe; on Linux it has no extension. Detect the OS.
# MKDIR_BUILD is the OS-appropriate "create build/ if missing" command -- the
# Windows form must not run on Linux (its `2>nul` would create a stray `nul`
# file), so we branch on the OS rather than share one recipe line.
ifeq ($(OS),Windows_NT)
    EXE := .exe
    MKDIR_BUILD := -@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
else
    EXE :=
    MKDIR_BUILD := @mkdir -p $(BUILD_DIR)
endif

SEQ_BIN    := $(BUILD_DIR)/seq_pipeline$(EXE)
OMP_BIN    := $(BUILD_DIR)/omp_pipeline$(EXE)
MPI_BIN    := $(BUILD_DIR)/mpi_pipeline$(EXE)
HYBRID_BIN := $(BUILD_DIR)/hybrid_pipeline$(EXE)

.PHONY: sequential openmp mpi hybrid phase3 all run run-omp run-mpi run-hybrid clean

all: sequential openmp

sequential: $(SEQ_BIN)

$(SEQ_BIN): src/sequential/main_sequential.cpp $(COMMON)
	$(MKDIR_BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo Built $@

# Phase 2: OpenMP target -- same common objects, plus -fopenmp.
# (On MinGW.org g++ this needs pthreads-w32: `mingw-get install mingw32-pthreads-w32`.)
openmp: $(OMP_BIN)

$(OMP_BIN): src/openmp/main_openmp.cpp $(COMMON)
	$(MKDIR_BUILD)
	$(CXX) $(CXXFLAGS) -fopenmp $^ -o $@
	@echo Built $@

# Phase 3: MPI target -- same common objects, built with mpicxx.
mpi: $(MPI_BIN)

$(MPI_BIN): src/mpi/main_mpi.cpp $(COMMON)
	$(MKDIR_BUILD)
	$(MPICXX) $(CXXFLAGS) -Wno-unknown-pragmas $^ -o $@
	@echo Built $@

# Phase 3: Hybrid target -- mpicxx + -fopenmp (activates the pragmas in the
# shared mpi_pipeline.hpp; the MPI target above compiles the same code without
# -fopenmp, so the two cannot diverge).
hybrid: $(HYBRID_BIN)

$(HYBRID_BIN): src/hybrid/main_hybrid.cpp $(COMMON)
	$(MKDIR_BUILD)
	$(MPICXX) $(CXXFLAGS) -fopenmp $^ -o $@
	@echo Built $@

phase3: mpi hybrid

run: sequential
	$(SEQ_BIN) data/agnews/agnews_10k.txt results/clean_10k.txt 5

run-omp: openmp
	$(OMP_BIN) data/agnews/agnews_10k.txt results/clean_omp_10k.txt 5 4

run-mpi: mpi
	mpirun --allow-run-as-root -np 4 $(MPI_BIN) data/agnews/agnews_10k.txt results/clean_10k.txt 5

run-hybrid: hybrid
	mpirun --allow-run-as-root -np 2 $(HYBRID_BIN) data/agnews/agnews_10k.txt results/clean_10k.txt 5 4

clean:
	-@rm -rf $(BUILD_DIR) 2>nul || true
