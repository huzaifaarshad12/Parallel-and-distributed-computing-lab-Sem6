# ---------------------------------------------------------------------------
# build.ps1  --  one-command build for Windows + MinGW-w64 g++
# ---------------------------------------------------------------------------
# Phase 1 builds only the sequential target. Phases 2 and 3 add their own
# targets to this script (openmp / mpi / hybrid). Run from the project root:
#
#     powershell -ExecutionPolicy Bypass -File build.ps1
#
# or, if your execution policy already allows local scripts:
#
#     .\build.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$CXX      = "g++"
$CXXFLAGS = @("-O2", "-std=c++17", "-Wall", "-Wextra")

# Shared (common) source files used by every version of the pipeline.
$COMMON = @(
    "src/common/1text_cleaning.cpp",
    "src/common/2quality_filter.cpp",
    "src/common/3deduplication.cpp",
    "src/common/4tokenizer.cpp",
    "src/common/io_utils.cpp"
)

# Make sure output folders exist.
New-Item -ItemType Directory -Force -Path "build"   | Out-Null
New-Item -ItemType Directory -Force -Path "results" | Out-Null

Write-Host "Building SEQUENTIAL target..." -ForegroundColor Cyan
$compileArgs = $CXXFLAGS + @("src/sequential/main_sequential.cpp") + $COMMON + @("-o", "build/seq_pipeline.exe")
& $CXX @compileArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK  -> build\seq_pipeline.exe" -ForegroundColor Green
} else {
    Write-Host "Build failed." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------------------
# Phase 2: OpenMP target. Same sources + the same common modules, but adds
# -fopenmp so the "#pragma omp parallel for" / reduction directives are
# enabled and libgomp is linked.
#
# Toolchain note: MinGW.org g++ 6.3.0's OpenMP runtime needs the pthreads-w32
# library. If linking fails with "cannot find -lpthread", install it once with:
#     mingw-get install mingw32-pthreads-w32
# -------------------------------------------------------------------------
Write-Host "Building OPENMP target..." -ForegroundColor Cyan
$ompArgs = $CXXFLAGS + @("-fopenmp", "src/openmp/main_openmp.cpp") + $COMMON + @("-o", "build/omp_pipeline.exe")
& $CXX @ompArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host "OK  -> build\omp_pipeline.exe" -ForegroundColor Green
} else {
    Write-Host "OpenMP build failed." -ForegroundColor Red
    exit 1
}

# -------------------------------------------------------------------------
# Phase 3 (MPI + Hybrid) is NOT built here. Windows MinGW g++ has no `mpicxx`,
# so the distributed targets are built and run under WSL (Ubuntu + OpenMPI)
# with `build.sh`. From a WSL shell, in the project root:
#
#     sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev   # once
#     ./build.sh            # builds seq + omp + mpi + hybrid
#     ./scripts/verify_mpi.sh
#     ./scripts/benchmark_mpi.sh
#
# See docs/report.md for the full toolchain and design notes.
# -------------------------------------------------------------------------
Write-Host "Phase 3 (MPI/hybrid) builds under WSL via ./build.sh -- see docs/report.md" -ForegroundColor DarkGray
