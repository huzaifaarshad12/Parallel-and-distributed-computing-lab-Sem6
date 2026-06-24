# PDC Pipeline — Web UI

A professional, web-based console for the **Parallel Data Preprocessing Pipeline
for LLM Fine-Tuning**. It runs the real compiled C++ binaries live and visualizes
the benchmark results for all four versions.

- **Frontend:** React + Vite + Recharts (Claude/Anthropic-style design)
- **Backend:** Node + Express — executes `build/seq_pipeline.exe` /
  `build/omp_pipeline.exe`, parses their output, and serves the benchmark CSVs.

## Features

- **Run Console** — choose a dataset (AG News 1K/10K/50K/full) and a parallelism
  mode (**Sequential · OpenMP · MPI · Hybrid**), set `min_words`, processes and
  threads, and execute. Shows the real counters, a **live speedup vs sequential**,
  per-stage timing chart, total time, the exact command, a SHA-256 of the cleaned
  output, and a **byte-identical-to-sequential** correctness badge, plus an output
  preview.
- **Benchmarks** — interactive OpenMP speedup/efficiency, MPI speedup, and Hybrid
  speedup charts, built from `results/benchmark_openmp.csv` and `benchmark_mpi.csv`.
- **About** — the pipeline, the design philosophy, and the end-to-end story.

> **All four versions run live.** Sequential & OpenMP execute the Windows `.exe`
> binaries directly; **MPI & Hybrid execute inside WSL** via
> `wsl.exe … mpirun …` (so WSL + the `build.sh` binaries must be present). The
> Benchmarks tab shows the full pre-measured sweep for every version.

## Prerequisites

- Node.js 18+ (`node --version`).
- The pipeline already built: run `.\build.ps1` from the project root so
  `build/seq_pipeline.exe` and `build/omp_pipeline.exe` exist.
- The dataset present in `data/agnews/` (`python scripts\download_agnews.py`).

## Install & run (production-style — one server)

```powershell
# from the project root
cd web\client;  npm install;  npm run build;  cd ..\..
cd web\server;  npm install;  npm start
# open http://localhost:3001
```

The server serves the built UI and the API on the same port (3001).

## Develop (hot reload — two servers)

```powershell
# terminal 1 — API
cd web\server;  npm install;  npm start          # http://localhost:3001

# terminal 2 — Vite dev server (proxies /api -> 3001)
cd web\client;  npm install;  npm run dev         # http://localhost:5173
```

## API

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/api/health` | server + binary availability |
| GET | `/api/datasets` | available AG News files + doc counts |
| POST | `/api/run` | run a pipeline (`{dataset, version, minWords, threads}`) |
| GET | `/api/benchmarks/openmp` | aggregated OpenMP speedup/efficiency |
| GET | `/api/benchmarks/mpi` | aggregated MPI + Hybrid speedup |
