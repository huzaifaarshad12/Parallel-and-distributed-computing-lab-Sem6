# Parallel Data Preprocessing Pipeline for LLM Fine-Tuning

A data preprocessing pipeline that prepares raw text documents for Large
Language Model (LLM) fine-tuning, implemented in C++ in four progressively
parallel versions: **Sequential → OpenMP → MPI → Hybrid (MPI + OpenMP)**.

> **Current status: ✅ Phase 1 COMPLETE.  ✅ Phase 2 (OpenMP) COMPLETE.
> ✅ Phase 3 COMPLETE — MPI + Hybrid verified byte-identical, **and** DistilGPT-2
> fine-tuned on the cleaned corpus (validation perplexity ≈ 95.3).**
> All three phases are done. Jump to [Project status & roadmap](#project-status--roadmap)
> for the full breakdown, or [Resuming later](#resuming-later--where-i-left-off)
> to re-run any phase.

---

## Project status & roadmap

The project is delivered to the instructor in **three phases**. This table is
the single source of truth for progress.

| Phase | Goal | Status |
|-------|------|--------|
| **Phase 1** | Sequential baseline + datasets + all 4 stages + stats + timing | ✅ **DONE** |
| **Phase 2** | OpenMP (shared-memory threads), benchmarks, speedup tables | ✅ **DONE** |
| **Phase 3** | MPI master–worker + Hybrid MPI+OpenMP, benchmarks | ✅ **DONE** |
| | DistilGPT-2 Colab fine-tuning (run; val perplexity ≈ 95.3) | ✅ **DONE** |

### Phase 1 checklist (what is DONE)

- [x] Project architecture designed (pure stage functions in `src/common/`)
- [x] Folder structure created
- [x] Complete sequential implementation (`src/sequential/main_sequential.cpp`)
- [x] Real dataset obtained — **AG News** via `scripts/download_agnews.py`
      (1K / 10K / 50K / full sizes)
- [x] All four stages implemented: Cleaning, Quality Filter, Deduplication, Tokenization
- [x] Statistics stored: total / removed / duplicates / kept / total tokens
- [x] Timing with `<chrono>` (per stage + total)
- [x] Every file explained (see this README + `docs/report.md`)
- [x] Build & run commands provided
- [x] Expected output captured (see [Expected output](#expected-output))
- [x] Phase 1 write-up + viva Q&A (`docs/report.md` §5 and Appendix A.1)
- [x] All 4 stages independently verified on the dataset (`docs/report.md` §10)

### Phase 2 checklist (DONE — OpenMP)

- [x] Add OpenMP version `src/openmp/main_openmp.cpp` (reuses `src/common/`)
- [x] `#pragma omp parallel for` on cleaning / filtering loops
- [x] `reduction(+:tokens)` for the token count
- [x] Thread-safe deduplication (parallel fingerprint hashing + serial,
      id-ordered "first occurrence" decision — guarantees identical output)
- [x] Explain shared vs private variables; avoid race conditions (see
      `docs/report.md` §6 and Appendix A.2)
- [x] Benchmark module + performance tables (1, 2, 4, 8, 12 threads) —
      `scripts/benchmark.ps1`
- [x] Compute & explain speedup and efficiency (Amdahl's law) — up to **2.71×**
- [x] **Correctness proven:** OpenMP output byte-identical to sequential at all
      thread counts (`scripts/verify_openmp.ps1`)
- [x] Phase 2 write-up + viva Q&A (`docs/report.md` §6, §8 and Appendix A.2)

### Phase 3 checklist (DONE — MPI + Hybrid + fine-tuning)

- [x] MPI version `src/mpi/main_mpi.cpp` — master–worker, scatter batches / gather results
- [x] Hybrid version `src/hybrid/main_hybrid.cpp` — MPI across processes + OpenMP inside each
      (both share one implementation in `src/common/mpi_pipeline.hpp`, so they cannot diverge)
- [x] Communication design, synchronization, load balancing explained
      (`docs/report.md` §7 and Appendix A.3)
- [x] Execution time / speedup / efficiency measured; tables + CSV
      (`scripts/benchmark_mpi.sh` → `results/benchmark_mpi.csv`) — up to **1.76×**
- [x] **Correctness proven:** MPI & hybrid output byte-identical to sequential at
      every process/thread count (`scripts/verify_mpi.sh`, all 28 configs pass)
- [x] Google Colab notebook fine-tunes **DistilGPT-2** on the cleaned data
      (`docs/colab_finetune_distilgpt2.ipynb`) — **run on a Colab T4 GPU**: 1 epoch
      over `clean_10k.txt`, **validation perplexity ≈ 95.3** (val loss 4.557, train
      loss 5.12→4.78 in ~44 s), generating coherent news-style text
- [x] Phase 3 write-up + viva Q&A (`docs/report.md` §7, §8 and Appendix A.3)

> ⚠️ **Toolchain notes.**
> - **Phases 1–2** run on Windows with MinGW g++. Phase 2's `-fopenmp` needed
>   the pthreads-w32 import library once: `mingw-get install mingw32-pthreads-w32`.
> - **Phase 3 (MPI)** has no `mpicxx` under Windows MinGW, so it is built and run
>   under **WSL (Ubuntu)** — which reads the same files via `/mnt/d`. One-time
>   setup: `sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev`
>   (provides g++ 13, OpenMPI 4.1 `mpicxx`/`mpirun`). Then `./build.sh` builds
>   all four targets. See [Build & run (Phase 3)](#build--run-phase-3--mpi--hybrid)
>   and `docs/report.md` §7.

---

## Complete workflow (start to finish)

This is the **whole project, end to end**, in the order you actually run it —
from a fresh checkout to a fine-tuned language model. It spans three
environments: **Windows** (Phases 1–2), **WSL/Ubuntu** (Phase 3, needs MPI), and
**Google Colab** (fine-tuning, needs a GPU). Each step links to the detailed
section below. Run every command from the project root.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ STEP 0  Get the dataset            (once)            Python               │
 │ STEP 1  Phase 1  Sequential        build + run       Windows  (MinGW g++) │
 │ STEP 2  Phase 2  OpenMP            build + run + verify + benchmark  ”     │
 │ STEP 3  Phase 3  MPI + Hybrid      build + run + verify + benchmark  WSL   │
 │ STEP 4  Fine-tune DistilGPT-2      upload clean file + run notebook  Colab │
 └─────────────────────────────────────────────────────────────────────────┘
   raw AG News ──▶ cleaned corpus (identical from every version) ──▶ trained LLM
```

### STEP 0 — Get the dataset (once, Python)

```powershell
pip install datasets
python scripts\download_agnews.py      # -> data\agnews\agnews_1k/10k/50k/full.txt
```

### STEP 1 — Phase 1: Sequential (Windows)

```powershell
.\build.ps1                                                                   # builds seq + omp .exe
build\seq_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5
```

Prints the statistics + per-stage timings and writes `results\clean_10k.txt`.
Details: [Build & run (Phase 1)](#build--run-phase-1).

### STEP 2 — Phase 2: OpenMP (Windows)

```powershell
build\omp_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5 8    # 8 threads
.\scripts\verify_openmp.ps1            # proves OpenMP output == sequential (byte-identical)
.\scripts\benchmark.ps1                # thread sweep -> results\benchmark_openmp.csv + speedup tables
```

Details: [Build & run (Phase 2)](#build--run-phase-2--openmp).

### STEP 3 — Phase 3: MPI + Hybrid (WSL / Ubuntu)

Open a **WSL** shell (the project is visible at `/mnt/d/...`), then:

```bash
# one-time toolchain install
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev

cd "/mnt/d/Documents/Semester6/PDCL/PDC Project"
./build.sh                                                                     # builds all 4 targets

# run MPI (4 processes) and Hybrid (2 processes x 4 threads)
mpirun --allow-run-as-root -np 4 build/mpi_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5
mpirun --allow-run-as-root -np 2 build/hybrid_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5 4

./scripts/verify_mpi.sh        # proves MPI & hybrid output == sequential (28 configs)
./scripts/benchmark_mpi.sh     # process/thread sweep -> results/benchmark_mpi.csv + tables
```

Details: [Build & run (Phase 3)](#build--run-phase-3--mpi--hybrid).

### STEP 4 — Fine-tune DistilGPT-2 (Google Colab, GPU)

1. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → `docs/colab_finetune_distilgpt2.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. When the upload button appears (step 2 of the
   notebook), upload a **cleaned** file — `results/clean_10k.txt` (fast) or
   `results/clean_full.txt` (stronger) — **not** a raw `agnews_*.txt`.
4. The notebook tokenizes, fine-tunes (1 epoch), prints **validation perplexity**
   + sample generations, then the **last cell zips and downloads** the model
   (`distilgpt2-agnews-final.zip`).

Measured run (T4, `clean_10k.txt`): val perplexity ≈ **95.3**, coherent
news-style text. Details: [Fine-tuning DistilGPT-2](#fine-tuning-distilgpt-2-on-the-cleaned-corpus).

> **The thread that ties it together:** every version in Steps 1–3 produces a
> **byte-identical** cleaned corpus (verified). Step 4 trains a model on it — the
> full path from *messy data* → *parallel cleaning* → *fine-tuned LLM*.

---

## ▶️ Copy-paste run log (verified 2026-06-22)

This section is the **fastest path**: copy each command into the terminal, in
order, and you should see the output shown beneath it. Phases 1–2 run in
**Windows PowerShell**; Phase 3 runs in a **WSL/Ubuntu** shell. The single proof
of correctness is that **every version produces the same SHA-256 hash and the
same counters** — only the timings differ.

### A) Phases 1 & 2 — Windows (PowerShell)

```powershell
.\build.ps1
```
```
Building SEQUENTIAL target...
OK  -> build\seq_pipeline.exe
Building OPENMP target...
OK  -> build\omp_pipeline.exe
Phase 3 (MPI/hybrid) builds under WSL via ./build.sh -- see docs/report.md
```

```powershell
build\seq_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5
```
```
==================== SEQUENTIAL ====================
Total documents      : 10000
Removed (low quality): 0
Duplicates removed   : 55
Kept documents       : 9945
Total tokens         : 402830
----------------------------------------------------
Stage timings (ms):
  load      : 41.965
  clean     : 50.935
  filter    : 26.209
  dedup     : 5.488
  tokenize  : 25.153
  TOTAL     : 158.161
====================================================
Cleaned dataset written to: results\clean_10k.txt
```

```powershell
build\omp_pipeline.exe data\agnews\agnews_10k.txt results\clean_omp.txt 5 8
```
```
==================== OPENMP (8 threads) ====================
Total documents      : 10000
Removed (low quality): 0
Duplicates removed   : 55
Kept documents       : 9945
Total tokens         : 402830
----------------------------------------------------
Stage timings (ms):
  load      : 14.607
  clean     : 12.937
  filter    : 6.472
  dedup     : 1.387
  tokenize  : 6.067
  TOTAL     : 43.471
====================================================
Cleaned dataset written to: results\clean_omp.txt
```

**Prove the two outputs are byte-identical** (the hashes must be equal):

```powershell
(Get-FileHash results\clean_10k.txt -Algorithm SHA256).Hash
(Get-FileHash results\clean_omp.txt -Algorithm SHA256).Hash
```
```
51B11BC876C4B36EAE5CE6BD32DAAF1253E2564F1EBC2370F013AA7FF8F32920
51B11BC876C4B36EAE5CE6BD32DAAF1253E2564F1EBC2370F013AA7FF8F32920
```

**Automated correctness check** (sequential vs OpenMP on every dataset / thread count):

```powershell
.\scripts\verify_openmp.ps1
```
```
==== dataset 1k ====
  seq: total=1000 removed=0 dups=2 kept=998 tokens=41588
  OK    1 threads: output byte-identical + counters match
  OK    2 threads: output byte-identical + counters match
  OK    4 threads: output byte-identical + counters match
  OK    8 threads: output byte-identical + counters match

==== dataset 10k ====
  seq: total=10000 removed=0 dups=55 kept=9945 tokens=402830
  OK    1 threads: output byte-identical + counters match
  ... (2/4/8 threads all OK) ...

==== dataset 50k ====
  seq: total=50000 removed=0 dups=112 kept=49888 tokens=1993026
  ... (1/2/4/8 threads all OK) ...

==== dataset full ====
  seq: total=120000 removed=0 dups=218 kept=119782 tokens=4750500
  ... (1/2/4/8 threads all OK) ...

ALL CHECKS PASSED: OpenMP == Sequential on every dataset/thread count.
```

**Benchmark + speedup/efficiency tables** (`results\benchmark_openmp.csv`):

```powershell
.\scripts\benchmark.ps1
```
```
==== dataset 10k ====    1t=135.200  2t= 82.220  4t= 58.230  8t= 58.852  12t= 45.933 ms
==== dataset 50k ====    1t=643.056  2t=416.290  4t=335.599  8t=315.616  12t=294.390 ms
==== dataset full ====   1t=1951.584 2t=1073.806 4t=923.614 8t=760.732 12t=684.259 ms

================ SPEEDUP (S = T1/Tp) ================
dataset          p=1       p=2       p=4       p=8      p=12
10k            1.00x     1.64x     2.32x     2.30x     2.94x
50k            1.00x     1.54x     1.92x     2.04x     2.18x
full           1.00x     1.82x     2.11x     2.57x     2.85x

============== EFFICIENCY (E = S/p) ==============
dataset          p=1       p=2       p=4       p=8      p=12
10k             100%       82%       58%       29%       25%
50k             100%       77%       48%       25%       18%
full            100%       91%       53%       32%       24%
```

> Best OpenMP speedup observed: **2.94×** (10K, 12 threads); full set **2.85×**.

### B) Phase 3 — WSL / Ubuntu (MPI + Hybrid)

```bash
# one-time toolchain install (skip if already installed)
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev

cd "/mnt/d/Documents/Semester6/PDCL/PDC Project"
./build.sh
```
```
Building SEQUENTIAL target...
OK  -> build/seq_pipeline
Building OPENMP target...
OK  -> build/omp_pipeline
Building MPI target...
OK  -> build/mpi_pipeline
Building HYBRID (MPI + OpenMP) target...
OK  -> build/hybrid_pipeline
All targets built into build/
```

```bash
./build/seq_pipeline                                   data/agnews/agnews_10k.txt results/out_seq.txt 5
mpirun --allow-run-as-root -np 4 build/mpi_pipeline    data/agnews/agnews_10k.txt results/out_mpi.txt 5
mpirun --allow-run-as-root -np 2 build/hybrid_pipeline data/agnews/agnews_10k.txt results/out_hyb.txt 5 4
```
```
==================== SEQUENTIAL ====================        (TOTAL 270.265 ms)
==================== MPI (4 procs) ====================     (TOTAL 204.041 ms)
==================== HYBRID (2 procs x 4 threads) ==== (TOTAL 227.916 ms)
# all three: total=10000 removed=0 dups=55 kept=9945 tokens=402830
```

**Prove all three outputs are byte-identical** (the three hashes must match):

```bash
sha256sum results/out_seq.txt results/out_mpi.txt results/out_hyb.txt
```
```
51b11bc876c4b36eae5ce6bd32daaf1253e2564f1ebc2370f013aa7ff8f32920  results/out_seq.txt
51b11bc876c4b36eae5ce6bd32daaf1253e2564f1ebc2370f013aa7ff8f32920  results/out_mpi.txt
51b11bc876c4b36eae5ce6bd32daaf1253e2564f1ebc2370f013aa7ff8f32920  results/out_hyb.txt
```

> Note: this is the **same hash** (`51b11bc8…`) the Windows seq + OpenMP runs
> produced above — all **four** versions agree byte-for-byte across both OSes.

**Automated correctness check** (28 configs: MPI 1/2/4/8 procs + hybrid 2×2/2×4/4×3, on all 4 datasets):

```bash
./scripts/verify_mpi.sh
```
```
==== dataset 1k ====   seq: [1000 0 2 998 41588]
  OK   mpi -np 1/2/4/8 : bytes+counters match
  OK   hybrid 2px2t / 2px4t / 4px3t : bytes+counters match
==== dataset 10k ====  seq: [10000 0 55 9945 402830]      ... all OK ...
==== dataset 50k ====  seq: [50000 0 112 49888 1993026]   ... all OK ...
==== dataset full ==== seq: [120000 0 218 119782 4750500] ... all OK ...

ALL CHECKS PASSED: MPI and HYBRID == Sequential on every dataset/config.
```

**Benchmark + speedup/efficiency tables** (`results/benchmark_mpi.csv`):

```bash
./scripts/benchmark_mpi.sh
```
```
================ MPI SPEEDUP  (S = T1 / Tp,  baseline = mpi 1 proc) ================
dataset        p=1       p=2       p=4       p=8
10k          1.00x     1.29x     1.45x     1.64x
50k          1.00x     1.31x     1.31x     1.37x
full         1.00x     1.30x     1.47x     1.61x

================ MPI EFFICIENCY  (E = S / p) ================
dataset        p=1       p=2       p=4       p=8
10k           100%       64%       36%       21%
50k           100%       66%       33%       17%
full          100%       65%       37%       20%

================ HYBRID SPEEDUP  (baseline = mpi 1 proc, by total workers) ========
dataset        2x2       2x4       4x2       2x6       4x3       6x2
10k          1.51x     1.43x     1.54x     1.33x     1.60x     1.69x
50k          1.80x     1.52x     1.47x     1.45x     1.41x     1.55x
full         1.70x     1.36x     1.47x     1.46x     1.41x     1.62x
```

> Phase 3 peaks around **1.6–1.8×** on this single 12-core node — **lower than
> Phase 2's OpenMP** on purpose: on one machine every MPI rank shares the same
> memory bus and disk *and* must copy data between processes, while the `/mnt/d`
> serial read is slow. MPI's real payoff is **multi-node**. See the analysis in
> [Phase 3 results](#phase-3-results-this-12-core-laptop-wsl-min_words--5-best-of-5)
> and `docs/report.md` §8.4.

---

## The pipeline (4 stages)

| Stage | What it does | Module |
|-------|--------------|--------|
| 1. Text Cleaning | lowercase, remove special characters, normalize whitespace | `src/common/text_cleaning.*` |
| 2. Quality Filtering | drop documents below a minimum word count | `src/common/quality_filter.*` |
| 3. Deduplication | fingerprint each doc (FNV-1a hash), drop duplicates | `src/common/deduplication.*` |
| 4. Tokenization | split into word tokens, count total tokens | `src/common/tokenizer.*` |

---

## What is happening? (the big picture)

The job of this program is simple to state: **take a messy file of raw text and
turn it into a clean file that is ready to train a language model on**, while
counting exactly what it did.

Think of it like a factory assembly line. A document enters at one end, passes
through four machines (stages), and either falls off the line (it was junk or a
duplicate) or comes out the other end clean. At the end we get a report card:
how many documents came in, how many were thrown away and why, how many
survived, and how many word-tokens the final corpus contains.

```
  data/agnews/agnews_10k.txt               results/clean_10k.txt
  (raw news articles)                       (clean text, ready for training)
        │                                            ▲
        ▼                                            │
   ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
   │ STAGE 1 │──▶│ STAGE 2  │──▶│  STAGE 3  │──▶│ STAGE 4  │──▶ statistics + file
   │ Clean   │   │ Filter   │   │ Dedup     │   │ Tokenize │
   └─────────┘   └──────────┘   └───────────┘   └──────────┘
   lowercase,     drop docs      drop docs       count the
   strip symbols, with too few   we have seen    word-tokens
   fix spaces     words          before          that remain
```

### Step-by-step: what one document goes through

Here is a *real* raw line from AG News (the very first article in the dataset):

```
Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.
```

1. **Stage 1 — Cleaning.** Everything is lowercased; symbols (`.`, `(`, `)`,
   `-`, `'`, `,`, the stray `\b`) are removed; runs of spaces collapse to one:
   `wall st bears claw back into the black reuters reuters short sellers wall street s dwindling band of ultra cynics are seeing green again`
2. **Stage 2 — Quality filter.** We count the words. If a document has fewer
   than `min_words` (default `5`), it is *removed* as low quality. This article
   has ~24 words, so it passes. (AG News articles are full sentences, so almost
   nothing is removed here — see the results below.)
3. **Stage 3 — Deduplication.** We compute a 64-bit fingerprint (a hash) of the
   cleaned text. If we have already seen that exact fingerprint earlier in the
   file, this document is a *duplicate* and is dropped. Otherwise we keep it and
   remember its fingerprint.
4. **Stage 4 — Tokenization.** We split the surviving text into word-tokens and
   add the count to a running total.

A document is only written to the output file if it survives stages 2 **and** 3.

### End-to-end workflow (what YOU actually do)

```
   ┌────────────────────────┐
   │ 1. Download dataset     │  python scripts\download_agnews.py
   │    (Python, run once)    │  -> creates data\agnews\agnews_*.txt
   └────────────┬───────────┘
                │
   ┌────────────▼───────────┐
   │ 2. Build the C++ binary │  .\build.ps1
   │    (compile with g++)    │  -> creates build\seq_pipeline.exe
   └────────────┬───────────┘
                │
   ┌────────────▼───────────┐
   │ 3. Run the pipeline     │  build\seq_pipeline.exe <in> <out> <min_words>
   │    (process the data)    │  -> prints stats + writes results\clean_*.txt
   └────────────────────────┘
```

### What happens *inside* the program when you run it

`src/sequential/main_sequential.cpp` is the conductor. In order, it:

1. **Reads command-line arguments** — input file, output file, and `min_words`.
2. **Loads** the file into memory as a `std::vector<Document>` (one `Document`
   = one line), via `read_documents()`.
3. **Runs the 4 stages in turn**, each wrapped in a `Timer` so we know how long
   every stage took. Stages 2 and 3 build *new* vectors of survivors (this is
   faster than deleting from the middle of a vector).
4. **Updates the counters** in a `PipelineStats` struct: total / removed /
   duplicates / kept / total tokens.
5. **Writes** the surviving documents to the output file via
   `write_documents()`.
6. **Prints** the statistics table and per-stage timings.

### Why is the code split into so many files?

Each stage is a small, pure function in `src/common/` (it takes a string,
returns a result, and shares nothing). The "assembly line" loop that calls them
lives separately in `main_sequential.cpp`. This matters because:

- **It mirrors real software engineering** — small, single-purpose, testable
  modules instead of one giant `main`.
- **It is the foundation for Phases 2 & 3.** The parallel versions (OpenMP, MPI)
  will reuse these *exact same* stage functions and only change the loop that
  drives them. Because the logic never changes, the parallel output is
  guaranteed to match the sequential output — that is how we prove correctness.

---

## Folder structure

```
PDC Project/
├── README.md                     # this file
├── build.ps1                     # Windows build (MinGW g++): seq + openmp
├── build.sh                      # WSL/Linux build: seq + openmp + mpi + hybrid
├── Makefile                      # alternative build for make users (all targets)
├── data/
│   └── agnews/                   # real dataset: agnews_1k/10k/50k/full.txt
├── scripts/
│   ├── download_agnews.py        # downloads AG News from Hugging Face
│   ├── benchmark.ps1             # Phase 2 OpenMP thread sweep (Windows)
│   ├── verify_openmp.ps1         # Phase 2 correctness check (Windows)
│   ├── benchmark_mpi.sh          # Phase 3 MPI+hybrid sweep (WSL)
│   └── verify_mpi.sh             # Phase 3 correctness check (WSL)
├── src/
│   ├── common/                   # SHARED stage logic (reused by every version)
│   │   ├── document.hpp           #   the Document struct
│   │   ├── timer.hpp              #   chrono wall-clock timer
│   │   ├── statistics.hpp         #   counters + per-stage timings
│   │   ├── 1text_cleaning.*       #   Stage 1
│   │   ├── 2quality_filter.*      #   Stage 2
│   │   ├── 3deduplication.*       #   Stage 3
│   │   ├── 4tokenizer.*           #   Stage 4
│   │   ├── io_utils.*             #   file read/write + CSV stats
│   │   └── mpi_pipeline.hpp       #   Phase 3 shared MPI driver (mpi + hybrid)
│   ├── sequential/               # Phase 1: main_sequential.cpp ✅
│   ├── openmp/                   # Phase 2: main_openmp.cpp ✅
│   ├── mpi/                      # Phase 3: main_mpi.cpp ✅
│   └── hybrid/                   # Phase 3: main_hybrid.cpp ✅
├── results/                      # cleaned datasets + benchmark CSVs
└── docs/
    ├── report.md                 # THE complete project report (all phases + viva Q&A)
    ├── graph_openmp_speedup.png      #   performance graphs embedded in report.md
    ├── graph_openmp_efficiency.png
    ├── graph_mpi_speedup.png
    ├── graph_hybrid_speedup.png
    └── colab_finetune_distilgpt2.ipynb  # DistilGPT-2 fine-tuning notebook
```

**Why a `common/` folder?** Every stage is a pure, per-document function. The
sequential, OpenMP, MPI and hybrid versions differ only in *how they loop over
documents*, not in the stage logic. Keeping the logic in `common/` means the
parallel versions are guaranteed to compute the same result — which is exactly
how we verify correctness.

---

## Prerequisites

- **MinGW-w64 g++** on PATH (check with `g++ --version`).
- **Python 3** with the Hugging Face `datasets` package: `pip install datasets`
  (needed once, to download AG News).

If you don't have g++: install [MSYS2](https://www.msys2.org/), then in the
MSYS2 terminal run `pacman -S mingw-w64-ucrt-x86_64-gcc`, and add the
`...\ucrt64\bin` folder to your Windows PATH.

---

## Build & run (Phase 1)

### 1. Download the real dataset (AG News)

```powershell
pip install datasets
python scripts\download_agnews.py
```

This downloads ~120,000 real news articles and writes four files (all real
text) so we have several sizes to benchmark in later phases:
`data\agnews\agnews_1k.txt`, `agnews_10k.txt`, `agnews_50k.txt`,
`agnews_full.txt`. The smaller files are exact prefixes of the larger ones.

> The dataset id used is `fancyzhx/ag_news` (the canonical Hugging Face repo).
> The script falls back to the legacy `ag_news` alias automatically.

### 2. Build the sequential pipeline

```powershell
.\build.ps1
```

Or manually:

```powershell
g++ -O2 -std=c++17 -Wall src/sequential/main_sequential.cpp `
    src/common/1text_cleaning.cpp src/common/2quality_filter.cpp `
    src/common/3deduplication.cpp src/common/4tokenizer.cpp `
    src/common/io_utils.cpp -o build/seq_pipeline.exe
```

### 3. Run it

```powershell
build\seq_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5
```

Arguments: `<input_file> <output_file> [min_words]` (default `min_words = 5`).
Try the other sizes (`agnews_1k.txt`, `agnews_50k.txt`, `agnews_full.txt`) too.

---

## Build & run (Phase 2 — OpenMP)

`.\build.ps1` builds **both** `seq_pipeline.exe` and `omp_pipeline.exe`. The
OpenMP binary takes two optional extra arguments:

```powershell
# omp_pipeline <input> <output> [min_words] [threads] [csv_path]
build\omp_pipeline.exe data\agnews\agnews_full.txt results\clean_full.txt 5 8
```

`[threads]` sets the thread count (else `OMP_NUM_THREADS`, else all cores).
`[csv_path]`, if given, appends a benchmark row.

Two helper scripts:

```powershell
.\scripts\verify_openmp.ps1   # proves OpenMP output == sequential (byte-identical)
.\scripts\benchmark.ps1       # thread sweep -> results\benchmark_openmp.csv + speedup tables
```

### Understanding speedup and efficiency

These are the two numbers used to judge any parallel program:

- **Speedup** `S(p) = T(1) / T(p)` — how many times faster the program runs with
  `p` threads than with one. `S = 4` means "4× faster". The ideal is `S = p`
  (linear speedup), which real programs rarely reach.
- **Efficiency** `E(p) = S(p) / p` — the *fraction* of that ideal you actually
  got, i.e. how well each thread is being used. `E = 100%` is perfect linear
  scaling; `E = 30%` means each core is only ~30% as productive as it would be
  alone (the rest is lost to serial work and memory contention).

> **Two valid baselines.** "Speedup" can be measured against the **sequential
> binary** (`T_seq / T_omp` — the practical "what did I gain by switching?"
> number, which includes any OpenMP overhead) or against the **OpenMP binary at
> 1 thread** (`T(1) / T(p)` — which isolates pure thread-scaling). The tables
> below use the 1-thread OpenMP baseline; the worked example uses the sequential
> baseline. Both are correct — they just answer slightly different questions.

### Phase 2 results (this 12-core laptop, `min_words = 5`, best of 5 runs)

Total wall-clock time (ms), with **speedup** `S = T(1)/T(p)` and **efficiency**
`E = S/p` relative to the OpenMP 1-thread time:

| Dataset | 1 thr | 2 | 4 | 8 | 12 | Best speedup | Eff. @ best |
|---------|------:|--:|--:|--:|---:|:------------:|:-----------:|
| 10K  | 130.8  | 84.3   | 66.9  | 50.8  | 49.7  | 2.63× | 22% |
| 50K  | 686.7  | 483.2  | 415.6 | 345.0 | 329.1 | 2.09× | 17% |
| Full | 2098.3 | 1080.7 | 986.2 | 816.1 | 775.1 | **2.71×** | 23% |

Efficiency is high at low thread counts (e.g. **97%** at 2 threads on the full
set) and falls as threads are added — the classic shape of a memory-bound
workload hitting a serial floor.

### Worked example: sequential vs OpenMP-8 on the 10K dataset

Comparing one sequential run (133.56 ms total) to one OpenMP run at 8 threads
(58.00 ms total):

- **Speedup** `= 133.56 / 58.00 =` **2.30×**
- **Efficiency** `= 2.30 / 8 =` **29%**

The per-stage breakdown shows *where* the speedup comes from — and where it
doesn't:

| Stage | Seq (ms) | OMP-8 (ms) | Stage speedup |
|-------|---------:|-----------:|:-------------:|
| load     | 15.77 | 15.93 | ~1.0× (serial file I/O — no gain) |
| clean    | 49.16 | 11.72 | **4.20×** |
| filter   | 25.92 |  5.30 | **4.89×** |
| dedup    |  5.75 | ~0.00 | hashing parallelized away (sub-ms) |
| tokenize | 20.69 |  8.53 | 2.43× |
| **TOTAL**| **133.56** | **58.00** | **2.30×** |

### Why speedup is not linear (and why that's expected)

The compute stages scale well (clean/filter ~4–5×), but two effects cap the
**overall** number:

1. **A serial floor (Amdahl's law).** Loading the file is serial I/O and does
   **not** parallelize (~16 ms on 10K, ~148 ms on full, roughly constant at any
   thread count). Once the parallel stages shrink, that fixed cost becomes a
   large *share* of the total (≈27% of the 58 ms 10K run), which mathematically
   limits how fast the whole pipeline can get no matter how many cores you add.
2. **Memory-bandwidth saturation.** Each stage does very little work per byte
   (lowercase a char, test a space, multiply a hash), so it is limited by memory
   speed, not CPU. A few threads saturate the memory bus; extra cores then add
   little (note 8→12 threads barely improves the time).

This is why **bigger datasets scale better** (full: 2.71× vs 50K: 2.09×): more
documents per thread means the fixed serial costs are amortized over more useful
parallel work. It is also the empirical motivation for **Phase 3 (MPI)**, which
spreads the data — and the I/O — across processes/machines to break past this
single-node ceiling. The **counters stay identical to Phase 1** (verified
byte-for-byte). Full analysis in `docs/report.md` §6 & §8.

---

## Build & run (Phase 3 — MPI + Hybrid)

Phase 3 is built and run **under WSL (Ubuntu)** because Windows MinGW has no
`mpicxx`. WSL reads the same project files through `/mnt/d`, so nothing moves.

```bash
# one-time toolchain setup (inside WSL, in the project root)
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev

# build all four targets (seq, omp, mpi, hybrid)
./build.sh

# run MPI with 4 processes
mpirun --allow-run-as-root -np 4 build/mpi_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5

# run HYBRID with 2 processes x 4 OpenMP threads each (= 8-way)
mpirun --allow-run-as-root -np 2 build/hybrid_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5 4
```

Argument layout (both binaries): `<input> <output> [min_words] [threads] [csv_path]`.
`[threads]` sets OpenMP threads per process (hybrid only; ignored by pure MPI).
Add `--oversubscribe` to `mpirun` to launch more processes than physical cores.

Two helper scripts (run from the project root inside WSL):

```bash
./scripts/verify_mpi.sh      # proves MPI & hybrid output == sequential (28 configs)
./scripts/benchmark_mpi.sh   # process/thread sweep -> results/benchmark_mpi.csv + tables
```

### How MPI works here (master–worker, scatter → gather)

Rank 0 reads the file and **`MPI_Scatterv`s** contiguous, ordered document
batches to the workers; every rank runs the **same** `src/common/` stages on its
batch; **`MPI_Gatherv`** returns the survivors and **`MPI_Reduce`** combines the
counters; rank 0 then does the final **serial, id-ordered deduplication** and
writes the output — which is what keeps the result byte-identical to Phase 1.
The pure-MPI and hybrid builds are **the same source** (`mpi_pipeline.hpp`);
hybrid just adds `-fopenmp` to activate the per-document `#pragma omp` loops
inside each process, so they cannot diverge.

### Phase 3 results (this 12-core laptop, WSL, `min_words = 5`, best of 5)

Total wall-clock time (ms); baseline `T(1)` = MPI on 1 process; `S = T(1)/T(p)`:

| Dataset | MPI 1p | 2p | 4p | 8p | Best speedup |
|---------|-------:|---:|---:|---:|:------------:|
| 10K  | 308.4  | 343.6  | 307.9  | 285.8  | 1.08× |
| 50K  | 1478.6 | 1372.7 | 1455.3 | 1247.5 | 1.19× |
| Full | 3943.0 | 2637.7 | **2266.8** | 3162.7 | **1.74×** |

Hybrid on the full set tops out at **1.76×** (4 procs × 2 threads). The speedup
is **lower than Phase 2's OpenMP 2.71×**, and that is the headline lesson: on a
**single node** all MPI processes share one memory bus and one disk (the same
ceiling OpenMP hit) **and** MPI must *copy* data between processes, which OpenMP
(shared memory) never does. A large serial file read (~1050 ms; WSL's `/mnt/d`
9p mount is slow) further caps it by Amdahl's law. MPI's real payoff is **across
multiple machines** — P nodes give P× the memory bandwidth, disk throughput and
RAM — which a single laptop cannot show. The compute stages themselves
parallelize superbly (cleaning fell ~800 ms → 77 ms). Full analysis in
`docs/report.md` §7 & §8.

### Fine-tuning DistilGPT-2 on the cleaned corpus

`docs/colab_finetune_distilgpt2.ipynb` is a Google Colab notebook (GPU runtime)
that uploads a cleaned corpus, tokenizes it with DistilGPT-2's real BPE
tokenizer, fine-tunes the model, reports perplexity, and generates sample text.
**It has been run on a Colab T4 GPU** over `clean_10k.txt`: 1 epoch, ~44 s,
**validation perplexity ≈ 95.3**, producing coherent AG-News-style completions —
closing the loop from "messy data" to "a model trained on it." Full numbers and
sample outputs in `docs/report.md` §9.

---

## Expected output

Running on the 10K dataset prints a statistics + timing report like this:

```
==================== SEQUENTIAL ====================
Total documents      : 10000
Removed (low quality): 0
Duplicates removed   : 55
Kept documents       : 9945
Total tokens         : 402830
----------------------------------------------------
Stage timings (ms):
  load      : 43.081
  clean     : 80.977
  filter    : 47.101
  dedup     : 0.000
  tokenize  : 55.634
  TOTAL     : 259.329
====================================================
Cleaned dataset written to: results\clean_10k.txt
```

It also writes the cleaned corpus. For example, the first article changes from:

```
Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, ...
```
to:
```
wall st bears claw back into the black reuters reuters short sellers ...
```

### Measured results (this laptop, `min_words = 5`)

The **counters** are deterministic — they must stay identical for the OpenMP,
MPI and hybrid versions (only the timings should improve). Timings are
single-run and vary by machine; the ratios are what matter.

| Dataset | Total | Removed | Duplicates | Kept | Total tokens | Total time |
|---------|------:|--------:|-----------:|-----:|-------------:|-----------:|
| 1K   | 1 000   | 0 | 2   | 998     | 41 588    | ~40 ms  |
| 10K  | 10 000  | 0 | 55  | 9 945   | 402 830   | ~260 ms |
| 50K  | 50 000  | 0 | 112 | 49 888  | 1 993 026 | ~1.3 s  |
| Full | 120 000 | 0 | 218 | 119 782 | 4 750 500 | ~2.9 s  |

> **Why is "Removed" 0?** AG News articles are full sentences, so none fall
> below the 5-word quality bar — a realistic result for clean, curated data.
> Most of the runtime is spent in **cleaning**, **filtering** and
> **tokenization**, which are all per-document and so parallelize well: this is
> exactly the empirical motivation for the OpenMP phase. (Raise `min_words`,
> e.g. to `30`, to see the quality filter visibly drop documents.)

---

## Compilation commands for ALL phases (reference)

These are documented now so the build story is consistent across phases:

```powershell
# Sequential (Phase 1)
g++ -O2 -std=c++17 src/sequential/main_sequential.cpp src/common/*.cpp -o build/seq_pipeline.exe

# OpenMP (Phase 2)
g++ -O2 -std=c++17 -fopenmp src/openmp/main_openmp.cpp src/common/*.cpp -o build/omp_pipeline.exe

# MPI (Phase 3) -- under WSL / Linux with OpenMPI (no .exe; -Wno-unknown-pragmas
# silences the intentionally-ignored omp pragmas in the shared header)
mpicxx -O2 -std=c++17 -Wno-unknown-pragmas src/mpi/main_mpi.cpp src/common/*.cpp -o build/mpi_pipeline

# Hybrid MPI + OpenMP (Phase 3) -- same shared header, with -fopenmp
mpicxx -O2 -std=c++17 -fopenmp src/hybrid/main_hybrid.cpp src/common/*.cpp -o build/hybrid_pipeline
```

(`src/common/*.cpp` works with the `g++`/`mpicxx` glob on Linux/MSYS2; on plain
Windows PowerShell list the five files explicitly. `./build.sh` does all four.)

---

## Dataset source (for the viva)

The data is **AG News**, downloaded from the **Hugging Face Hub** with the
official `datasets` library via `scripts/download_agnews.py`, from the repository
**`fancyzhx/ag_news`** (~120,000 real news articles; we use the article text and
ignore the labels). Source: https://huggingface.co/datasets/fancyzhx/ag_news
See `docs/report.md` §4 for the full answer.

---

## Resuming later — where I left off

**All three phases are finished and verified.** Everything below is already on
disk and does **not** need redoing:

- `build\seq_pipeline.exe` / `build\omp_pipeline.exe` — Windows (MinGW) programs
- `build/seq_pipeline` / `omp_pipeline` / `mpi_pipeline` / `hybrid_pipeline` — WSL programs
- `data\agnews\agnews_1k/10k/50k/full.txt` — the downloaded dataset
- `results\clean_*.txt` — cleaned outputs
- `results\benchmark_openmp.csv` (Phase 2) and `results\benchmark_mpi.csv` (Phase 3)

### To re-run Phases 1 & 2 (Windows)

```powershell
build\seq_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5      # Phase 1
build\omp_pipeline.exe data\agnews\agnews_10k.txt results\clean_10k.txt 5 8    # Phase 2
.\scripts\verify_openmp.ps1                                                    # prove they match
```

(Rebuild with `.\build.ps1` only if you changed code; re-download with
`python scripts\download_agnews.py` only if `data\agnews\` is missing.)

### To re-run Phase 3 (inside WSL)

```bash
./build.sh                                                                     # builds all 4
mpirun --allow-run-as-root -np 4 build/mpi_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5                    # MPI
mpirun --allow-run-as-root -np 2 build/hybrid_pipeline \
       data/agnews/agnews_full.txt results/clean_full.txt 5 4                  # Hybrid
./scripts/verify_mpi.sh                                                        # prove they match
```

(If `mpicxx`/`mpirun` are missing:
`sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev`.)

### Optional next steps (beyond the assignment)

The pipeline is complete. If you ever extend it, the natural directions are
multi-node MPI (each rank reads its own file shard via MPI-IO to kill the serial
read bottleneck) and a distributed dedup so rank 0 is no longer a gather point —
see `docs/report.md` Appendix A.3.

---

## Where to read more

| Document | What's in it |
|----------|--------------|
| **`docs/report.md`** | **The complete project report** — one document covering all three phases: introduction, pipeline, architecture, dataset, per-phase design, performance results **with graphs**, fine-tuning, correctness, build guide, conclusion, and consolidated viva Q&A (Appendix A) |
| `docs/graph_openmp_speedup.png` | OpenMP speedup vs thread count |
| `docs/graph_openmp_efficiency.png` | OpenMP efficiency vs thread count |
| `docs/graph_mpi_speedup.png` | MPI speedup vs process count |
| `docs/graph_hybrid_speedup.png` | Hybrid speedup by procs×threads configuration |
| `docs/colab_finetune_distilgpt2.ipynb` | Colab notebook: fine-tune DistilGPT-2 on the cleaned corpus |
