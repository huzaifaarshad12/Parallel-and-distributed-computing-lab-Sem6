// ---------------------------------------------------------------------------
// main_openmp.cpp
// ---------------------------------------------------------------------------
// PHASE 2 DELIVERABLE: the shared-memory (OpenMP) version of the pipeline.
//
// It reuses the EXACT SAME stage functions from src/common/ as the sequential
// baseline. Only the loops that drive those functions change: the per-document
// work is spread across threads. Because the stage logic is untouched and every
// order-dependent decision is kept serial and in original-id order, this program
// is guaranteed to produce BYTE-IDENTICAL output and IDENTICAL counters to
// main_sequential.cpp. Only the timing improves. That equality is the project's
// correctness contract and is verified in Phase 2 (see docs/report.md).
//
// Parallelization strategy, stage by stage:
//   Stage 1  Cleaning      : "#pragma omp parallel for" -- each thread cleans a
//                            disjoint set of indices, writing only its own slots
//                            (no shared writes => no race).
//   Stage 2  Quality filter: parallel pass fills a per-index keep[] mask (pure,
//                            no shared writes), then a CHEAP serial compaction
//                            rebuilds the survivor vector in original order so
//                            the result is deterministic and the removed count
//                            is exact.
//   Stage 3  Deduplication : parallel pass computes the FNV-1a fingerprint of
//                            every survivor (the expensive part), then a serial
//                            pass walks indices in order and keeps the first
//                            occurrence of each fingerprint -- identical "first
//                            seen wins, by id" semantics as the sequential code.
//   Stage 4  Tokenization  : "reduction(+:token_sum)" -- each thread sums its
//                            chunk into a private partial, merged at the end.
//
// Shared vs private (the viva talking point):
//   * Read-only shared    : docs/filtered/unique vectors, min_words.
//   * Written, but disjoint: keep[i], fp[i], docs[i].text -- each index is
//                            touched by exactly one thread, so these are safe.
//   * Reduction           : token_sum -- private partial per thread, summed.
//   * Loop index i        : private by definition of "parallel for".
//   * Serial state        : the dedup `seen` set and the survivor vectors'
//                            push_back are done OUTSIDE any parallel region, so
//                            they need no locking.
//
// Usage:
//   omp_pipeline <input_file> <output_file> [min_words] [threads] [csv_path]
//   omp_pipeline data/agnews/agnews_10k.txt results/clean_10k.txt 5 4
// ---------------------------------------------------------------------------

#include "../common/document.hpp"
#include "../common/statistics.hpp"
#include "../common/timer.hpp"
#include "../common/1text_cleaning.hpp"
#include "../common/2quality_filter.hpp"
#include "../common/3deduplication.hpp"
#include "../common/4tokenizer.hpp"
#include "../common/io_utils.hpp"

#include <omp.h>

#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <string>

int main(int argc, char** argv) {
    // ----- 1. Parse command-line arguments -------------------------------
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <output_file> [min_words] [threads] [csv_path]\n";
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const std::size_t min_words   = (argc >= 4)
                                        ? static_cast<std::size_t>(std::stoul(argv[3]))
                                        : 5; // sensible default quality bar

    // Optional explicit thread count. If omitted, OpenMP uses its default
    // (OMP_NUM_THREADS env var, else the number of logical cores).
    if (argc >= 5) {
        const int requested = std::stoi(argv[4]);
        if (requested > 0) {
            omp_set_num_threads(requested);
        }
    }
    const std::string csv_path = (argc >= 6) ? argv[5] : std::string();

    // The number of threads OpenMP will actually use for the parallel regions.
    const int threads_used = omp_get_max_threads();

    PipelineStats stats;
    Timer total_timer;   // measures the whole run

    try {
        // ----- 2. Load documents -----------------------------------------
        // File I/O is inherently serial (one stream); this is part of the
        // "serial fraction" that bounds the achievable speedup (Amdahl's law).
        Timer t;
        std::vector<Document> docs = read_documents(input_path);
        stats.time_load_ms = t.elapsed_ms();
        stats.total_documents = docs.size();

        // We use a signed loop counter throughout: the classic OpenMP-portable
        // form for "parallel for" (some compilers reject unsigned/size_t).
        const long long n_docs = static_cast<long long>(docs.size());

        // ----- Stage 1: cleaning (PARALLEL) ------------------------------
        // Each iteration overwrites only docs[i].text, so thread i and thread j
        // never touch the same memory -> no synchronization needed.
        t.reset();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < n_docs; ++i) {
            docs[i].text = clean_text(docs[i].text);
        }
        stats.time_clean_ms = t.elapsed_ms();

        // ----- Stage 2: quality filtering (PARALLEL mask + serial compact) -
        // Phase 1 built the survivor vector inside the loop, which is an
        // order-dependent operation we cannot parallelize directly. Instead we
        // split it: compute the keep/drop decision for every document in
        // parallel (pure, disjoint writes), then compact serially in index
        // order. The serial compaction is O(n) pointer work and cheap.
        t.reset();
        std::vector<char> keep(docs.size());
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < n_docs; ++i) {
            keep[i] = passes_quality(docs[i].text, min_words) ? 1 : 0;
        }
        std::vector<Document> filtered;
        filtered.reserve(docs.size());
        for (long long i = 0; i < n_docs; ++i) {
            if (keep[i]) {
                filtered.push_back(std::move(docs[i]));
            } else {
                ++stats.removed_documents;
            }
        }
        stats.time_filter_ms = t.elapsed_ms();

        // ----- Stage 3: deduplication (PARALLEL hash + serial dedup) ------
        // The expensive part of dedup is hashing every document; that is pure
        // and parallelizable. The decision "is this the first time I've seen
        // this fingerprint?" depends on earlier documents, so it must stay
        // serial to reproduce the sequential "first occurrence by id" result.
        t.reset();
        const long long n_filtered = static_cast<long long>(filtered.size());
        std::vector<std::uint64_t> fps(filtered.size());
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < n_filtered; ++i) {
            fps[i] = fnv1a_64(filtered[i].text);
        }
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(filtered.size() * 2);
        std::vector<Document> unique;
        unique.reserve(filtered.size());
        for (long long i = 0; i < n_filtered; ++i) {
            if (seen.insert(fps[i]).second) {
                unique.push_back(std::move(filtered[i])); // first occurrence
            } else {
                ++stats.duplicate_documents;               // duplicate -> drop
            }
        }
        stats.time_dedup_ms = t.elapsed_ms();

        // ----- Stage 4: tokenization / counting (PARALLEL reduction) ------
        // Textbook reduction: each thread accumulates a private partial sum of
        // token counts; OpenMP combines the partials into token_sum at the end.
        // Integer addition is associative, so the total is identical regardless
        // of how the work is split.
        t.reset();
        const long long n_unique = static_cast<long long>(unique.size());
        unsigned long long token_sum = 0;
        #pragma omp parallel for schedule(static) reduction(+:token_sum)
        for (long long i = 0; i < n_unique; ++i) {
            token_sum += count_tokens(unique[i].text);
        }
        stats.total_tokens = token_sum;
        stats.kept_documents = unique.size();
        stats.time_tokens_ms = t.elapsed_ms();

        // ----- 3. Write cleaned dataset ----------------------------------
        write_documents(output_path, unique);

        stats.time_total_ms = total_timer.elapsed_ms();
        stats.print("OPENMP (" + std::to_string(threads_used) + " threads)");
        std::cout << "Cleaned dataset written to: " << output_path << "\n";

        // ----- 4. Optional benchmark logging -----------------------------
        // When a CSV path is supplied (used by the benchmark script), append a
        // row so we can build the speedup/efficiency tables. processes = 1
        // because OpenMP is single-process, multi-threaded.
        if (!csv_path.empty()) {
            append_stats_csv(csv_path, "openmp", threads_used, /*processes=*/1,
                             stats.total_documents, stats);
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
