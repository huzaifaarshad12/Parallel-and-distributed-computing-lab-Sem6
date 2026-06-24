#pragma once
// ---------------------------------------------------------------------------
// statistics.hpp
// ---------------------------------------------------------------------------
// Counters produced by one run of the pipeline, plus per-stage timings.
// This is what we print at the end and what we compare across versions to
// verify correctness (the counters must be IDENTICAL for sequential / OpenMP /
// MPI / hybrid runs on the same input; only the timings change).
// ---------------------------------------------------------------------------

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <string>

struct PipelineStats {
    // --- correctness counters (must match across all versions) ---
    std::size_t        total_documents     = 0; // documents read from input
    std::size_t        removed_documents   = 0; // dropped by quality filter
    std::size_t        duplicate_documents = 0; // dropped by deduplication
    std::size_t        kept_documents      = 0; // survived all stages
    unsigned long long total_tokens        = 0; // tokens across kept documents

    // --- performance timings in milliseconds (informational only) ---
    double time_load_ms    = 0.0;
    double time_clean_ms   = 0.0;
    double time_filter_ms  = 0.0;
    double time_dedup_ms   = 0.0;
    double time_tokens_ms  = 0.0;
    double time_total_ms   = 0.0;

    // Human-readable report printed at the end of a run.
    void print(const std::string& label) const {
        std::cout << "\n==================== " << label
                  << " ====================\n";
        std::cout << "Total documents      : " << total_documents     << "\n";
        std::cout << "Removed (low quality): " << removed_documents    << "\n";
        std::cout << "Duplicates removed   : " << duplicate_documents  << "\n";
        std::cout << "Kept documents       : " << kept_documents       << "\n";
        std::cout << "Total tokens         : " << total_tokens         << "\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Stage timings (ms):\n";
        std::cout << "  load      : " << time_load_ms   << "\n";
        std::cout << "  clean     : " << time_clean_ms  << "\n";
        std::cout << "  filter    : " << time_filter_ms << "\n";
        std::cout << "  dedup     : " << time_dedup_ms  << "\n";
        std::cout << "  tokenize  : " << time_tokens_ms << "\n";
        std::cout << "  TOTAL     : " << time_total_ms  << "\n";
        std::cout << "====================================================\n";
    }
};
