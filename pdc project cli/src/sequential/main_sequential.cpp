// ---------------------------------------------------------------------------
// main_sequential.cpp
// ---------------------------------------------------------------------------
// PHASE 1 DELIVERABLE: the complete sequential baseline.
//
// This program runs the full 4-stage preprocessing pipeline on a text file and
// prints statistics + per-stage timings. It is the "reference" implementation:
// every later version (OpenMP, MPI, hybrid) must reproduce the SAME counters on
// the same input. Only the timing should improve.
//
// Pipeline:
//   Stage 1  Cleaning      : lowercase, strip specials, normalize whitespace
//   Stage 2  Quality filter: drop documents with fewer than min_words words
//   Stage 3  Deduplication : drop documents whose fingerprint was already seen
//   Stage 4  Tokenization  : count tokens across the surviving documents
//
// Usage:
//   seq_pipeline <input_file> <output_file> [min_words]
//   seq_pipeline data/agnews/agnews_10k.txt results/clean_10k.txt 5
// ---------------------------------------------------------------------------

#include "../common/document.hpp"
#include "../common/statistics.hpp"
#include "../common/timer.hpp"
#include "../common/1text_cleaning.hpp"
#include "../common/2quality_filter.hpp"
#include "../common/3deduplication.hpp"
#include "../common/4tokenizer.hpp"
#include "../common/io_utils.hpp"

#include <iostream>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <string>

int main(int argc, char** argv) {
    // ----- 1. Parse command-line arguments -------------------------------
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_file> <output_file> [min_words]\n";
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const std::size_t min_words   = (argc >= 4)
                                        ? static_cast<std::size_t>(std::stoul(argv[3]))
                                        : 5; // sensible default quality bar

    PipelineStats stats;
    Timer total_timer;   // measures the whole run

    try {
        // ----- 2. Load documents -----------------------------------------
        Timer t;
        std::vector<Document> docs = read_documents(input_path);
        stats.time_load_ms = t.elapsed_ms();
        stats.total_documents = docs.size();

        // ----- Stage 1: cleaning -----------------------------------------
        // Each document is cleaned independently and in place. This is the loop
        // Phase 2 will parallelize with "#pragma omp parallel for".
        t.reset();
        for (std::size_t i = 0; i < docs.size(); ++i) {
            docs[i].text = clean_text(docs[i].text);
        }
        stats.time_clean_ms = t.elapsed_ms();

        // ----- Stage 2: quality filtering --------------------------------
        // Keep only documents with >= min_words words. We build a new vector
        // rather than erasing in place (erasing from the middle of a vector is
        // O(n) each time). `removed_documents` counts what we dropped.
        t.reset();
        std::vector<Document> filtered;
        filtered.reserve(docs.size());
        for (const Document& d : docs) {
            if (passes_quality(d.text, min_words)) {
                filtered.push_back(d);
            } else {
                ++stats.removed_documents;
            }
        }
        stats.time_filter_ms = t.elapsed_ms();

        // ----- Stage 3: deduplication ------------------------------------
        // Fingerprint each surviving document and keep only first occurrences.
        // `seen` is the shared state that makes this stage interesting to
        // parallelize later (Phase 2 handles it with per-thread sets merged at
        // the end; the ORDER of "first occurrence" must follow original id).
        t.reset();
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(filtered.size() * 2);
        std::vector<Document> unique;
        unique.reserve(filtered.size());
        for (const Document& d : filtered) {
            std::uint64_t fp = fnv1a_64(d.text);
            if (seen.insert(fp).second) {
                unique.push_back(d);          // first time we see this fingerprint
            } else {
                ++stats.duplicate_documents;  // duplicate -> drop
            }
        }
        stats.time_dedup_ms = t.elapsed_ms();

        // ----- Stage 4: tokenization / counting --------------------------
        // Count tokens across the documents that survived. This is a textbook
        // reduction (sum), which Phase 2 expresses with "reduction(+:...)".
        t.reset();
        unsigned long long token_sum = 0;
        for (const Document& d : unique) {
            token_sum += count_tokens(d.text);
        }
        stats.total_tokens = token_sum;
        stats.kept_documents = unique.size();
        stats.time_tokens_ms = t.elapsed_ms();

        // ----- 3. Write cleaned dataset ----------------------------------
        write_documents(output_path, unique);

        stats.time_total_ms = total_timer.elapsed_ms();
        stats.print("SEQUENTIAL");
        std::cout << "Cleaned dataset written to: " << output_path << "\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
