#pragma once
// ---------------------------------------------------------------------------
// mpi_pipeline.hpp  --  shared MPI (and MPI+OpenMP) driver for Phase 3
// ---------------------------------------------------------------------------
// This header holds the ONE implementation of the distributed pipeline. Both
// Phase 3 binaries include it:
//
//   * src/mpi/main_mpi.cpp        compiled WITHOUT -fopenmp  -> pure MPI
//   * src/hybrid/main_hybrid.cpp  compiled WITH    -fopenmp  -> MPI + OpenMP
//
// The "#pragma omp parallel for" directives below are simply ignored by the
// compiler when -fopenmp is absent, so the MPI binary runs the exact same code
// path serially inside each process, while the hybrid binary additionally
// spreads each process's chunk across threads. Keeping a single source of truth
// is what guarantees the two versions compute IDENTICAL results -- the same
// principle that keeps every stage in src/common/.
//
// --- Correctness contract -------------------------------------------------
// The output file and the five counters MUST be byte-/value-identical to the
// sequential baseline (src/sequential/main_sequential.cpp) for any number of
// processes and threads. We achieve this exactly as the OpenMP version does:
//   * every per-document stage (clean / quality / fingerprint / token-count)
//     is a pure function, so distributing it changes nothing;
//   * the only ORDER-DEPENDENT decisions -- "is this the first time I have seen
//     this fingerprint?" (dedup) and the final write order -- are performed
//     SERIALLY on the master (rank 0), walking documents in original id order.
//
// --- Communication design (master-worker, scatter/gather) -----------------
//   1. Rank 0 reads the whole file (single reader, like the sequential code).
//   2. The documents are split into contiguous, ordered blocks -- one per rank.
//      Because the blocks are contiguous, "rank order" == "original id order",
//      which is what makes the final merge trivially ordered.
//   3. MPI_Scatterv distributes each rank's block of raw text (a length array
//      + a packed byte blob).
//   4. Each rank runs stages 1-2 (clean, quality filter) on its block and
//      computes, for every survivor, its 64-bit fingerprint and token count
//      (stages 3a, 4 -- the parallelizable parts).
//   5. MPI_Gatherv returns every survivor (id, fingerprint, token count, text)
//      to rank 0; MPI_Reduce sums the per-rank "removed" counts and takes the
//      max per-stage time (the slowest worker bounds the wall time).
//   6. Rank 0 performs the serial, id-ordered deduplication (stage 3b), sums
//      the token counts of the survivors, and writes the output -- reproducing
//      the sequential result exactly.
// ---------------------------------------------------------------------------

// We use only MPI's C API. Skipping the (deprecated) C++ bindings avoids a pile
// of -Wcast-function-type warnings from OpenMPI's mpicxx.h and is the
// recommended way to include <mpi.h> from C++.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include "document.hpp"
#include "statistics.hpp"
#include "1text_cleaning.hpp"
#include "2quality_filter.hpp"
#include "3deduplication.hpp"
#include "4tokenizer.hpp"
#include "io_utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Fixed-size record describing one document that survived the quality filter.
// POD with four 8-byte fields => no padding => safe to ship as raw MPI_BYTE
// between ranks of the same architecture (all our ranks share one machine).
struct SurvMeta {
    std::uint64_t id;    // original document id (for ordering / dedup)
    std::uint64_t fp;    // FNV-1a fingerprint of the cleaned text
    std::uint64_t ntok;  // token count of the cleaned text
    std::uint64_t len;   // byte length of the cleaned text in the gathered blob
};

// Runs the full distributed pipeline. `label` is printed in the report header
// ("MPI" or "HYBRID"). Returns a process exit code. Call MPI_Init via this
// function (it owns Init/Finalize) so the thin mains stay tiny.
inline int run_pipeline_mpi(int argc, char** argv, const std::string& label) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ----- 1. Parse arguments (identical on every rank) ------------------
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <input_file> <output_file> [min_words] [threads] [csv_path]\n";
        }
        MPI_Finalize();
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string output_path = argv[2];
    const std::size_t min_words   = (argc >= 4)
                                        ? static_cast<std::size_t>(std::stoul(argv[3]))
                                        : 5;
    const int requested_threads   = (argc >= 5) ? std::stoi(argv[4]) : 0;
    const std::string csv_path    = (argc >= 6) ? argv[5] : std::string();

#ifdef _OPENMP
    if (requested_threads > 0) {
        omp_set_num_threads(requested_threads);
    }
    const int threads_used = omp_get_max_threads();
#else
    (void)requested_threads;            // ignored in the pure-MPI build
    const int threads_used = 1;
#endif

    PipelineStats stats;
    const double t_total_start = MPI_Wtime();

    // ----- 2. Rank 0 reads the whole file --------------------------------
    std::vector<Document> all_docs;
    long long N = 0;
    if (rank == 0) {
        try {
            const double t0 = MPI_Wtime();
            all_docs = read_documents(input_path);
            stats.time_load_ms = (MPI_Wtime() - t0) * 1000.0;
        } catch (const std::exception& ex) {
            std::cerr << "ERROR: " << ex.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = static_cast<long long>(all_docs.size());
    }
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // ----- 3. Block partition (every rank computes its own range) --------
    // Contiguous blocks => global id order is preserved across ranks.
    const long long base = (size > 0) ? N / size : 0;
    const long long rem  = (size > 0) ? N % size : 0;
    auto block_count = [&](int r) -> long long { return base + (r < rem ? 1 : 0); };
    const long long my_count = block_count(rank);
    long long my_offset = 0;
    for (int r = 0; r < rank; ++r) my_offset += block_count(r);

    // ----- 4. Scatter the raw text -------------------------------------
    // First scatter per-document byte lengths, then scatter the packed text.
    std::vector<int> doc_counts, doc_displs;     // (root only) docs per rank
    std::vector<int> sendlens;                   // (root only) length of each doc
    std::vector<char> sendtext;                  // (root only) packed text blob
    std::vector<int> byte_counts, byte_displs;   // (root only) bytes per rank
    if (rank == 0) {
        doc_counts.resize(size);
        doc_displs.resize(size);
        long long off = 0;
        for (int r = 0; r < size; ++r) {
            doc_counts[r] = static_cast<int>(block_count(r));
            doc_displs[r] = static_cast<int>(off);
            off += block_count(r);
        }
        sendlens.resize(N);
        long long total_bytes = 0;
        for (long long i = 0; i < N; ++i) {
            sendlens[i] = static_cast<int>(all_docs[i].text.size());
            total_bytes += sendlens[i];
        }
        sendtext.resize(static_cast<std::size_t>(total_bytes));
        long long pos = 0;
        for (long long i = 0; i < N; ++i) {
            std::memcpy(&sendtext[pos], all_docs[i].text.data(), sendlens[i]);
            pos += sendlens[i];
        }
        byte_counts.resize(size);
        byte_displs.resize(size);
        long long bpos = 0;
        for (int r = 0; r < size; ++r) {
            long long b = 0;
            for (int i = doc_displs[r]; i < doc_displs[r] + doc_counts[r]; ++i) b += sendlens[i];
            byte_counts[r] = static_cast<int>(b);
            byte_displs[r] = static_cast<int>(bpos);
            bpos += b;
        }
    }

    std::vector<int> my_lens(static_cast<std::size_t>(my_count));
    MPI_Scatterv(rank == 0 ? sendlens.data()   : nullptr,
                 rank == 0 ? doc_counts.data()  : nullptr,
                 rank == 0 ? doc_displs.data()  : nullptr,
                 MPI_INT,
                 my_lens.data(), static_cast<int>(my_count), MPI_INT,
                 0, MPI_COMM_WORLD);

    long long my_bytes = 0;
    for (long long i = 0; i < my_count; ++i) my_bytes += my_lens[i];
    std::vector<char> my_text(static_cast<std::size_t>(my_bytes));
    MPI_Scatterv(rank == 0 ? sendtext.data()    : nullptr,
                 rank == 0 ? byte_counts.data()  : nullptr,
                 rank == 0 ? byte_displs.data()  : nullptr,
                 MPI_CHAR,
                 my_text.data(), static_cast<int>(my_bytes), MPI_CHAR,
                 0, MPI_COMM_WORLD);

    // Reconstruct this rank's Document block (ids follow the global order).
    std::vector<Document> docs(static_cast<std::size_t>(my_count));
    {
        long long pos = 0;
        for (long long i = 0; i < my_count; ++i) {
            docs[i].id = static_cast<std::size_t>(my_offset + i);
            docs[i].text.assign(&my_text[pos], static_cast<std::size_t>(my_lens[i]));
            pos += my_lens[i];
        }
    }

    // ----- Stage 1: cleaning (parallel within the rank) ------------------
    double tic = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < my_count; ++i) {
        docs[i].text = clean_text(docs[i].text);
    }
    const double clean_ms = (MPI_Wtime() - tic) * 1000.0;

    // ----- Stage 2: quality filter (parallel mask + serial compact) ------
    tic = MPI_Wtime();
    std::vector<char> keep(static_cast<std::size_t>(my_count));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < my_count; ++i) {
        keep[i] = passes_quality(docs[i].text, min_words) ? 1 : 0;
    }
    std::vector<Document> filtered;
    filtered.reserve(static_cast<std::size_t>(my_count));
    long long local_removed = 0;
    for (long long i = 0; i < my_count; ++i) {
        if (keep[i]) filtered.push_back(std::move(docs[i]));
        else         ++local_removed;
    }
    const double filter_ms = (MPI_Wtime() - tic) * 1000.0;

    // ----- Stage 3a + 4: fingerprint + token count (parallel) ------------
    const long long nf = static_cast<long long>(filtered.size());
    std::vector<std::uint64_t> fps(static_cast<std::size_t>(nf));
    std::vector<std::uint64_t> toks(static_cast<std::size_t>(nf));

    tic = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < nf; ++i) {
        fps[i] = fnv1a_64(filtered[i].text);
    }
    const double hash_ms = (MPI_Wtime() - tic) * 1000.0;

    tic = MPI_Wtime();
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < nf; ++i) {
        toks[i] = static_cast<std::uint64_t>(count_tokens(filtered[i].text));
    }
    const double token_ms = (MPI_Wtime() - tic) * 1000.0;

    // Pack this rank's survivors into a metadata array + a text blob.
    std::vector<SurvMeta> meta(static_cast<std::size_t>(nf));
    std::string blob;
    {
        std::size_t total = 0;
        for (long long i = 0; i < nf; ++i) total += filtered[i].text.size();
        blob.reserve(total);
        for (long long i = 0; i < nf; ++i) {
            meta[i].id   = static_cast<std::uint64_t>(filtered[i].id);
            meta[i].fp   = fps[i];
            meta[i].ntok = toks[i];
            meta[i].len  = static_cast<std::uint64_t>(filtered[i].text.size());
            blob += filtered[i].text;
        }
    }

    // ----- 5. Gather survivors + reduce counters/timings to rank 0 -------
    const int my_surv       = static_cast<int>(nf);
    const int my_meta_bytes = my_surv * static_cast<int>(sizeof(SurvMeta));
    const int my_blob_bytes = static_cast<int>(blob.size());

    std::vector<int> surv_counts, blob_counts;
    if (rank == 0) { surv_counts.resize(size); blob_counts.resize(size); }
    MPI_Gather(&my_surv,       1, MPI_INT, rank == 0 ? surv_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_blob_bytes, 1, MPI_INT, rank == 0 ? blob_counts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<SurvMeta> all_meta;
    std::vector<char>     all_blob;
    std::vector<int> meta_counts, meta_displs, blob_displs;
    if (rank == 0) {
        meta_counts.resize(size);
        meta_displs.resize(size);
        blob_displs.resize(size);
        long long tot_surv = 0, tot_blob = 0;
        for (int r = 0; r < size; ++r) {
            meta_counts[r] = surv_counts[r] * static_cast<int>(sizeof(SurvMeta));
            meta_displs[r] = static_cast<int>(tot_surv * sizeof(SurvMeta));
            tot_surv += surv_counts[r];
            blob_displs[r] = static_cast<int>(tot_blob);
            tot_blob += blob_counts[r];
        }
        all_meta.resize(static_cast<std::size_t>(tot_surv));
        all_blob.resize(static_cast<std::size_t>(tot_blob));
    }

    MPI_Gatherv(meta.data(), my_meta_bytes, MPI_BYTE,
                rank == 0 ? all_meta.data()  : nullptr,
                rank == 0 ? meta_counts.data(): nullptr,
                rank == 0 ? meta_displs.data(): nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(blob.data(), my_blob_bytes, MPI_BYTE,
                rank == 0 ? all_blob.data()  : nullptr,
                rank == 0 ? blob_counts.data(): nullptr,
                rank == 0 ? blob_displs.data(): nullptr,
                MPI_BYTE, 0, MPI_COMM_WORLD);

    long long total_removed = 0;
    MPI_Reduce(&local_removed, &total_removed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double clean_max = 0, filter_max = 0, hash_max = 0, token_max = 0;
    MPI_Reduce(&clean_ms,  &clean_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&filter_ms, &filter_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hash_ms,   &hash_max,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&token_ms,  &token_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ----- 6. Master: serial id-ordered dedup + write --------------------
    int exit_code = 0;
    if (rank == 0) {
        const long long M = static_cast<long long>(all_meta.size());

        // Byte offset of each gathered survivor inside all_blob.
        std::vector<long long> boff(static_cast<std::size_t>(M));
        {
            long long acc = 0;
            for (long long i = 0; i < M; ++i) { boff[i] = acc; acc += all_meta[i].len; }
        }

        // Gathered order is already global-id order (contiguous blocks gathered
        // in rank order). We stable-sort by id anyway so correctness does not
        // depend on that reasoning -- a cheap, defensive guarantee.
        std::vector<long long> order(static_cast<std::size_t>(M));
        for (long long i = 0; i < M; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](long long a, long long b) { return all_meta[a].id < all_meta[b].id; });

        const double tdd = MPI_Wtime();
        std::unordered_set<std::uint64_t> seen;
        seen.reserve(static_cast<std::size_t>(M) * 2);
        std::vector<long long> kept;
        kept.reserve(static_cast<std::size_t>(M));
        unsigned long long token_sum = 0;
        long long dups = 0;
        for (long long k = 0; k < M; ++k) {
            const long long idx = order[k];
            if (seen.insert(all_meta[idx].fp).second) {
                kept.push_back(idx);
                token_sum += all_meta[idx].ntok;
            } else {
                ++dups;
            }
        }
        const double dedup_decide_ms = (MPI_Wtime() - tdd) * 1000.0;

        // Write survivors in id order -- byte-identical to write_documents().
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            std::cerr << "ERROR: Could not open output file: " << output_path << "\n";
            exit_code = 1;
        } else {
            for (long long idx : kept) {
                out.write(&all_blob[boff[idx]], static_cast<std::streamsize>(all_meta[idx].len));
                out.put('\n');
            }
        }

        stats.total_documents     = static_cast<std::size_t>(N);
        stats.removed_documents   = static_cast<std::size_t>(total_removed);
        stats.duplicate_documents = static_cast<std::size_t>(dups);
        stats.kept_documents      = kept.size();
        stats.total_tokens        = token_sum;
        stats.time_clean_ms       = clean_max;
        stats.time_filter_ms      = filter_max;
        stats.time_dedup_ms       = hash_max + dedup_decide_ms;  // parallel hash + serial decide
        stats.time_tokens_ms      = token_max;
        stats.time_total_ms       = (MPI_Wtime() - t_total_start) * 1000.0;

        std::string header = label + " (" + std::to_string(size) + " procs";
#ifdef _OPENMP
        header += " x " + std::to_string(threads_used) + " threads";
#endif
        header += ")";
        stats.print(header);
        std::cout << "Cleaned dataset written to: " << output_path << "\n";

        if (!csv_path.empty()) {
#ifdef _OPENMP
            const std::string version = "hybrid";
#else
            const std::string version = "mpi";
#endif
            append_stats_csv(csv_path, version, threads_used, size,
                             stats.total_documents, stats);
        }
    }

    MPI_Finalize();
    return exit_code;
}
