#pragma once
// ---------------------------------------------------------------------------
// io_utils.hpp
// ---------------------------------------------------------------------------
// File input/output helpers. We use a "one document per line" file format for
// the whole project. It is the simplest format that (a) parses trivially,
// (b) splits cleanly into batches for MPI in Phase 3, and (c) is human
// readable. The dataset generators guarantee no document contains a newline.
// ---------------------------------------------------------------------------

#include "document.hpp"
#include "statistics.hpp"
#include <string>
#include <vector>

// Reads `path` and returns one Document per non-empty line. The `id` field is
// the line index (0-based) among the documents that were kept. Throws
// std::runtime_error if the file cannot be opened.
std::vector<Document> read_documents(const std::string& path);

// Writes the cleaned/kept documents back out, one per line.
void write_documents(const std::string& path, const std::vector<Document>& docs);

// Appends one CSV row of stats to `path` (creating a header if the file is new).
// Used by the Phase 2/3 benchmark module, but defined here so all versions
// share it.
void append_stats_csv(const std::string& path,
                      const std::string& version,
                      int threads,
                      int processes,
                      std::size_t dataset_size,
                      const PipelineStats& stats);
