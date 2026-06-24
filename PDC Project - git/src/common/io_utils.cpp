// ---------------------------------------------------------------------------
// io_utils.cpp
// ---------------------------------------------------------------------------
#include "io_utils.hpp"
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>

std::vector<Document> read_documents(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open input file: " + path);
    }

    std::vector<Document> docs;
    std::string line;
    std::size_t id = 0;

    while (std::getline(in, line)) {
        // Strip a trailing '\r' so Windows-style CRLF files behave the same as
        // Unix LF files (important since the dataset may be generated on either).
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;   // blank lines are not documents
        }
        docs.push_back(Document{ id++, std::move(line) });
    }
    return docs;
}

void write_documents(const std::string& path, const std::vector<Document>& docs) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open output file: " + path);
    }
    for (const Document& d : docs) {
        out << d.text << '\n';
    }
}

// Returns true if a file already exists (used to decide whether to write a CSV
// header). Implemented with stat() to avoid opening the file twice.
static bool file_exists(const std::string& path) {
    struct stat buffer;
    return ::stat(path.c_str(), &buffer) == 0;
}

void append_stats_csv(const std::string& path,
                      const std::string& version,
                      int threads,
                      int processes,
                      std::size_t dataset_size,
                      const PipelineStats& stats) {
    bool existed = file_exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }
    if (!existed) {
        out << "version,threads,processes,dataset_size,total_documents,"
               "removed_documents,duplicate_documents,kept_documents,"
               "total_tokens,time_total_ms\n";
    }
    out << version << ',' << threads << ',' << processes << ','
        << dataset_size << ',' << stats.total_documents << ','
        << stats.removed_documents << ',' << stats.duplicate_documents << ','
        << stats.kept_documents << ',' << stats.total_tokens << ','
        << stats.time_total_ms << '\n';
}
