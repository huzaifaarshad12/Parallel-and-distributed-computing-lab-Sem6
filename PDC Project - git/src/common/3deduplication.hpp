#pragma once
// ---------------------------------------------------------------------------
// deduplication.hpp  (Stage 3)
// ---------------------------------------------------------------------------
// Exact-duplicate removal via document fingerprints. We hash each cleaned
// document into a 64-bit fingerprint and keep only the FIRST occurrence of each
// fingerprint. Using a fingerprint instead of comparing full strings makes the
// "have I seen this before?" check O(1) on average and tiny in memory.
//
// We implement FNV-1a (Fowler–Noll–Vo) by hand instead of std::hash<string>
// because FNV-1a gives a STABLE, well-defined value across compilers and runs.
// That stability matters in Phase 3: an MPI worker on one machine must produce
// the same fingerprint as the sequential reference on another.
// ---------------------------------------------------------------------------

#include <string>
#include <cstdint>

// 64-bit FNV-1a fingerprint of `text`.
std::uint64_t fnv1a_64(const std::string& text);
