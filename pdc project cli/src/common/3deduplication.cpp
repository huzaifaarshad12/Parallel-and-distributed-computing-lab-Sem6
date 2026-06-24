// ---------------------------------------------------------------------------
// deduplication.cpp  (Stage 3 implementation)
// ---------------------------------------------------------------------------
#include "3deduplication.hpp"

// Standard 64-bit FNV-1a constants.
//   offset basis = 14695981039346656037
//   prime        = 1099511628211
// Algorithm: start from the offset basis; for each byte, XOR it in, then
// multiply by the prime. The XOR-before-multiply order is what makes it "1a"
// (the variant with better avalanche behaviour than plain FNV-1).
std::uint64_t fnv1a_64(const std::string& text) {
    const std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
    const std::uint64_t FNV_PRIME  = 1099511628211ULL;

    std::uint64_t hash = FNV_OFFSET;
    for (unsigned char byte : text) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= FNV_PRIME;
    }
    return hash;
}
