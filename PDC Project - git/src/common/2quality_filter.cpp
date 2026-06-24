// ---------------------------------------------------------------------------
// quality_filter.cpp  (Stage 2 implementation)
// ---------------------------------------------------------------------------
#include "2quality_filter.hpp"
#include <cctype>

// Count words by scanning for transitions from "in a space" to "in a word".
// We do not allocate any vector here (unlike a split-based approach), so this
// is cheap and allocation-free.
std::size_t count_words(const std::string& text) {
    std::size_t count = 0;
    bool in_word = false;

    for (char c : text) {
        bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (!is_space && !in_word) {
            // We just stepped from whitespace into a new word.
            ++count;
            in_word = true;
        } else if (is_space) {
            in_word = false;
        }
    }
    return count;
}

bool passes_quality(const std::string& text, std::size_t min_words) {
    return count_words(text) >= min_words;
}
