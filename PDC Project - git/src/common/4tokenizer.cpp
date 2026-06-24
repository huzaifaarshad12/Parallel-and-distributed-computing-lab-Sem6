// ---------------------------------------------------------------------------
// tokenizer.cpp  (Stage 4 implementation)
// ---------------------------------------------------------------------------
#include "4tokenizer.hpp"
#include <sstream>
#include <cctype>

std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    // operator>> on a stream already skips whitespace and splits on it, which
    // is exactly whitespace tokenization.
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

std::size_t count_tokens(const std::string& text) {
    // Same word-boundary scan used by the quality filter: a token starts at
    // every transition from whitespace into a non-whitespace character.
    std::size_t count = 0;
    bool in_word = false;
    for (char c : text) {
        bool is_space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (!is_space && !in_word) {
            ++count;
            in_word = true;
        } else if (is_space) {
            in_word = false;
        }
    }
    return count;
}
