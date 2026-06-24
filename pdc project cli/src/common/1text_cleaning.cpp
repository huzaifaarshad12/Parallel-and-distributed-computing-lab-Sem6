// ---------------------------------------------------------------------------
// text_cleaning.cpp  (Stage 1 implementation)
// ---------------------------------------------------------------------------
#include "1text_cleaning.hpp"
#include <cctype>

// Single-pass cleaner. We walk the input character by character and build the
// output. Doing all three operations (lowercase / strip specials / normalize
// whitespace) in ONE pass keeps the stage cache-friendly and fast.
//
// Logic:
//   - If the character is alphanumeric, lowercase it and append it.
//   - Otherwise it is a "separator" (space, punctuation, symbol). We append a
//     single space, but only if the previous emitted character was not already
//     a space. This collapses runs of separators and removes special chars in
//     one go.
//   - Finally we strip a trailing space if one was emitted.
//
// `prev_space` starts true so any leading separators are trimmed automatically.
std::string clean_text(const std::string& input) {
    std::string out;
    out.reserve(input.size());      // avoid repeated reallocations

    bool prev_space = true;         // treat start-of-string as following a space

    for (char c : input) {
        unsigned char uc = static_cast<unsigned char>(c);

        if (std::isalnum(uc)) {
            // Keep alphanumerics, lowercased.
            out.push_back(static_cast<char>(std::tolower(uc)));
            prev_space = false;
        } else {
            // Any non-alphanumeric becomes a single separating space.
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
        }
    }

    // Remove a trailing space if the last input char was a separator.
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }

    return out;
}
