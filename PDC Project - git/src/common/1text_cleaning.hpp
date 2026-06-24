#pragma once
// ---------------------------------------------------------------------------
// text_cleaning.hpp  (Stage 1)
// ---------------------------------------------------------------------------
// Declarations for the text-cleaning stage. The actual work is in
// text_cleaning.cpp. Cleaning is a PURE function of a single string, which is
// exactly why this stage is "embarrassingly parallel": every document can be
// cleaned independently with no shared state. Phase 2 will simply wrap the loop
// that calls clean_text() with an OpenMP pragma.
// ---------------------------------------------------------------------------

#include <string>

// Returns a cleaned copy of `input`:
//   1. lowercases every letter,
//   2. removes special characters (anything that is not a-z or 0-9),
//   3. normalizes whitespace (collapses runs of separators into one space and
//      trims leading/trailing spaces).
std::string clean_text(const std::string& input);
