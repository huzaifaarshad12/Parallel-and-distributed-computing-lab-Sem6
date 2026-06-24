#pragma once
// ---------------------------------------------------------------------------
// quality_filter.hpp  (Stage 2)
// ---------------------------------------------------------------------------
// Quality filtering removes documents that are too short to be useful for LLM
// fine-tuning (e.g. fragments like "click here"). The rule is simple and
// per-document, so this stage is also independent across documents.
// ---------------------------------------------------------------------------

#include <string>
#include <cstddef>

// Counts whitespace-separated words in `text`. Assumes `text` has already been
// cleaned (single spaces, no leading/trailing space) but works on any input.
std::size_t count_words(const std::string& text);

// Returns true if the document has at least `min_words` words (i.e. it passes
// the quality bar and should be kept).
bool passes_quality(const std::string& text, std::size_t min_words);
