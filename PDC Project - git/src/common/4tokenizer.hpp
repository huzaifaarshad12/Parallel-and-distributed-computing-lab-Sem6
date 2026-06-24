#pragma once
// ---------------------------------------------------------------------------
// tokenizer.hpp  (Stage 4)
// ---------------------------------------------------------------------------
// The final stage turns cleaned text into tokens (words). For LLM fine-tuning a
// real tokenizer (BPE/WordPiece) is used, but for a preprocessing pipeline a
// whitespace tokenizer is the correct, standard first step: it lets us count
// tokens and emit a clean word stream. The Colab notebook in Phase 3 then feeds
// these cleaned documents to DistilGPT-2's real BPE tokenizer.
// ---------------------------------------------------------------------------

#include <string>
#include <vector>
#include <cstddef>

// Splits cleaned `text` into a vector of whitespace-separated tokens.
std::vector<std::string> tokenize(const std::string& text);

// Counts tokens without materializing the vector (faster, no allocation).
// For cleaned text this equals count_words(), but we keep a dedicated function
// so the tokenization stage is conceptually self-contained.
std::size_t count_tokens(const std::string& text);
