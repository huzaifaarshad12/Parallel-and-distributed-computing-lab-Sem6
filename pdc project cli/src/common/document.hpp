#pragma once
// ---------------------------------------------------------------------------
// document.hpp
// ---------------------------------------------------------------------------
// The single data structure that flows through every stage of the pipeline.
//
// We keep it deliberately tiny: a stable original id (so we can always trace a
// document back to its position in the raw input) and the text itself. As the
// document moves through the stages, `text` is overwritten with its cleaned
// version. The same struct is reused unchanged by the sequential, OpenMP, MPI
// and hybrid versions of the project.
// ---------------------------------------------------------------------------

#include <string>
#include <cstddef>

struct Document {
    std::size_t id;     // 0-based index of this document in the original input
    std::string text;   // raw text on input; cleaned text after stage 1
};
