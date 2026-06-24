


#pragma once
// ---------------------------------------------------------------------------
// timer.hpp
// ---------------------------------------------------------------------------
// A very small wall-clock timer built on <chrono>. We use wall-clock time
// (high_resolution_clock) rather than CPU time because for parallel code the
// number we care about is "how long did the user wait", which is exactly what
// wall-clock measures. The same timer is reused in every phase so that the
// sequential, OpenMP, MPI and hybrid timings are directly comparable.
// ---------------------------------------------------------------------------

#include <chrono>

class Timer {
public:
    Timer() { reset(); }

    // Restart the timer from "now".
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    // Elapsed milliseconds since construction or the last reset().
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    // Elapsed seconds (convenience).
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};
