#pragma once
#include <random>
#include <vector>
#include "csv_io.hpp"

// Runs N Monte Carlo trials for one race.
// Each trial samples a Gaussian perturbation with std dev `sigma` around each
// horse's predicted_finish_time. The horse with the lowest sampled time wins
// that trial. Returns win probabilities in the same order as `horses`.
//
// `sigma` is global for now; the function signature accepts it per-call so
// callers can pass a per-distance or per-class value without code changes.
std::vector<double> simulate_race(
    const std::vector<HorseEntry>& horses,
    double sigma,
    int n_trials,
    std::mt19937_64& rng
);
