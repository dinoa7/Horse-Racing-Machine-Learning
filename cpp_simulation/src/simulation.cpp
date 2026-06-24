#include "simulation.hpp"
#include <algorithm>

std::vector<double> simulate_race(
    const std::vector<HorseEntry>& horses,
    double sigma,
    int n_trials,
    std::mt19937_64& rng
) {
    const int n = static_cast<int>(horses.size());
    std::vector<int> win_counts(n, 0);
    std::normal_distribution<double> noise(0.0, sigma);

    for (int trial = 0; trial < n_trials; ++trial) {
        int winner = 0;
        double best = horses[0].predicted_finish_time + noise(rng);
        for (int i = 1; i < n; ++i) {
            double t = horses[i].predicted_finish_time + noise(rng);
            if (t < best) {
                best = t;
                winner = i;
            }
        }
        ++win_counts[winner];
    }

    std::vector<double> probs(n);
    for (int i = 0; i < n; ++i) {
        probs[i] = static_cast<double>(win_counts[i]) / n_trials;
    }
    return probs;
}
