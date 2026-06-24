#pragma once
#include <vector>
#include "csv_io.hpp"

struct Metrics {
    double top1_accuracy;  // fraction of races where argmax(win_prob) == actual winner
    double brier_score;    // mean((win_prob - won)^2) over all horse-race rows
    double log_loss;       // mean cross-entropy over all horse-race rows
    int n_races;
    int n_horses;
};

Metrics compute_metrics(const std::vector<SimResult>& results);

// Prints races/horses where model win_probability exceeds bookmaker implied probability.
void report_ev_bets(const std::vector<SimResult>& results);
