#include "metrics.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>

// Clip probabilities away from 0/1 before taking log to avoid -inf.
static constexpr double LOG_CLIP = 1e-7;

// Build a race-keyed index in insertion order so results stay stable across runs.
static std::map<std::string, std::vector<const SimResult*>> group_by_race(
    const std::vector<SimResult>& results)
{
    std::map<std::string, std::vector<const SimResult*>> by_race;
    for (const auto& r : results) {
        by_race[r.race_id].push_back(&r);
    }
    return by_race;
}

Metrics compute_metrics(const std::vector<SimResult>& results) {
    auto by_race = group_by_race(results);

    int correct      = 0;
    double brier_sum = 0.0;
    double ll_sum    = 0.0;
    int total_horses = 0;

    for (const auto& [race_id, horses] : by_race) {
        // Argmax win probability
        const SimResult* predicted_winner = *std::max_element(
            horses.begin(), horses.end(),
            [](const SimResult* a, const SimResult* b) {
                return a->win_probability < b->win_probability;
            });

        // Actual winner (won == 1)
        const SimResult* actual_winner = nullptr;
        for (const auto* h : horses) {
            if (h->won == 1) { actual_winner = h; break; }
        }

        if (actual_winner && predicted_winner->horse_id == actual_winner->horse_id) {
            ++correct;
        }

        for (const auto* h : horses) {
            double p = h->win_probability;
            double y = static_cast<double>(h->won);
            brier_sum += (p - y) * (p - y);
            ll_sum    += -(y * std::log(std::max(p, LOG_CLIP)) +
                          (1.0 - y) * std::log(std::max(1.0 - p, LOG_CLIP)));
            ++total_horses;
        }
    }

    int n_races = static_cast<int>(by_race.size());
    return {
        n_races      > 0 ? static_cast<double>(correct) / n_races : 0.0,
        total_horses > 0 ? brier_sum / total_horses : 0.0,
        total_horses > 0 ? ll_sum    / total_horses : 0.0,
        n_races,
        total_horses
    };
}

void report_ev_bets(const std::vector<SimResult>& results) {
    auto by_race = group_by_race(results);

    int positive_ev_selections = 0;
    int races_with_any_ev_bet  = 0;

    for (const auto& [race_id, horses] : by_race) {
        // Normalize raw implied probabilities to remove the bookmaker overround.
        double raw_sum = 0.0;
        for (const auto* h : horses) {
            if (h->win_odds > 0.0) raw_sum += 1.0 / h->win_odds;
        }
        if (raw_sum == 0.0) continue;

        bool any_ev = false;
        for (const auto* h : horses) {
            if (h->win_odds <= 0.0) continue;
            double implied = (1.0 / h->win_odds) / raw_sum;
            if (h->win_probability > implied) {
                ++positive_ev_selections;
                any_ev = true;
            }
        }
        if (any_ev) ++races_with_any_ev_bet;
    }

    int n_races = static_cast<int>(by_race.size());
    std::cout << "\n--- Expected Value Betting ---\n";
    std::cout << "Races with >= 1 positive-EV horse: "
              << races_with_any_ev_bet << " / " << n_races
              << " (" << (n_races > 0 ? 100.0 * races_with_any_ev_bet / n_races : 0.0) << "%)\n";
    std::cout << "Total positive-EV selections:      " << positive_ev_selections << "\n";
    std::cout << "(A positive-EV horse has win_probability > normalized implied probability from odds.)\n";
}
