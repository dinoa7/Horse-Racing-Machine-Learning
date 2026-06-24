#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "csv_io.hpp"
#include "metrics.hpp"
#include "simulation.hpp"

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --input      PATH   sim_input.csv path  "
           "(default: ../general_dataset/sim_input.csv)\n"
        << "  --output     PATH   output CSV path     "
           "(default: ../general_dataset/sim_output.csv)\n"
        << "  --sigma      FLOAT  residual std dev in seconds (overrides --sigma-file)\n"
        << "  --sigma-file PATH   file containing residual sigma "
           "(default: ../general_dataset/residual_sigma.txt)\n"
        << "  --trials     N      Monte Carlo trials per race (default: 10000)\n"
        << "  --seed       N      RNG seed for reproducibility  (default: 42)\n"
        << "  --help              Print this message\n";
}

int main(int argc, char* argv[]) {
    std::string input_path  = "../general_dataset/sim_input.csv";
    std::string output_path = "../general_dataset/sim_output.csv";
    std::string sigma_file  = "../general_dataset/residual_sigma.txt";
    double      sigma       = -1.0;   // sentinel: read from file
    int         n_trials    = 10000;
    uint64_t    seed        = 42;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for argument: " << arg << "\n";
            return 1;
        }
        if      (arg == "--input")      input_path  = argv[++i];
        else if (arg == "--output")     output_path = argv[++i];
        else if (arg == "--sigma")      sigma       = std::stod(argv[++i]);
        else if (arg == "--sigma-file") sigma_file  = argv[++i];
        else if (arg == "--trials")     n_trials    = std::stoi(argv[++i]);
        else if (arg == "--seed")       seed        = std::stoull(argv[++i]);
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        if (sigma < 0.0) {
            sigma = read_residual_sigma(sigma_file);
            std::cout << "Residual sigma: " << sigma << " s  (loaded from " << sigma_file << ")\n";
        } else {
            std::cout << "Residual sigma: " << sigma << " s  (from --sigma flag)\n";
        }

        auto entries = read_sim_input(input_path);
        std::cout << "Loaded " << entries.size() << " horse entries from " << input_path << "\n";

        // Group entries by race, preserving insertion order via map on race_id string.
        // Per-race loop is independent — straightforward to parallelize with OpenMP later.
        std::map<std::string, std::vector<HorseEntry>> by_race;
        for (auto& e : entries) {
            by_race[e.race_id].push_back(e);
        }
        std::cout << "Races: " << by_race.size()
                  << "  |  Trials per race: " << n_trials
                  << "  |  Seed: " << seed << "\n\n";

        std::mt19937_64 rng(seed);
        std::vector<SimResult> all_results;
        all_results.reserve(entries.size());

        for (auto& [race_id, horses] : by_race) {
            auto probs = simulate_race(horses, sigma, n_trials, rng);
            for (size_t i = 0; i < horses.size(); ++i) {
                all_results.push_back({
                    race_id,
                    horses[i].horse_id,
                    probs[i],
                    horses[i].won,
                    horses[i].win_odds
                });
            }
        }

        write_sim_output(output_path, all_results);
        std::cout << "Results written to " << output_path << "\n";

        auto m = compute_metrics(all_results);
        std::cout << "\n--- Metrics (" << m.n_races << " races, "
                  << m.n_horses << " horse-race rows) ---\n";
        std::cout << "Top-1 accuracy : " << m.top1_accuracy * 100.0 << "%\n";
        std::cout << "Brier score    : " << m.brier_score << "\n";
        std::cout << "Log loss       : " << m.log_loss << "\n";

        report_ev_bets(all_results);

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
