#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <random>

#include "csv_io.hpp"
#include "metrics.hpp"
#include "simulation.hpp"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static HorseEntry make_entry(const std::string& race, const std::string& horse,
                              double pred, double actual, int won, double odds) {
    return {race, horse, pred, actual, won, odds};
}

static SimResult make_result(const std::string& race, const std::string& horse,
                              double prob, int won, double odds) {
    return {race, horse, prob, won, odds};
}

// ---------------------------------------------------------------------------
// simulation tests
// ---------------------------------------------------------------------------
TEST_CASE("Single horse in a race always wins", "[simulation]") {
    std::mt19937_64 rng(0);
    auto probs = simulate_race({make_entry("r1", "h1", 80.0, 80.5, 1, 1.5)},
                               0.5, 1000, rng);
    REQUIRE(probs.size() == 1);
    REQUIRE(probs[0] == Catch::Approx(1.0));
}

TEST_CASE("Clearly faster horse wins nearly every trial", "[simulation]") {
    // 30 second gap with sigma = 0.5 => faster horse wins ~100% of trials
    std::mt19937_64 rng(1);
    std::vector<HorseEntry> field = {
        make_entry("r1", "fast", 60.0, 60.1, 1, 2.0),
        make_entry("r1", "slow", 90.0, 90.5, 0, 5.0),
    };
    auto probs = simulate_race(field, 0.5, 10'000, rng);
    REQUIRE(probs[0] > 0.999);
    REQUIRE(probs[1] < 0.001);
}

TEST_CASE("Two identical horses split win probability near 50/50", "[simulation]") {
    std::mt19937_64 rng(2);
    std::vector<HorseEntry> field = {
        make_entry("r1", "h1", 80.0, 80.0, 1, 2.0),
        make_entry("r1", "h2", 80.0, 80.5, 0, 2.0),
    };
    auto probs = simulate_race(field, 1.0, 100'000, rng);
    REQUIRE(probs[0] == Catch::Approx(0.5).epsilon(0.02));
    REQUIRE(probs[1] == Catch::Approx(0.5).epsilon(0.02));
}

TEST_CASE("Win probabilities for a full field sum to exactly 1.0", "[simulation]") {
    std::mt19937_64 rng(3);
    std::vector<HorseEntry> field;
    for (int i = 0; i < 12; ++i) {
        field.push_back(make_entry("r1", "h" + std::to_string(i),
                                   80.0 + i * 0.3, 80.0 + i * 0.3,
                                   i == 0 ? 1 : 0, 2.0 + i));
    }
    auto probs = simulate_race(field, 0.5, 10'000, rng);
    double total = 0.0;
    for (double p : probs) total += p;
    // Exact by construction: win_counts sum to n_trials.
    REQUIRE(total == Catch::Approx(1.0).epsilon(1e-9));
}

// ---------------------------------------------------------------------------
// metrics tests
// ---------------------------------------------------------------------------
TEST_CASE("Top-1 accuracy is 1.0 when model assigns highest prob to actual winner",
          "[metrics]") {
    std::vector<SimResult> results = {
        make_result("r1", "winner", 0.85, 1, 2.0),
        make_result("r1", "loser1", 0.10, 0, 5.0),
        make_result("r1", "loser2", 0.05, 0, 9.0),
    };
    auto m = compute_metrics(results);
    REQUIRE(m.top1_accuracy == Catch::Approx(1.0));
    REQUIRE(m.n_races == 1);
    REQUIRE(m.n_horses == 3);
}

TEST_CASE("Top-1 accuracy is 0.0 when model consistently picks the wrong horse",
          "[metrics]") {
    std::vector<SimResult> results = {
        make_result("r1", "winner", 0.10, 1, 2.0),
        make_result("r1", "loser",  0.90, 0, 1.2),
    };
    auto m = compute_metrics(results);
    REQUIRE(m.top1_accuracy == Catch::Approx(0.0));
}

TEST_CASE("Brier score is 0 for perfect probability assignments", "[metrics]") {
    std::vector<SimResult> results = {
        make_result("r1", "winner", 1.0, 1, 2.0),
        make_result("r1", "loser",  0.0, 0, 5.0),
    };
    auto m = compute_metrics(results);
    REQUIRE(m.brier_score == Catch::Approx(0.0).margin(1e-9));
}
