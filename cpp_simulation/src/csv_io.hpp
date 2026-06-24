#pragma once
#include <string>
#include <vector>

struct HorseEntry {
    std::string race_id;
    std::string horse_id;
    double predicted_finish_time;
    double actual_finish_time;
    int won;
    double win_odds;
};

struct SimResult {
    std::string race_id;
    std::string horse_id;
    double win_probability;
    int won;
    double win_odds;
};

std::vector<HorseEntry> read_sim_input(const std::string& path);
double read_residual_sigma(const std::string& path);
void write_sim_output(const std::string& path, const std::vector<SimResult>& results);
