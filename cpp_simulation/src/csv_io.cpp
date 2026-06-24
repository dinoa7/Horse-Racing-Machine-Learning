#include "csv_io.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

static std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::vector<HorseEntry> read_sim_input(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    // Consume header: race_id,horse_id,predicted_finish_time,actual_finish_time,won,win_odds
    std::string header;
    std::getline(file, header);

    std::vector<HorseEntry> entries;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto f = split_line(line);
        if (f.size() < 6) continue;

        HorseEntry e;
        e.race_id               = f[0];
        e.horse_id              = f[1];
        e.predicted_finish_time = std::stod(f[2]);
        e.actual_finish_time    = std::stod(f[3]);
        e.won                   = std::stoi(f[4]);
        e.win_odds              = std::stod(f[5]);
        entries.push_back(std::move(e));
    }
    return entries;
}

double read_residual_sigma(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open sigma file: " + path);
    }
    double sigma;
    if (!(file >> sigma)) {
        throw std::runtime_error("Failed to parse sigma value from: " + path);
    }
    return sigma;
}

void write_sim_output(const std::string& path, const std::vector<SimResult>& results) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file for writing: " + path);
    }
    file << "race_id,horse_id,win_probability,won,win_odds\n";
    file << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        file << r.race_id << ","
             << r.horse_id << ","
             << r.win_probability << ","
             << r.won << ","
             << r.win_odds << "\n";
    }
}
