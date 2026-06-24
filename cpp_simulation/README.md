# Monte Carlo Win-Probability Simulator

Converts the PyTorch model's deterministic finish-time predictions into per-horse
**win probabilities** via Gaussian Monte Carlo sampling.

## How it works

For each race the simulator:

1. Takes each horse's `predicted_finish_time` as the mean of a Gaussian.
2. Draws N independent noise samples (std dev = `sigma`, the model's residual standard deviation on the test set).
3. In each trial the horse with the lowest sampled time wins.
4. Win probability = (trials won) / N.

This converts a single-point prediction into a probability distribution over outcomes,
and allows meaningful metrics (Brier score, log loss) and expected-value bet detection.

## Prerequisites

- CMake 3.16+
- A C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
- Internet access on **first** build only — Catch2 is downloaded via `FetchContent`

## Step 1 — generate the input data

Run the Python model from the `neural_network/` directory:

```bash
cd neural_network
python neural_net_regression.py
```

This writes two files into `general_dataset/`:
- `sim_input.csv` — one row per horse in the test set
- `residual_sigma.txt` — a single float: the std dev of (predicted − actual) on the test set

## Step 2 — build

```bash
cd cpp_simulation
mkdir build && cd build
cmake ..
cmake --build .
```

On Windows with Visual Studio you may need:

```bash
cmake --build . --config Release
```

## Step 3 — run

From inside the `build/` directory:

```bash
# Linux / macOS
./monte_carlo

# Windows
monte_carlo.exe
```

With options:

```bash
./monte_carlo --trials 50000 --seed 99 --sigma 1.2
```

### CLI reference

| Option | Default | Description |
|---|---|---|
| `--input PATH` | `../general_dataset/sim_input.csv` | Input CSV from Python |
| `--output PATH` | `../general_dataset/sim_output.csv` | Output CSV |
| `--sigma FLOAT` | *(read from file)* | Residual std dev in seconds; overrides `--sigma-file` |
| `--sigma-file PATH` | `../general_dataset/residual_sigma.txt` | File produced by the Python script |
| `--trials N` | `10000` | Monte Carlo trials per race |
| `--seed N` | `42` | RNG seed — same seed + same input = identical output |
| `--help` | | Print usage |

## Output

`sim_output.csv` columns:

| Column | Description |
|---|---|
| `race_id` | Race identifier |
| `horse_id` | Horse identifier |
| `win_probability` | Fraction of trials this horse had the lowest sampled time |
| `won` | Ground truth (1 = actual winner) |
| `win_odds` | Decimal odds from the dataset |

Metrics printed to stdout after the run:

- **Top-1 accuracy** — fraction of races where the horse with the highest win probability actually won
- **Brier score** — mean squared error of win probabilities vs. binary outcomes (lower is better; 0 = perfect)
- **Log loss** — cross-entropy of probabilities vs. outcomes (lower is better)
- **EV bets** — horses where `win_probability > 1/win_odds` (normalized per race to remove bookmaker margin)

## Running tests

```bash
# From build/
ctest --output-on-failure

# Or run the test binary directly for verbose output:
./run_tests -v          # Linux/macOS
run_tests.exe -v        # Windows
```

Tests cover:
- Single-horse race → win probability = 1.0
- Clearly faster horse (30 s gap, σ = 0.5 s) → win probability > 99.9%
- Two identical horses → win probabilities ≈ 0.5 each (±2%)
- Full 12-horse field → probabilities sum to exactly 1.0
- Top-1 accuracy = 1.0 when model correctly ranks the winner highest
- Top-1 accuracy = 0.0 when model always picks the wrong horse
- Brier score = 0 for perfect probability assignments

## Noise model assumptions

- Residuals are Gaussian and i.i.d. across horses and races.
- A single global `sigma` is used. The function signatures accept sigma per call, so
  per-distance or per-race-class sigma can be plugged in without changing the core loop.
- Horse finish times are sampled **independently** within a trial — shared within-race
  effects (track bias, pace) are not modelled.

## Parallelization

The per-race loop in `main.cpp` processes races sequentially. Because each race's
`simulate_race` call is independent, adding `#pragma omp parallel for` (or `std::async`
tasks) around that loop is the only change needed for multithreading.
