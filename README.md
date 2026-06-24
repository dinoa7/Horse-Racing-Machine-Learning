# Horse Racing Machine Learning

Predicts horse race finish times using a PyTorch neural network, then converts point predictions into win probabilities via a C++ Monte Carlo simulator.

## Project Structure

```
neural_network/       — PyTorch model (training, evaluation, prediction)
linear_regression/    — Baseline linear regression notebooks
cpp_simulation/       — Monte Carlo win-probability simulator
general_dataset/      — Cleaned CSV data and model output files
```

## Dataset

**general_dataset/**
`cleaned_race_data.csv` — All major features, combining data from `runs.csv` and `races.csv`.

**train_data/**
`train_data.csv` — 70% of the data used for training.

**val_data/**
`val_data.csv` — 15% of the data used for validation.

**test_data/**
`test_data.csv` — 15% of the data used for testing.

### Columns discarded during cleaning

```python
columns_discarded = [
    'time1_y', 'time2_y', 'time3_y', 'time4_y', 'time5_y', 'time6_y', 'time7',
    'place_combination1', 'place_combination2', 'place_combination3', 'place_combination4',
    'place_dividend1', 'place_dividend2', 'place_dividend3', 'place_dividend4',
    'win_combination1', 'win_combination2', 'win_dividend1', 'win_dividend2',
    'position_sec1', 'position_sec2', 'position_sec3', 'position_sec4', 'position_sec5', 'position_sec6',
    'behind_sec1', 'behind_sec2', 'behind_sec3', 'behind_sec4', 'behind_sec5', 'behind_sec6',
    'horse_no',
    'race_no', 'config',
    'sec_time1', 'sec_time2', 'sec_time3',
    'sec_time4', 'sec_time5', 'sec_time6', 'sec_time7',
    'prize',
    'time1_x', 'time2_x', 'time3_x', 'time4_x', 'time5_x', 'time6_x',
    'place_odds'
]
```

## Typical Workflow

1. Train the Python model to generate simulator input data
2. Build and run the C++ simulator to get win probabilities

---

## Step 1 — Install Python dependencies

```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm
```

## Step 2 — Train the model

```bash
cd neural_network
python neural_net_regression.py
```

Writes two files to `general_dataset/`:
- `sim_input.csv` — one row per horse in the test set
- `residual_sigma.txt` — residual std dev (seconds) from the test set

The notebooks (`neural_net_regression.ipynb`, `linear_regression/linear_regression.ipynb`) can be run interactively in Jupyter.

## Step 3 — Build the C++ simulator

Requires CMake 3.16+ and a C++17 compiler. Catch2 is downloaded automatically on first build.

```powershell
cd cpp_simulation
mkdir build; cd build
cmake ..
cmake --build . --config Release
```

## Step 4 — Run the simulator

```powershell
# Windows (from cpp_simulation/build/)
./monte_carlo.exe

# With options
./monte_carlo.exe --trials 50000 --seed 99 --sigma 1.2
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--input PATH` | `../../general_dataset/sim_input.csv` | Input CSV from Python |
| `--output PATH` | `../../general_dataset/sim_output.csv` | Output CSV |
| `--sigma FLOAT` | *(read from file)* | Residual std dev in seconds |
| `--sigma-file PATH` | `../../general_dataset/residual_sigma.txt` | File produced by Python script |
| `--trials N` | `10000` | Monte Carlo trials per race |
| `--seed N` | `42` | RNG seed |

## Running tests

```powershell
# From cpp_simulation/build/
ctest --output-on-failure

# Verbose output
./run_tests.exe -v
```
