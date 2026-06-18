## Dataset Information

**general_dataset**  
`cleaned_race_data.csv` — Contains all major features, combining data from both `runs.csv` and `races.csv`.

**train_data**  
`train_data.csv` — 70% of the data used for training.

**val_data**  
`val_data.csv` — 15% of the data used for validation.

**test_data**  
`test_data.csv` — 15% of the data used for testing.

---

### 🧹 Columns Discarded During Cleaning

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
