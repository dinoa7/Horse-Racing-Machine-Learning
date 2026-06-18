import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def evaluate_predictions(y_true, y_pred):
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mae = np.mean(np.abs(y_true - y_pred))
  print(f"RMSE: {rmse:.4f}")
  print(f"MAE:  {mae:.4f}")
  return rmse, mae

def plot_predicted_vs_actual(y_true, y_pred):
  plt.figure(figsize=(8, 6))
  plt.scatter(y_true, y_pred, alpha=0.6)
  plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
  plt.xlabel("Actual Finish Time")
  plt.ylabel("Predicted Finish Time")
  plt.title("Predicted vs Actual Finish Time")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

def main():
  # --- Step 1: Load filtered test and training data ---
  train_df = pd.read_csv('filtered_train_data.csv')
  test_df = pd.read_csv('filtered_test_data.csv')
    
  # Save important meta columns before processing
  test_meta = test_df[['race_id', 'won']]
    
  # --- Step 2: One-hot encode 'race_class' using the same categories ---
  race_class_categories = sorted(train_df['race_class'].unique())
  train_df = pd.get_dummies(train_df, columns=['race_class'], drop_first=False)
  test_df = pd.get_dummies(test_df, columns=['race_class'], drop_first=False)
    
  # --- Step 3: Manually define feature columns ---
  feature_cols = ['declared_weight', 'actual_weight', 'win_odds', 'distance'] + \
                   [f'race_class_{c}' for c in race_class_categories]
    
  # --- Step 4: Prepare feature matrices ---
  X_train = train_df[feature_cols]
  y_train = train_df['finish_time']
    
  X_test = test_df.reindex(columns=feature_cols, fill_value=0)
  y_test = test_df['finish_time']
    
  # --- Step 5: Standardize features ---
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
    
  # --- Step 6: Train model ---
  model = LinearRegression()
  model.fit(X_train_scaled, y_train)
    
  # --- Step 7: Predict on test set ---
  predicted = model.predict(X_test_scaled)
  test_df['predicted_finish_time'] = predicted
    
  # Recover race_id and won
  test_df['race_id'] = test_meta['race_id'].values
  test_df['won'] = test_meta['won'].values
    
  # --- Step 8: Group by race_id and evaluate ---
  def predict_winner(group):
      predicted_winner_index = group['predicted_finish_time'].idxmin()
      return group.loc[predicted_winner_index, 'won'] == 1
    
  # Bug fix #7: include_groups=False avoids DeprecationWarning in pandas 2.x
  results = test_df.groupby('race_id').apply(predict_winner, include_groups=False)
    
  # --- Step 9: Calculate accuracy ---
  accuracy = results.mean()
  print(f"Prediction Accuracy: {accuracy:.4f} ({results.sum()} correct out of {len(results)} races)")

  #rmse
  evaluate_predictions(y_test, predicted)
  #plot for doc
  plot_predicted_vs_actual(y_test, predicted)

if __name__ == "__main__":
  main()
