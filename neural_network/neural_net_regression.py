"""
Python Version: 3.13.1

"""
import pandas as pd
import numpy as np
#Pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch import optim
#SkLearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
#Matplot
import matplotlib.pyplot as plt
#other
import sys
sys.path.insert(1,"../linear_regression")
from linear_regression_QOL import plot_predicted_vs_actual


##************************Data Grab************************
dataset_csv = pd.read_csv("cleaned_race_data.csv")

# Removed 'won' (data leakage — directly encodes finish order) and 'race_id' is
# kept only for grouping, not as a model feature.
filt_dataset_csv = dataset_csv[["race_id","won","finish_time","draw","horse_rating",
                           "declared_weight","horse_age","actual_weight","win_odds","distance","race_class"]]

#************************Data Setup************************
data_org = filt_dataset_csv.copy(True)

##************************one-hot encoding of race_class************************
data_org = pd.get_dummies(data_org,columns=["race_class"],drop_first=False,dtype=int)
target_org = data_org["finish_time"]
data_org = data_org.drop(columns="finish_time")

# Keep race_id and won for post-prediction evaluation only — not model features
race_id_col = data_org["race_id"]
won_col = data_org["won"]
data_org = data_org.drop(columns=["race_id", "won"])

#************************Data Split************************
data_org, x_test, target_org, y_test, race_id_train, race_id_test, won_train, won_test = train_test_split(
    data_org, target_org, race_id_col, won_col, test_size=0.1, random_state=42)
x_metric_dataset = x_test.copy(True)
y_tst_metric = y_test.copy(True).to_numpy()

#************************ MODEL DEFINITION************************
class NeuralNet(nn.Module):
    def __init__(self,features):
        super(NeuralNet,self).__init__()
        self.mlayers = nn.Sequential(
            nn.Linear(features,64),
            nn.LeakyReLU(),
            nn.Linear(64,32),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.LeakyReLU(),
            nn.Linear(16,8),
            nn.Dropout(0.2),
            nn.Linear(8,1)
        )
    def forward(self, x):
        x = self.mlayers(x)
        return x.view(-1)

#************************Cuda Setup************************
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#************************Hyper Parameters************************
LOSS_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 512

#************************Loss Function************************
loss_fn = nn.MSELoss()

#************************5-fold cross-validation************************
skf = KFold(n_splits=5)

final_scaler = StandardScaler()
final_scaler.fit(data_org.values)

#************************Model Training and Validation************************
hist_trn_mse = []
hist_val_mse = []

print(EPOCHS, " epochs:")
for i, (train_idx, test_idx) in enumerate(skf.split(data_org, target_org)):

    print("  {}-fold:".format(i+1))

    model = NeuralNet(data_org.shape[1])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LOSS_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_data, y_train = data_org.iloc[train_idx], target_org.iloc[train_idx]
    val_data, y_val = data_org.iloc[test_idx], target_org.iloc[test_idx]

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data.values)
    val_data = scaler.transform(val_data.values)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    val_data = torch.tensor(val_data, dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    train_dataset = TensorDataset(train_data, y_train)
    val_dataset = TensorDataset(val_data, y_val)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    fold_t_loss = []
    fold_v_loss = []

    #************************TRAIN + VALIDATE PER EPOCH************************
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            train_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        fold_t_loss.append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                y_pred = model(data)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item() * data.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        fold_v_loss.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)

    mse_avg = val_loss / len(val_loader.dataset)
    print(f"   val mse: {mse_avg:.4f}")

    hist_trn_mse.append(fold_t_loss)
    hist_val_mse.append(fold_v_loss)

#************************LOSS Curve************************
plt.figure()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot(np.mean(hist_trn_mse, axis=0), label='train_loss')
plt.plot(np.mean(hist_val_mse, axis=0), label='val_loss')
plt.title("Learning Curve")
plt.legend()
plt.show()

#************************Final model trained on ALL training data************************
final_model = NeuralNet(data_org.shape[1])
final_model = final_model.to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=LOSS_RATE)
final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, patience=10, factor=0.5)

all_train_data = torch.tensor(final_scaler.transform(data_org.values), dtype=torch.float32)
all_train_target = torch.tensor(target_org.to_numpy(), dtype=torch.float32)
all_train_dataset = TensorDataset(all_train_data, all_train_target)
all_train_loader = DataLoader(all_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())

print("\nTraining final model on full training set...")
for epoch in range(EPOCHS):
    final_model.train()
    epoch_loss = 0.0
    for data, target in all_train_loader:
        data, target = data.to(device), target.to(device)
        final_optimizer.zero_grad()
        output = final_model(data)
        loss = loss_fn(output, target)
        epoch_loss += loss.item() * data.size(0)
        loss.backward()
        final_optimizer.step()
    final_scheduler.step(epoch_loss / len(all_train_loader.dataset))

#************************Model Test Phase************************
x_test_scaled = final_scaler.transform(x_test.values)

x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

y_preds = []
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available())

final_model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        tst_pred = final_model(data)
        y_preds.append(tst_pred.item())

#******************************METRICS******************
x_metric_dataset = x_metric_dataset.reset_index(drop=True)
x_metric_dataset["race_id"] = race_id_test.reset_index(drop=True)
x_metric_dataset["won"] = won_test.reset_index(drop=True)
x_metric_dataset["Finish_Time"] = y_preds

#************************Accuracy************************
def predict_winner(group):
    predicted_winner_index = group['Finish_Time'].idxmin()
    return group.loc[predicted_winner_index, 'won'] == 1

results = x_metric_dataset.groupby('race_id').apply(predict_winner, include_groups=False)
accuracy = results.mean()
print(f"Prediction Accuracy: {accuracy:.4f} ({results.sum()} correct out of {len(results)} races)")

#************************More Plot************************
plot_predicted_vs_actual(y_tst_metric, y_preds)
