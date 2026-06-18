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
#from filted dataset in regression folder
filt_dataset_csv = dataset_csv[["race_id","won","finish_time","draw","horse_rating",
                           "declared_weight","horse_age","actual_weight","win_odds","distance","race_class"]]

#************************Data Setup************************
data_org = filt_dataset_csv.copy(True) #Preserve original copy

##************************hotshot encoding of race_class************************
data_org = pd.get_dummies(data_org,columns=["race_class"],drop_first=False,dtype=int)
target_org = data_org["finish_time"]
data_org = data_org.drop(columns="finish_time")

#************************Data Split************************
#95% train - test split
#Split data for k-fold and test for metric
data_org, x_test, target_org, y_test = train_test_split(data_org,target_org, test_size=0.1, stratify=data_org["won"])
x_metric_dataset = x_test.copy(True)
y_tst_metric = y_test.copy(True).to_numpy()

#************************ MODEL DEFINITION************************
"""
Model Definition:
still need to be tuned

Final layer should be linear with
an output of 1 for regression
"""
class NeuralNet(nn.Module):
    def __init__(self,features):
        super(NeuralNet,self).__init__()
        self.mlayers = nn.Sequential(
            nn.Linear(features,19), #19
            nn.ReLU(),
            nn.Linear(19,16),
            nn.Dropout(0.6),
            nn.Linear(16,13),
            nn.LeakyReLU(),
            nn.Linear(13,10),
            nn.Tanh(),
            nn.Linear(10,7),
            nn.ReLU(),
            nn.Linear(7,1)   
        )       
    def forward(self, x):
        x = self.mlayers(x)
        return x.view(-1) #turns to 1d array

#************************Cuda and Model Setup************************

#Select cuda if availiable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#Model definition and cuda selection
model = NeuralNet(data_org.shape[1])
model = model.to(device) #transfer model to device memory

#************************Hyper Paramaters************************

LOSS_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 512

#************************optimizer and Loss************************
optimizer = optim.Adam(model.parameters(), lr=LOSS_RATE) 
#Loss function
loss_fn = nn.MSELoss() #regreesion

#************************5-fold cross-validation#************************

skf = KFold(n_splits=5)

#************************Model Training and Validation************************

hist_trn_mse =[]
hist_val_mse = []

#Split data into 5 folds
print(EPOCHS, " epoches:")
for i, (train_idx, test_idx) in enumerate(skf.split(data_org,target_org)):

    #Training & test subset
    print("  {}-fold:".format(i+1))

    train_data, y_train = data_org.iloc[train_idx], target_org.iloc[train_idx]
    val_data, y_val = data_org.iloc[test_idx], target_org.iloc[test_idx]
    
    #Standardize
    scaler = StandardScaler()  
    train_data = scaler.fit_transform(train_data.values)
    val_data = scaler.transform(val_data.values)
    
    #Convert to tensor data
    train_data = torch.tensor(train_data,dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(),dtype=torch.float32)
    
    val_data = torch.tensor(val_data,dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(),dtype=torch.float32)
    
    #Convert to tensor dataset for dataloader
    train_dataset = TensorDataset(train_data,y_train)
    val_dataset = TensorDataset(val_data,y_val)
    
    #DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True)
    
    #Fold Loss Traking
    fold_t_loss = []
    fold_v_loss = []
    
    #************************TRAIN************************
    model.train()
    for epoch in range(EPOCHS):
        train_loss = 0.0
        
        for data,target in train_loader:
            data,target = data.to(device),target.to(device)#Transfer data to device memory
            
            output = model(data)
            optimizer.zero_grad()
            
            loss = loss_fn(output,target)
            train_loss += loss.item() * data.size(0)
            
            loss.backward()
            
            optimizer.step()
            
        fold_t_loss.append(train_loss/len(train_loader.dataset))
            
    #************************Validation************************
        model.eval()
        val_loss = 0.0
        total = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data,target = data.to(device),target.to(device) #transfer data to device memory
                
                y_pred = model(data)
                #
                loss = loss_fn(y_pred,target) #calculate loss for plotting
                val_loss += loss.item() * data.size(0)
                
        #store validation loss for ploting
        fold_v_loss.append(val_loss/len(val_loader.dataset))
        
    mse_avg = val_loss / len(val_loader.dataset)     
    print(f"   test mse: {mse_avg:.4f}")

#Store fold loss info for plot 
hist_trn_mse.append(fold_t_loss)
hist_val_mse.append(fold_v_loss)
#************************************************************************

#************************LOSS Curve************************
plt.figure()
#plt.xlim(0,EPOCHES)
plt.xlabel("epoch")
plt.ylabel("Loss")
#plt.ylim(0,3.0)
plt.plot(np.mean(hist_trn_mse,axis=0), label='train_loss')
plt.plot(np.mean(hist_val_mse,axis=0),label='val_loss')
plt.title("Learning Curve")
plt.legend()

#************************Model Test Phase************************

#Prepare test dataset for evaluation
x_test = scaler.transform(x_test.values)

x_test = torch.tensor(x_test,dtype=torch.float32)
y_test = torch.tensor(y_test.to_numpy(),dtype=torch.float32)

y_preds = [] #Store predictions for metrics
#Convert to Tensor
test_dataset = TensorDataset(x_test,y_test)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,pin_memory=True)

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data,target = data.to(device),target.to(device)
        tst_pred = model(data) #prediction
        y_preds.append(tst_pred.item())
        
#******************************METRICS******************
       
#Store predicted finish time into x dataset
x_metric_dataset.reset_index() 
x_metric_dataset["Finish_Time"] = y_preds

#************************Accuracy************************
#Checks if loswest finish_time matches won(1) label based on race_id
def predict_winner(group):
    predicted_winner_index = group['Finish_Time'].idxmin()
    return group.loc[predicted_winner_index, 'won'] == 1

results = x_metric_dataset.groupby('race_id').apply(predict_winner)
accuracy = results.mean()
print(f"Prediction Accuracy: {accuracy:.4f} ({results.sum()} correct out of {len(results)} races)")

#************************More Plot************************
plot_predicted_vs_actual(y_tst_metric,y_preds)


