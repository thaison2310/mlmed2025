import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
import joblib

from dataset import TorchDataset

# Load and combine dataset
ptbdb_normal = pd.read_csv("dataset/ptbdb_normal.csv",header = None)
ptbdb_abnormal = pd.read_csv("dataset/ptbdb_abnormal.csv",header = None)
ptbdb = pd.concat([ptbdb_abnormal,ptbdb_normal], axis = 0, ignore_index=True)

X = ptbdb.iloc[:, :-1].values
y = ptbdb.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test dataset as csv for later evaluation
test_data = np.column_stack((X_test, y_test))
np.savetxt("dataset/ptbdb_test.csv", test_data, delimiter=",", fmt="%.6f")

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
joblib.dump(scaler,"scaler/ptbdb_scaler.pkl")

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)   

y_train = torch.tensor(y_train, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TorchDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# Import trained model from train_mitbih.py
model = torch.load("model/model_full_mitbih.pth")

# Training
model.traln()
criterion = torch.nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30 

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels) 
        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Save model for later evaluation
torch.save(model, "model/model_full_ptbdb.pth")