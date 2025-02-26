import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import joblib
from model import CNN
from dataset import TorchDataset

# Load dataset
mitbih_train = pd.read_csv( "dataset/mitbih_train.csv",header = None)

X_train = mitbih_train.iloc[:, :-1].values  
y_train = mitbih_train.iloc[:, -1].values  

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler/mitbih_scaler.pkl")  

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)   

y_train = torch.tensor(y_train, dtype=torch.long)

batch_size = 32
train_loader = DataLoader(TorchDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

num_classes = len(torch.unique(y_train))
input_length = X_train.shape[2]  
device = torch.device("cpu")

# Import model
model = CNN(num_classes, input_length).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# Save model
torch.save(model, "model/model_full_mitbih.pth")