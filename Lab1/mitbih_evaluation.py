import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dataset import TorchDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

mitbih_test = pd.read_csv("dataset/mitbih_test.csv", header=None)
X_test = mitbih_test.iloc[:, :-1].values  
y_test = mitbih_test.iloc[:, -1].values
scaler = StandardScaler()
scaler = joblib.load("scaler/mitbih_scaler.pkl")  

X_test = scaler.transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) 
y_test = torch.tensor(y_test, dtype=torch.long)
batch_size = 32
test_loader = DataLoader(TorchDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

num_classes = len(torch.unique(y_test))

model = torch.load("model/model_full_mitbih.pth")
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()