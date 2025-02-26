import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mitbih Dataset
mitbih_train = pd.read_csv( "dataset/mitbih_train.csv",header = None)
mitbih_test = pd.read_csv("dataset/mitbih_test.csv",header = None)

mitbih_train_labels = mitbih_train.iloc[:,-1].replace({0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'})
plt.hist(mitbih_train_labels)
plt.title("Mitbih Class Distribution in training set")
plt.savefig("eda/hist_mitbih.png")

plt.figure(figsize=(8, 4))
plt.plot(mitbih_train.iloc[0, :-1], color="b", linewidth=1.5)
plt.xlabel("Time")
plt.ylabel("Signal")
plt.title("MIT-BIH Dataset First Sample")
plt.savefig("eda/mitbih_sample.png")

# PTBDB Dataset
ptbdb_abnormal = pd.read_csv( "dataset/ptbdb_abnormal.csv",header = None)
ptbdb_normal = pd.read_csv("dataset/ptbdb_normal.csv",header = None)
ptbdb = pd.concat([ptbdb_abnormal,ptbdb_normal], axis = 0, ignore_index=True)

ptbdb_labels = ptbdb.iloc[:,-1].replace({0: 'Normal', 1: 'Abnormal'})
plt.hist(ptbdb_labels)
plt.title("PTBDB Class Distribution")
plt.savefig("eda/hist_ptbdb.png")

plt.figure(figsize=(8, 4))
plt.plot(ptbdb.iloc[0,:-1],color="b", linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title("PTBDB Dataset First Sample")
plt.savefig("eda/ptbdb_sample.png")
