import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data/thz_data.csv")

X = data[["Distance", "Blockage", "Humidity"]].values
y = data["PathLoss"].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

time_steps = 5
X_seq, y_seq = [], []

for i in range(len(X) - time_steps):
    X_seq.append(X[i:i+time_steps])
    y_seq.append(y[i+time_steps])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print("Prepared sequence shape:", X_seq.shape)
