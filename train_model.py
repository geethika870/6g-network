import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

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

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

model = Sequential([
    Conv1D(filters=64, kernel_size=2, activation="relu",
           input_shape=(time_steps, 3)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=30, batch_size=32)

model.save("model/channel_predictor.h5")
print("CNN-LSTM model trained and saved")
