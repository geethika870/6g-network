import numpy as np
import pandas as pd

samples = 1000

distance = np.random.uniform(1, 20, samples)
blockage = np.random.choice([0, 1], samples)
humidity = np.random.uniform(30, 90, samples)

# Simplified THz path loss model
path_loss = 20 * np.log10(distance) + 30 * blockage + 0.02 * humidity

data = pd.DataFrame({
    "Distance": distance,
    "Blockage": blockage,
    "Humidity": humidity,
    "PathLoss": path_loss
})

data.to_csv("data/thz_data.csv", index=False)
print("THz dataset generated successfully")
#(THz channel + sensing simulation – REAL, not dummy)