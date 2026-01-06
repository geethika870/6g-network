import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.beamforming import select_beamforming

model = load_model("model/channel_predictor.h5")

st.set_page_config(page_title="AI-Enabled THz ISAC System", layout="wide")

st.title("AI-Enabled THz Integrated Communication and Sensing System")
st.write("CNN-LSTM based Channel Prediction & Adaptive Beamforming for 6G Networks")

tabs = st.tabs(["Manual Input", "Bulk CSV Upload", "System Info"])

# ---------------- MANUAL INPUT ----------------
with tabs[0]:
    st.header("Manual Time-Series Input")
    time_steps = 5
    sequence = []

    for i in range(time_steps):
        st.subheader(f"Time Step {i+1}")
        col1, col2, col3 = st.columns(3)

        with col1:
            d = st.number_input("Distance (m)", 0.1, 50.0, key=f"d{i}")
        with col2:
            b = st.selectbox("Blockage", [0, 1],
                             format_func=lambda x: "Yes" if x else "No",
                             key=f"b{i}")
        with col3:
            h = st.number_input("Humidity (%)", 0.0, 100.0, key=f"h{i}")

        sequence.append([d, b, h])

    if st.button("Predict Channel"):
        seq = np.array(sequence).reshape(1, time_steps, 3)
        loss = model.predict(seq)[0][0]
        beam = select_beamforming(loss)

        st.success("Prediction Completed")
        st.metric("Predicted Path Loss (dB)", f"{loss:.2f}")
        st.info(f"Beamforming Strategy: **{beam}**")

# ---------------- BULK CSV ----------------
with tabs[1]:
    st.header("Bulk Prediction using CSV")
    st.write("CSV must contain: Distance, Blockage, Humidity")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        data = pd.read_csv(file)
        st.dataframe(data.head())

        X = data[["Distance", "Blockage", "Humidity"]].values
        time_steps = 5
        sequences = []

        for i in range(len(X) - time_steps):
            sequences.append(X[i:i+time_steps])

        sequences = np.array(sequences)
        preds = model.predict(sequences).flatten()

        result = data.iloc[time_steps:].copy()
        result["Predicted_PathLoss"] = preds
        result["Beamforming"] = result["Predicted_PathLoss"].apply(select_beamforming)

        st.subheader("Results")
        st.dataframe(result.head())

        fig, ax = plt.subplots()
        ax.plot(result["Predicted_PathLoss"])
        ax.set_title("Predicted Path Loss Trend")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Path Loss (dB)")
        st.pyplot(fig)

# ---------------- INFO ----------------
with tabs[2]:
    st.header("About the System")
    st.write("""
    This system uses a CNN-LSTM deep learning model.
    CNN extracts spatial features from THz sensing data,
    while LSTM captures temporal channel variations.
    The predicted channel condition is used to select
    an adaptive beamforming strategy.
    """)
