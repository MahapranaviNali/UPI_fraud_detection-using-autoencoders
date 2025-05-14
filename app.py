# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("🔍 UPI Fraud & Anomaly Detection")

st.markdown("Upload transaction data to detect anomalies using a trained autoencoder.")

# Load model
model = load_model("model/autoencoder_model.h5")
scaler = MinMaxScaler()

# Upload
file = st.file_uploader("Upload CSV", type="csv")

if file:
    df = pd.read_csv(file)
    df_num = df.select_dtypes(include=[np.number]).fillna(0)

    # Normalize
    data_scaled = scaler.fit_transform(df_num)

    # Predict
    reconstructions = model.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)
    anomalies = (mse > threshold).astype(int)

    # Append predictions
    df['Reconstruction_Error'] = mse
    df['Anomaly'] = anomalies

    st.subheader("Anomaly Summary")
    st.write(f"🔺 Detected {df['Anomaly'].sum()} anomalous transactions out of {len(df)}")

    st.dataframe(df[df['Anomaly'] == 1])

    csv = df.to_csv(index=False)
    st.download_button("📥 Download Results", csv, "anomaly_results.csv", "text/csv")