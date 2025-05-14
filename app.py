# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from tensorflow.keras.models import load_model

# Load the scaler and model
scaler = joblib.load("scaler.pkl")
model = load_model("autoencoder_model.h5")

def detect_fraud(data):
    X = data[['Sender UPI ID', 'Receiver UPI ID', 'Amount (INR)']]
    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    threshold = 0.95 * np.max(mse)  # Simplified
    y_pred = ["FRAUD" if e > threshold else "LEGIT" for e in mse]
    data['Prediction'] = y_pred
    data['Reconstruction Error'] = mse
    return data, threshold

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🔒 UPI Fraud Detection using Autoencoder")

uploaded_file = st.file_uploader("Upload a CSV file with transactions", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop(columns=["Transaction ID", "Timestamp", "Sender Name", "Receiver Name"])

    # Encode using same logic as training
    df['Sender UPI ID'] = pd.factorize(df['Sender UPI ID'])[0]
    df['Receiver UPI ID'] = pd.factorize(df['Receiver UPI ID'])[0]

    result, threshold = detect_fraud(df)

    st.subheader("🔢 Prediction Results")
    st.write(result[['Sender UPI ID', 'Receiver UPI ID', 'Amount (INR)', 'Prediction']])

    fraud_count = result['Prediction'].value_counts()
    st.bar_chart(fraud_count)

    st.subheader("📊 Reconstruction Error Distribution")
    fig, ax = plt.subplots()
    sns.histplot(result['Reconstruction Error'], bins=50, kde=True, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold')
    ax.set_title("Reconstruction Error vs Frequency")
    ax.legend()
    st.pyplot(fig)

    st.download_button("Download Results as CSV", result.to_csv(index=False), "predictions.csv", "text/csv")
