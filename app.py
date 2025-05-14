# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = load_model("autoencoder_model.h5", compile=False)

def predict_transaction(sender_upi, receiver_upi, amount):
    # Factorize input like in training (manual encoding)
    sender_encoded = abs(hash(sender_upi)) % (10**6)
    receiver_encoded = abs(hash(receiver_upi)) % (10**6)
    
    X = np.array([[sender_encoded, receiver_encoded, amount]])
    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    threshold = 0.95 * np.max(mse)
    prediction = "FRAUD" if mse[0] > threshold else "LEGIT"
    return prediction, mse[0], threshold

st.set_page_config(page_title="UPI Fraud Detection", layout="centered")
st.title("🔐 UPI Fraud Detection (Manual Entry)")

# Manual form
with st.form("manual_input"):
    st.subheader("Enter transaction details:")
    sender = st.text_input("Sender UPI ID", "user1@upi")
    receiver = st.text_input("Receiver UPI ID", "merchant@upi")
    amount = st.number_input("Amount (INR)", min_value=0.01, step=0.01)
    submitted = st.form_submit_button("Check Transaction")

    if submitted:
        prediction, error, threshold = predict_transaction(sender, receiver, amount)
        st.markdown(f"### 🧾 Prediction: **{prediction}**")
        st.markdown(f"- Reconstruction Error: `{error:.6f}`")
        st.markdown(f"- Threshold: `{threshold:.6f}`")

        fig, ax = plt.subplots()
        sns.barplot(x=["Reconstruction Error", "Threshold"], y=[error, threshold], ax=ax)
        ax.set_title("Error vs Threshold")
        st.pyplot(fig)
