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
model = load_model("autoencoder_model.h5", compile=False)

def detect_fraud(data):
    X = data[['Sender UPI ID', 'Receiver UPI ID', 'Amount (INR)']]
    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    threshold = 0.95 * np.max(mse)
    y_pred = ["FRAUD" if e > threshold else "LEGIT" for e in mse]
    data['Prediction'] = y_pred
    data['Reconstruction Error'] = mse
    return data, threshold

def predict_transaction(sender_upi, receiver_upi, amount):
    sender_encoded = abs(hash(sender_upi)) % (10**6)
    receiver_encoded = abs(hash(receiver_upi)) % (10**6)
    X = np.array([[sender_encoded, receiver_encoded, amount]])
    X_scaled = scaler.transform(X)
    X_pred = model.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
    threshold = 0.95 * np.max(mse)
    prediction = "FRAUD" if mse[0] > threshold else "LEGIT"
    return prediction, mse[0], threshold

st.set_page_config(page_title="UPI Fraud Detection", layout="wide")
st.title("🔒 UPI Fraud Detection using Autoencoder")

# User selects approach
approach = st.radio("Choose Input Method:", ("📁 CSV Upload", "📝 Manual Entry"))

if approach == "📁 CSV Upload":
    uploaded_file = st.file_uploader("Upload a CSV file with transactions", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=["Transaction ID", "Timestamp", "Sender Name", "Receiver Name"])
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

elif approach == "📝 Manual Entry":
    st.subheader("📝 Manual Transaction Check")

    with st.form("manual_input"):
        sender = st.text_input("Sender UPI ID", "user1@upi")
        receiver = st.text_input("Receiver UPI ID", "merchant@upi")
        amount = st.number_input("Amount (INR)", min_value=0.01, step=0.01)
        submitted = st.form_submit_button("Check Transaction")

        if submitted:
            prediction, error, threshold = predict_transaction(sender, receiver, amount)
            st.markdown(f"### 🧾 Prediction: **{prediction}**")
            st.markdown(f"- Reconstruction Error: `{error:.6f}`")
            st.markdown(f"- Threshold: `{threshold:.6f}`")

            fig2, ax2 = plt.subplots()
            sns.barplot(x=["Reconstruction Error", "Threshold"], y=[error, threshold], ax=ax2)
            ax2.set_title("Error vs Threshold")
            st.pyplot(fig2)
