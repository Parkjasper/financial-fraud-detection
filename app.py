import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('Frauddetection.pkl')

# Define the Streamlit app
st.title("Financial Transaction Fraud Detection")

# Description of the app
st.write("""
## About
Welcome to our Credit Card Fraud Detection Web App! This app aims to detect fraudulent credit card transactions based on transaction details such as amount, sender/receiver information, and transaction type.

**Contributors:** Jerin Jasper, Selvalakshmi, Santhiya, Aganiya
""")

# Upload CSV file containing transaction details
st.sidebar.header('Upload Transaction Details')
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Prediction function
def predict(transaction_data):
    # Make prediction
    prediction = model.predict(transaction_data)
    return "Fraudulent" if prediction == 1 else "Not Fraudulent"

# Detection result
if file is not None:
    df = pd.read_csv(file)
    st.write("Uploaded Transaction Details:")
    st.write(df)

    # Select relevant features
    features = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                   'newbalanceDest', 'isFlaggedFraud', 'Type']]

    # Make prediction
    result = predict(features.values)
    st.write(f"**Prediction:** The transaction is **{result}**.")

# Display some sample data
st.write("""
## Sample Transaction Details
Here are some sample transaction details that you can use for testing:
- Amount: 3478.18
- Old Balance Orig: 19853.00
- New Balance Orig: 16374.82
- Old Balance Dest: 0.00
- New Balance Dest: 0.00
- Transaction Type: CASH_OUT
""")
