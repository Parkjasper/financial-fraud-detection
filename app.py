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

# Input features of the transaction
st.sidebar.header('Transaction Details')
amount = st.sidebar.number_input("Transaction Amount (Rs)", min_value=0.0, max_value=110000.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance Orig (Rs)", min_value=0.0, max_value=110000.0)
oldbalanceDest = st.sidebar.number_input("Old Balance Dest (Rs)", min_value=0.0, max_value=110000.0)
newbalanceOrig = st.sidebar.number_input("New Balance Orig (Rs)", min_value=0.0, max_value=110000.0)
newbalanceDest = st.sidebar.number_input("New Balance Dest (Rs)", min_value=0.0, max_value=110000.0)
transaction_type = st.sidebar.selectbox("Transaction Type", 
                                       {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4})

# Prediction function
def predict(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type):
    # Make prediction
    features = np.array([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type]])
    prediction = model.predict(features)
    return "Fraudulent" if prediction == 1 else "Not Fraudulent"

# Detection result
if st.button("Detect Fraud"):
    result = predict(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type)
    st.write(f"**Prediction:** The transaction is **{result}**.")

# Display some sample data
st.write("""
## Sample Transaction Details
Here are some sample transaction details that you can use for testing:
- Amount: 3478.18 Rs
- Old Balance Orig: 19853.00 Rs
- New Balance Orig: 16374.82 Rs
- Old Balance Dest: 0.00 Rs
- New Balance Dest: 0.00 Rs
- Transaction Type: CASH_OUT
""")
