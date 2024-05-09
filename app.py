import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load('Frauddetection.pkl')

# Function to make prediction
def predict(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type):
    try:
        # Convert input values to float and int as necessary
        amount = float(amount)
        oldbalanceOrg = float(oldbalanceOrg)
        newbalanceOrig = float(newbalanceOrig)
        oldbalanceDest = float(oldbalanceDest)
        newbalanceDest = float(newbalanceDest)
        transaction_type = int(transaction_type)

        # Create feature array
        features = np.array([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type]])

        # Use the loaded model to predict
        prediction = model.predict(features)

        # Return prediction result
        return "Fraudulent" if prediction == 1 else "Not Fraudulent"
    
    except ValueError:
        return "Error: Invalid input. Please ensure all input values are numeric."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Streamlit app
def main():
    st.title("Financial Transaction Fraud Detection")

    # Sidebar for user inputs
    st.sidebar.header('Transaction Details')
    amount = st.sidebar.number_input("Transaction Amount (Rs)")
    oldbalanceOrg = st.sidebar.number_input("Old Balance Orig (Rs)")
    oldbalanceDest = st.sidebar.number_input("Old Balance Dest (Rs)")
    newbalanceOrig = st.sidebar.number_input("New Balance Orig (Rs)")
    newbalanceDest = st.sidebar.number_input("New Balance Dest (Rs)")
    transaction_type = st.sidebar.selectbox("Transaction Type", 
                                       {"CASH_IN": 0, "CASH_OUT": 1, "PAYMENT": 2})

    # Prediction and result display
    if st.sidebar.button("Detect Fraud"):
        result = predict(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type)
        st.write(f"*Prediction:* The transaction is *{result}*.")

    # Display sample data for reference
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

    # Model Evaluation
    st.sidebar.subheader("Model Evaluation")
    if st.sidebar.checkbox("Show Evaluation Metrics"):
        # Dummy evaluation data (replace with actual evaluation data)
        y_true = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1]

        st.subheader("Model Evaluation Metrics")
        report = classification_report(y_true, y_pred)
        st.text(report)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        st.pyplot()

if __name__ == "__main__":
    main()
