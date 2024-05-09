import streamlit as st
import joblib
import numpy as np
from pydantic import BaseModel

# Load the machine learning model
model = joblib.load('Frauddetection.pkl')

# Define the Streamlit app title and description
st.title("Credit Card Fraud Detection API")
st.write("""
An API that utilizes a Machine Learning model to detect if a credit card transaction is fraudulent or not based on features such as hours, amount, and transaction type.
""")

# Define the fraudDetection class for input data validation
class fraudDetection(BaseModel):
    step: int
    types: int
    amount: float
    oldbalanceorig: float
    newbalanceorig: float
    oldbalancedest: float
    newbalancedest: float

# Function to predict fraud based on input data
def predict_fraud(data: fraudDetection):
    try:
        # Convert input data to numpy array
        features = np.array([[data.step, data.types, data.amount, data.oldbalanceorig, data.newbalanceorig, data.oldbalancedest, data.newbalancedest], data.isflaggedfraud])

        # Log input features
        st.write("Input features:", features)

        # Use the loaded model to predict
        prediction = model.predict(features)

        # Log prediction
        st.write("Prediction:", prediction)

        # Return prediction result
        return "fraudulent" if prediction == 1 else "not fraudulent"

    except Exception as e:
        # Log and return error message
        st.error(f"An error occurred: {str(e)}")
        return "Error"

# Streamlit app main code
if __name__ == "__main__":
    # Display introductory message
    st.markdown("### Credit Card Fraud Detection API 🙌🏻")

    # Display instructions
    st.markdown("#### Instructions:")
    st.markdown("- Enter the transaction details in the sidebar.")
    st.markdown("- Click the 'Predict Fraud' button to see the prediction result.")

    # Collect input data from the user
    st.sidebar.markdown("### Input Transaction Details:")
    step = st.sidebar.number_input("Step", value=0, step=1)
    types = st.sidebar.number_input("Types", value=0, step=1)
    amount = st.sidebar.number_input("Amount", value=0.0, step=0.01)
    oldbalanceorig = st.sidebar.number_input("Old Balance Orig", value=0.0, step=0.01)
    newbalanceorig = st.sidebar.number_input("New Balance Orig", value=0.0, step=0.01)
    oldbalancedest = st.sidebar.number_input("Old Balance Dest", value=0.0, step=0.01)
    newbalancedest = st.sidebar.number_input("New Balance Dest", value=0.0, step=0.01)
    isflaggedfraud = 0
if amount >= 200000:
  isflaggedfraud = 1
else:
  isflaggedfraud = 0

    # Create a fraudDetection object
    input_data = fraudDetection(step=step, types=types, amount=amount, oldbalanceorig=oldbalanceorig, newbalanceorig=newbalanceorig, oldbalancedest=oldbalancedest, newbalancedest=newbalancedest, isflaggedfraud=isflaggedfraud)

    # Predict fraud based on input data
    if st.sidebar.button("Predict Fraud"):
        prediction_result = predict_fraud(input_data)
        st.write(f"The transaction is {prediction_result}.")
