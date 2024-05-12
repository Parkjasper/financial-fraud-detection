import streamlit as st
import joblib
import numpy as np
from pydantic import BaseModel

# Load the machine learning model
model = joblib.load('Frauddetection.pkl')

# Define the Streamlit app title and description
st.title("Financial Transaction Fraud Detection")
st.write("""
An API that utilizes a Machine Learning model to detect if a financial transaction is fraudulent or not.""")

# Define the fraudDetection class for input data validation
class fraudDetection(BaseModel):
    step: int
    types: int
    amount: float
    oldbalanceorig: float
    oldbalancedest: float
    newbalanceorig: float
    newbalancedest: float
    isflaggedfraud: int

# Function to predict fraud based on input data
def predict_fraud(data: fraudDetection):
    try:
        # Convert input data to numpy array
        features = np.array([[data.step, data.types, data.amount, data.oldbalanceorig, data.newbalanceorig, data.oldbalancedest, data.newbalancedest, data.isflaggedfraud]])

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
    st.markdown("### Financial Transaction Fraud Detection 🙌🏻")

    # Display instructions
    st.markdown("#### Instructions:")
    st.markdown("- Enter the transaction details in the sidebar.")
    st.markdown("- Click the 'Predict Fraud' button to see the prediction result.")
    st.markdown("-For types of transaction⬇️")
    st.markdown("""
                 0 for 'Cash In' Transaction\n 
                 1 for 'Cash Out' Transaction\n 
                 2 for 'Debit' Transaction\n
                 3 for 'Payment' Transaction\n  
                 4 for 'Transfer' Transaction\n""")

    # Collect input data from the user
    st.sidebar.markdown("### Input Transaction Details:")
    step = st.sidebar.number_input("Step", value=0, step=1)
    types = st.sidebar.number_input("Types", value=0, step=1)
    amount = st.sidebar.number_input("Amount", value=0.0, step=0.01)
    oldbalanceorig = st.sidebar.number_input("Sender balance before transaction was made", value=0.0, step=0.01)
    oldbalancedest = st.sidebar.number_input("Sender balance after transaction was made", value=0.0, step=0.01)
    newbalanceorig = st.sidebar.number_input("Receiver balance before transaction was made", value=0.0, step=0.01)
    newbalancedest = st.sidebar.number_input("Receiver balance after transaction was made", value=0.0, step=0.01)
    isflaggedfraud = 0
    if amount >= 200000:
        isflaggedfraud = 1
    else:
        isflaggedfraud = 0

    # Create a fraudDetection object
    input_data = fraudDetection(
        step=step,
        types=types,
        amount=amount,
        oldbalanceorig=oldbalanceorig,
        oldbalancedest=oldbalancedest,
        newbalanceorig=newbalanceorig,
        newbalancedest=newbalancedest,
        isflaggedfraud=isflaggedfraud
    )

    # Predict fraud based on input data
    if st.sidebar.button("Predict Fraud"):
        prediction_result = predict_fraud(input_data)
        st.write(f"The transaction is {prediction_result}.")
