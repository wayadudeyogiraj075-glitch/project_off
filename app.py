import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("Model Prediction App")

st.write("Enter the details below to get a prediction:")

# Create input fields based on model features
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
salary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Preprocess inputs
# Map Gender to numeric if your model expects it (0 for Female, 1 for Male)
gender_numeric = 1 if gender == "Male" else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame([[gender_numeric, age, salary]], 
                           columns=['Gender', 'Age', 'EstimatedSalary'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Result: Positive (1)")
    else:
        st.info("Result: Negative (0)")
