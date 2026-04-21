import streamlit as st
import pandas as pd
import pickle

# Load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("Model Prediction Dashboard")
st.write("Enter the details below to get a prediction.")

# Input fields based on model features
gender = st.selectbox("Gender", options=["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=25)
salary = st.number_input("Estimated Salary", min_value=0, value=50000)

# Convert Gender to numeric (assuming 1 for Male, 0 for Female - adjust if necessary)
gender_numeric = 1 if gender == "Male" else 0

# Create a DataFrame for the model
input_data = pd.DataFrame([[gender_numeric, age, salary]], 
                           columns=['Gender', 'Age', 'EstimatedSalary'])

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.success("Target Outcome: Yes")
    else:
        st.info("Target Outcome: No")
