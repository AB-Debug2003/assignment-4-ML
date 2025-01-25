import streamlit as st
import numpy as np
import pickle

# Load the pre-trained SVM model (replace 'svm_model.pkl' with your actual model file)
model_path = "svm_model.pkl"
with open(model_path, "rb") as file:
    svm_model = pickle.load(file)

# Function to predict based on user input
def predict(age, estimated_salary, gender):
    # Preprocess the input
    gender_encoded = 1 if gender == "Male" else 0
    features = np.array([[gender_encoded, age, estimated_salary]])
    prediction = svm_model.predict(features)
    return "Purchased" if prediction[0] == 1 else "Not Purchased"

# Streamlit app
st.title("SVM Prediction App")
st.write("This app uses an SVM model to predict whether a user will make a purchase.")

# User input form
with st.form("prediction_form"):
    st.subheader("Enter user details")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=100, value=25)
    estimated_salary = st.number_input("Estimated Salary", min_value=1000, max_value=100000, step=1000, value=50000)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction and display
if submitted:
    result = predict(age, estimated_salary, gender)
    st.subheader("Prediction Result")
    st.write(f"The user is predicted to: **{result}**")