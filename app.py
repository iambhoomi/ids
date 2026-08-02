import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('model.pkl')

st.title("Intrusion Detection System")

# Create input fields for 4 features
feature1 = st.number_input("Enter feature 1")
feature2 = st.number_input("Enter feature 2")
feature3 = st.number_input("Enter feature 3")
feature4 = st.number_input("Enter feature 4")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction[0])