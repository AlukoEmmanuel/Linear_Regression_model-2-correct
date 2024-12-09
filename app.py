import streamlit as st
#import requests as
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

# app import title and description
st.title("Power Output Prediction")
st.write("Enter the values for the following features to predict power output (PE).")

# User input for each feature
ambient_temp = st.number_input("Ambient Temperature (AT)", value=15.0)
exhaust_vacuum = st.number_input("Exhaust Vacuum (V)", value=40.0)
ambient_pressure = st.number_input("Ambient Pressure (AP)", value=1000.0)
relative_humidity = st.number_input("Relative Humidity (RH)", value=75.0)

# Create a button for making the prediction
if st.button("Predict"):
    input_data = np.array([[ambient_temp, exhaust_vacuum, ambient_pressure, relative_humidity]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Power Output (PE): {prediction[0]}")

    # Use the codes below when connecting via fastapi
    # prepare data
    # input_data = {
    #    "AT": ambient_temp,
    #    "V": exhaust_vacuum,
    #    "AP": ambient_pressure,
    #    "RH": relative_humidity
    # }

    # make api call
    # response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

    # if response.status_code == 200:
    #    prediction = response.json()["prediction"]
    #    st.write(f"Predicted Power Output (PE): {prediction}")
    # else:
    #    st.write("Error in prediction. Please try again.")
