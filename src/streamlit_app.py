import streamlit as st
import requests
import pandas as pd

st.title("Water Potability Prediction")
st.write("Enter water quality metrics to check potability.")

# Input fields
ph = st.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness", 0.0, 300.0, 200.0)
solids = st.number_input("Solids", 0.0, 50000.0, 20000.0)
chloramines = st.number_input("Chloramines", 0.0, 14.0, 7.0)
sulfate = st.number_input("Sulfate", 0.0, 500.0, 300.0)
conductivity = st.number_input("Conductivity", 0.0, 800.0, 400.0)
organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 15.0)
trihalomethanes = st.number_input("Trihalomethanes", 0.0, 120.0, 60.0)
turbidity = st.number_input("Turbidity", 0.0, 7.0, 4.0)

if st.button("Predict"):
    payload = {
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }
    
    try:
        # Assuming the API is running on localhost:8000
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            # The API returns 'potability' (0 or 1)
            prediction = result.get("potability")
            probability = result.get("probability", 0.0)
            
            if prediction == 1:
                st.success(f"Water is Potable! (Confidence: {probability:.2f})")
            else:
                st.error(f"Water is Not Potable. (Confidence: {probability:.2f})")
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {e}")
