import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌱 Smart Crop Recommendation System")
st.write("Enter soil and environmental conditions")

# Inputs
N = st.number_input("Nitrogen (N)", 0, 200)
P = st.number_input("Phosphorus (P)", 0, 200)
K = st.number_input("Potassium (K)", 0, 200)
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
ph = st.number_input("pH value", 0.0, 14.0)
rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)

if st.button("Predict Crop"):
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    data_scaled = scaler.transform(data)
    
    prediction = model.predict(data_scaled)
    crop = le.inverse_transform(prediction)
    
    st.success(f"🌾 Recommended Crop: {crop[0]}")