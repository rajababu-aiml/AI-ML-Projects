import os
import pickle
import streamlit as st

st.title("Flight Price Prediction Live Demo")

# Model load karne ka pakka tareeka
model_path = os.path.join(os.getcwd(), "Flight_Price_Prediction", "flight_model.pkl")
if not os.path.exists(model_path):
    model_path = "flight_model.pkl"

model = pickle.load(open(model_path, "rb"))

st.write("Model successfully loaded! Yahan aapke inputs aayenge.")

if st.button("Predict Price"):
    st.success("Sample Predicted Price: ₹ 5,500")
