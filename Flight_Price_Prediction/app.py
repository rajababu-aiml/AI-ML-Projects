import os
import pickle
import streamlit as st

st.title("Flight Price Prediction Live Demo")

# Model ko secure tarike se load karne ke liye path
model_path = os.path.join(os.path.dirname(__file__), "flight_model.pkl")
model = pickle.load(open(model_path, "rb"))

st.write("Model successfully loaded! Yahan aapke inputs aayenge.")

# Predict button
if st.button("Predict Price"):
    st.success("Sample Predicted Price: ₹ 5,500")
