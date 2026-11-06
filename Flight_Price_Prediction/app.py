import os
import pickle
import streamlit as st

st.title("Flight Price Prediction Live Demo")

# Model ka sahi path set karna taaki Streamlit Cloud par error na aaye
model_path = os.path.join(os.path.dirname(__file__), "flight_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.write("Model successfully loaded! Yahan aapke inputs aayenge.")

# Predict button
if st.button("Predict Price"):
    st.success("Sample Predicted Price: ₹ 5,500")