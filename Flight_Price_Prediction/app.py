import pickle
import streamlit as st

st.title("Flight Price Prediction Live Demo")

# Saved model load karna
with open("flight_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write("Model successfully loaded! Yahan aapke inputs aayenge.")

# Predict button
if st.button("Predict Price"):
    st.success("Sample Predicted Price: ₹ 5,500")