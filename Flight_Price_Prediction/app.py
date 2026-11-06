import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Config
st.set_page_config(page_title="Flight Price Prediction", page_icon="✈️", layout="wide")

st.title("✈️ Flight Price Prediction Live Demo")
st.markdown("This application predicts flight ticket prices based on various parameters using a Machine Learning model.")

# Load the trained model safely
model_path = "flight_model.pkl"
if not os.path.exists(model_path):
    model_path = "Flight_Price_Prediction/flight_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("Model file (`flight_model.pkl`) not found! Please ensure it is uploaded.")
else:
    st.success("Model successfully loaded!")

    st.sidebar.header("🔍 Enter Flight Details")

    # User Input Widgets based on your project features
    col1, col2 = st.sidebar.columns(2)

    with col1:
        airline = st.selectbox("Airline", ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Multiple carriers", "Vistara", "Air Asia", "GoAir"])
        source = st.selectbox("Source", ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"])
        journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=24)
        dep_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=10)
        arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=20)
        duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=40, value=2)

    with col2:
        destination = st.selectbox("Destination", ["New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"])
        total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4], format_func=lambda x: "Non-stop" if x == 0 else f"{x} Stop(s)")
        journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)
        dep_min = st.number_input("Departure Minute (0-59)", min_value=0, max_value=59, value=30)
        arrival_min = st.number_input("Arrival Minute (0-59)", min_value=0, max_value=59, value=10)
        duration_mins = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=50)

    if st.button("Predict Price", type="primary"):
        # Construct input dataframe matching training features structure (30 columns)
        # Defining all categorical options for One-Hot Encoding alignment
        airlines_list = ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 
                         'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 
                         'Trujet', 'Vistara', 'Vistara Premium economy']
        
        sources_list = ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']
        destinations_list = ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

        input_data = {
            'Total_Stops': [total_stops],
            'Journey_day': [journey_day],
            'Journey_month': [journey_month],
            'Dep_hour': [dep_hour],
            'Dep_min': [dep_min],
            'Arrival_hour': [arrival_hour],
            'Arrival_min': [arrival_min],
            'Duration_hours': [duration_hours],
            'Duration_mins': [duration_mins]
        }

        # Dynamic One-Hot Encoding for Airline
        for a in airlines_list:
            col_name = f"Airline_{a}"
            input_data[col_name] = [1 if airline == a else 0]

        # Dynamic One-Hot Encoding for Source
        for s in sources_list:
            col_name = f"Source_{s}"
            input_data[col_name] = [1 if source == s else 0]

        # Dynamic One-Hot Encoding for Destination
        for d in destinations_list:
            col_name = f"Destination_{d}"
            input_data[col_name] = [1 if destination == d else 0]

        df_pred = pd.DataFrame(input_data)

        try:
            # Predict using the loaded model
            prediction = model.predict(df_pred)
            st.markdown(f"### Predicted Flight Ticket Price: **₹ {int(prediction[0]):,}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
