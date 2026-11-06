import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Flight Price Prediction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for professional look
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0px;
    }
    .sub-text {
        color: #555555;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<p class="main-title">✈️ Flight Price Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">An end-to-end Machine Learning web application to estimate flight ticket fares accurately.</p>', unsafe_allow_html=True)
st.markdown("---")

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
    st.error("Model file (`flight_model.pkl`) not found in the repository! Please ensure it is uploaded.")
else:
    # Sidebar for Inputs
    st.sidebar.header("🎛️ Flight Parameters")
    st.sidebar.markdown("Provide the journey details below:")

    with st.sidebar.expander("📌 Airline & Route", expanded=True):
        airline = st.selectbox("Airline", ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Multiple carriers", "Vistara", "Air Asia", "GoAir"])
        source = st.selectbox("Source", ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"])
        destination = st.selectbox("Destination", ["New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"])
        total_stops = st.selectbox("Total Stops", [0, 1, 2, 3, 4], format_func=lambda x: "Non-stop" if x == 0 else f"{x} Stop(s)")

    with st.sidebar.expander("📅 Date & Schedule", expanded=False):
        journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=24)
        journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=3)
        dep_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=10)
        dep_min = st.number_input("Departure Minute (0-59)", min_value=0, max_value=59, value=30)
        arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=20)
        arrival_min = st.number_input("Arrival Minute (0-59)", min_value=0, max_value=59, value=10)

    with st.sidebar.expander("⏱️ Duration", expanded=False):
        duration_hours = st.number_input("Duration (Hours)", min_value=0, max_value=40, value=2)
        duration_mins = st.number_input("Duration (Minutes)", min_value=0, max_value=59, value=50)

    # Main Panel layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Selected Flight Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Airline", "Source", "Destination", "Total Stops", "Date", "Departure", "Arrival", "Duration"],
            "Details": [airline, source, destination, "Non-stop" if total_stops == 0 else f"{total_stops} Stop(s)", 
                        f"{journey_day}/{journey_month}/2019", f"{dep_hour:02d}:{dep_min:02d}", 
                        f"{arrival_hour:02d}:{arrival_min:02d}", f"{duration_hours}h {duration_mins}m"]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        predict_btn = st.button("🚀 Calculate Flight Price", type="primary", use_container_width=True)

    with col2:
        st.subheader("💡 Prediction Result")
        
        if predict_btn:
            with st.spinner("Analyzing flight data and predicting price..."):
                # One-Hot Encoding mappings matching training features structure
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

                for a in airlines_list:
                    input_data[f"Airline_{a}"] = [1 if airline == a else 0]

                for s in sources_list:
                    input_data[f"Source_{s}"] = [1 if source == s else 0]

                for d in destinations_list:
                    input_data[f"Destination_{d}"] = [1 if destination == d else 0]

                df_pred = pd.DataFrame(input_data)

                try:
                    prediction = model.predict(df_pred)
                    predicted_price = int(prediction[0])
                    
                    # Displaying professional metric card
                    st.metric(label="Estimated Ticket Fare", value=f"₹ {predicted_price:,}")
                    st.success("Prediction generated successfully!")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.info("Click **'Calculate Flight Price'** to view the estimated fare.")
