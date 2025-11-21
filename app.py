import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Airline Passenger Satisfaction Prediction")
st.write("Enter passenger details to predict satisfaction (Satisfied / Neutral or Dissatisfied)")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=25)
flight_distance = st.number_input("Flight Distance", min_value=1, max_value=10000, value=500)
departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=500, value=0)
arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=500, value=0)

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
travel_type = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])

# Service ratings
cols = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"
]

service_values = {}
for col in cols:
    service_values[col] = st.slider(col, 0, 5, 3)

# Prepare input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Flight Distance": flight_distance,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay,
    "Gender": 1 if gender == "Male" else 0,
    "Customer Type": 1 if customer_type == "Loyal Customer" else 0,
    "Type of Travel": 1 if travel_type == "Business travel" else 0,
    "Class": {"Eco": 0, "Eco Plus": 1, "Business": 2}[class_type],
    **service_values
}])

# ---- IMPORTANT ENGINEERED FEATURES (required by model) ----

# 1. Total Delay
input_data["Total Delay"] = (
    input_data["Departure Delay in Minutes"] +
    input_data["Arrival Delay in Minutes"]
)

# 2. Average Service Rating
service_cols = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"
]

input_data["Average Service Rating"] = input_data[service_cols].mean(axis=1)

# Predict
if st.button("Predict Satisfaction"):
    pred = model.predict(input_data)[0]
    result = "Satisfied 😀" if pred == 1 else "Neutral or Dissatisfied 😐"
    st.success(f"Prediction: {result}")
