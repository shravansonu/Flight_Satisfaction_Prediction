import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "models/model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Flight Satisfaction Prediction App ✈️")
st.write("Upload passenger data to predict whether they are **Satisfied** or **Dissatisfied**.")

model = load_model()

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(data.head())

    try:
        # Predict
        preds = model.predict(data)

        st.write("### Prediction Results")
        data["Prediction"] = preds
        data["Prediction"] = data["Prediction"].map({1: "Satisfied", 0: "Dissatisfied"})

        st.dataframe(data)

    except Exception as e:
        st.error(f"Error during prediction: {e}")


st.write("---")
st.write("Built with ❤️ using Streamlit")
