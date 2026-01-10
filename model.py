import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model & Scalers
# -----------------------------
model = load_model("Model_Artifacts/car_purchase_prediction_model.keras")
feature_scaler = joblib.load("Model_Artifacts/scaler_X.pkl")
target_scaler = joblib.load("Model_Artifacts/scaler_y.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# -----------------------------
# UI Header
# -----------------------------
st.title("🚗 Car Sales Price Prediction")
st.markdown("Predict car purchase price using **Artificial Neural Network (ANN)**")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter Customer Details")
gender = st.selectbox("Gender", ["Female", "Male"])
gender_encoded = 1 if gender == "Male" else 0
age = st.number_input("Age", min_value=18, max_value=80, value=35)
annual_salary = st.number_input("Annual Salary ($)", min_value=10000, max_value=300000, value=60000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
net_worth = st.number_input("Net Worth ($)", min_value=0, max_value=1000000, value=100000)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Car Price"):
    
    # Prepare input
    input_data = np.array([[gender_encoded, age, annual_salary, credit_score, net_worth]])
    
    # Scale input
    input_scaled = feature_scaler.transform(input_data)
    
    # Predict (scaled output)
    prediction_scaled = model.predict(input_scaled)
    
    # Inverse scale output
    prediction = target_scaler.inverse_transform(prediction_scaled)
    
    # Display result
    st.success(f"💰 Estimated Car Purchase Price: **${prediction[0][0]:,.2f}**")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Built with ❤️ using Streamlit & TensorFlow")
