import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

model = joblib.load("insurance_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("🩺 Medical Insurance Cost Predictor")
st.markdown("Predict your estimated insurance charges based on your information.")

# Sidebar user input
st.sidebar.header("Enter Your Details")

age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.radio("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.radio("Do you smoke?", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encoding inputs
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0
region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region_encoded = region_map[region]

# Create DataFrame for input
input_data = pd.DataFrame(
    [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
    columns=["age", "sex", "bmi", "children", "smoker", "region"]
)

# Scale only 'age' and 'bmi'
input_data[['age', 'bmi']] = scaler.transform(input_data[['age', 'bmi']])

# Prediction
if st.sidebar.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted Charges:")
    st.success(f"${prediction:,.2f}")
