import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, encoders, and scaler
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("minmax_scaler.pkl")

# Get encoders for STATE and DISTRICT
state_le = label_encoders["STATE_NAME"]
district_le = label_encoders["DISTRICT_NAME"]

# Define column names in same order as training data
input_columns = [
    "STATE_NAME",
    "DISTRICT_NAME",
    "NO_OF_ROAD_WORK_SANCTIONED",
    "LENGTH_OF_ROAD_WORK_SANCTIONED",
    "NO_OF_BRIDGES_SANCTIONED",
    "COST_OF_WORKS_SANCTIONED",
    "NO_OF_ROAD_WORKS_COMPLETED",
    "LENGTH_OF_ROAD_WORK_COMPLETED",
    "NO_OF_BRIDGES_COMPLETED",
    "EXPENDITURE_OCCURED",
    "NO_OF_ROAD_WORKS_BALANCE",
    "LENGTH_OF_ROAD_WORK_BALANCE",
    "NO_OF_BRIDGES_BALANCE"
]

st.title("PMGSY Scheme Classifier")

st.image(
    "https://cdnbbsr.s3waas.gov.in/s3e6c2dc3dee4a51dcec3a876aa2339a78/uploads/2025/01/20250129686928869.jpg",
    caption="Rural Road Construction under PMGSY",
    use_column_width=True
)


# Categorical dropdowns
state_input = st.selectbox("Select State", state_le.classes_)
district_input = st.selectbox("Select District", district_le.classes_)

# Numerical inputs
no_of_road_work_sanctioned = st.number_input("No. of Road Works Sanctioned", min_value=0)
length_of_road_work_sanctioned = st.number_input("Length of Road Works Sanctioned (in km)", min_value=0.0)
no_of_bridges_sanctioned = st.number_input("No. of Bridges Sanctioned", min_value=0)
cost_of_works_sanctioned = st.number_input("Cost of Works Sanctioned (₹ Lakhs)", min_value=0.0)
no_of_road_works_completed = st.number_input("No. of Road Works Completed", min_value=0)
length_of_road_work_completed = st.number_input("Length of Road Work Completed (in km)", min_value=0.0)
no_of_bridges_completed = st.number_input("No. of Bridges Completed", min_value=0)
expenditure_occured = st.number_input("Expenditure Occurred (₹ Lakhs)", min_value=0.0)
no_of_road_works_balance = st.number_input("No. of Road Works Balance", min_value=0)
length_of_road_work_balance = st.number_input("Length of Road Work Balance (in km)", min_value=0.0)
no_of_bridges_balance = st.number_input("No. of Bridges Balance", min_value=0)

# Predict button
if st.button("Predict PMGSY Scheme"):
    # Encode categorical values using saved encoders
    state_encoded = state_le.transform([state_input])[0]
    district_encoded = district_le.transform([district_input])[0]

    # Prepare input data
    input_data = pd.DataFrame([[
        state_encoded,
        district_encoded,
        no_of_road_work_sanctioned,
        length_of_road_work_sanctioned,
        no_of_bridges_sanctioned,
        cost_of_works_sanctioned,
        no_of_road_works_completed,
        length_of_road_work_completed,
        no_of_bridges_completed,
        expenditure_occured,
        no_of_road_works_balance,
        length_of_road_work_balance,
        no_of_bridges_balance
    ]], columns=input_columns)

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Predict using the trained model
    prediction = model.predict(input_scaled)

    # Load label encoder for target variable (if saved), or fit manually from training data if needed
    print(label_encoders)
    target_le = label_encoders.get("PMGSY_SCHEME", None)
    if target_le is not None:
        predicted_scheme = target_le.inverse_transform(prediction)[0]
    else:
        predicted_scheme = prediction[0]

    # Show result
    st.success(f"Predicted PMGSY Scheme: **{predicted_scheme}**")
