import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Medical Cost & Risk Predictor", layout="wide")
st.title("🏥 Medical Cost & High-Risk Prediction Tool")

# Load models and column list
lin_model = joblib.load('insurance_linear_model.pkl')
log_model = joblib.load('insurance_logistic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Sidebar to choose prediction type
st.sidebar.header("Managerial Decision Tool")
prediction_type = st.sidebar.radio(
    "What would you like to predict?",
    ("Medical Charges (Linear Regression)", "High-Cost Risk (Logistic Regression)")
)

st.header("Patient Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    
with col2:
    sex = st.selectbox("Sex", ['male', 'female'])
    region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
    smoker_input = st.selectbox("Smoker", ['no', 'yes'])

if st.button("Get Prediction", type="primary"):
    # Prepare the raw input DataFrame
    input_data = {
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_input],
        'region': [region]
    }
    input_df = pd.DataFrame(input_data)

    # --- Feature Engineering (Must match Colab exactly) ---
    input_df['is_smoker'] = (input_df['smoker'] == 'yes').astype(int)
    input_df['smoker_bmi_interaction'] = input_df['is_smoker'] * input_df['bmi']
    
    # Drop the original smoker column to match training data
    input_df = input_df.drop('smoker', axis=1)

    # Ensure column order matches training
    input_df = input_df[model_columns]

    if prediction_type == "Medical Charges (Linear Regression)":
        # Predict Charges
        prediction = lin_model.predict(input_df)[0]
        st.success(f"### Predicted Medical Charges: **${prediction:,.2f}**")
        
    else:
        # Predict High-Cost Risk
        proba = log_model.predict_proba(input_df)[0][1]
        prediction = log_model.predict(input_df)[0]
        
        status = "🔴 HIGH RISK (Above Median Cost)" if prediction == 1 else "🟢 LOW RISK (Below Median Cost)"
        st.success(f"### High-Cost Risk: {status}")
        st.info(f"Probability of being a high-cost patient: {proba:.2%}")
        
        if proba > 0.6:
            st.warning("Consider assigning a case manager to this patient for preventive care.")
