import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE = "/content/drive/MyDrive/customer-churn-predictor"

@st.cache_resource
def load_model():
    with open(f"{BASE}/models/churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{BASE}/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{BASE}/models/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they will churn.")
st.divider()

st.subheader("👤 Customer Details")
col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure          = st.slider("Tenure (months)", 0, 72, 12)
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

st.divider()
st.subheader("💳 Billing Details")
col3, col4 = st.columns(2)

with col3:
    streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col4:
    paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment          = st.selectbox("Payment Method", [
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"
                       ])
    monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges    = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.divider()

if st.button("🔍 Predict Churn", use_container_width=True):

    input_dict = {
        "gender":           1 if gender == "Male" else 0,
        "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "tenure":           tenure,
        "PhoneService":     1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    input_dict["MultipleLines_No phone service"] = 1 if multiple_lines == "No phone service" else 0
    input_dict["MultipleLines_Yes"]              = 1 if multiple_lines == "Yes" else 0
    input_dict["InternetService_Fiber optic"]    = 1 if internet == "Fiber optic" else 0
    input_dict["InternetService_No"]             = 1 if internet == "No" else 0
    input_dict["OnlineSecurity_No internet service"] = 1 if online_security == "No internet service" else 0
    input_dict["OnlineSecurity_Yes"]                 = 1 if online_security == "Yes" else 0
    input_dict["OnlineBackup_No internet service"]   = 1 if online_backup == "No internet service" else 0
    input_dict["OnlineBackup_Yes"]                   = 1 if online_backup == "Yes" else 0
    input_dict["DeviceProtection_No internet service"] = 1 if device_protect == "No internet service" else 0
    input_dict["DeviceProtection_Yes"]                 = 1 if device_protect == "Yes" else 0
    input_dict["TechSupport_No internet service"]    = 1 if tech_support == "No internet service" else 0
    input_dict["TechSupport_Yes"]                    = 1 if tech_support == "Yes" else 0
    input_dict["StreamingTV_No internet service"]    = 1 if streaming_tv == "No internet service" else 0
    input_dict["StreamingTV_Yes"]                    = 1 if streaming_tv == "Yes" else 0
    input_dict["StreamingMovies_No internet service"] = 1 if streaming_movies == "No internet service" else 0
    input_dict["StreamingMovies_Yes"]                 = 1 if streaming_movies == "Yes" else 0
    input_dict["Contract_One year"]                  = 1 if contract == "One year" else 0
    input_dict["Contract_Two year"]                  = 1 if contract == "Two year" else 0
    input_dict["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    input_dict["PaymentMethod_Electronic check"]         = 1 if payment == "Electronic check" else 0
    input_dict["PaymentMethod_Mailed check"]             = 1 if payment == "Mailed check" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    prediction    = model.predict(input_df)[0]
    probability   = model.predict_proba(input_df)[0][1]
    churn_percent = round(probability * 100, 1)

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is LIKELY TO CHURN")
        st.metric("Churn Probability", f"{churn_percent}%")
        st.markdown("""
        **Suggested Actions:**
        - 💰 Offer a discount or loyalty reward
        - 📞 Schedule a retention call
        - 📋 Suggest switching to a yearly contract
        """)
    else:
        st.success("✅ This customer is LIKELY TO STAY")
        st.metric("Churn Probability", f"{churn_percent}%")
        st.markdown("""
        **Customer looks healthy!**
        - 🌟 Consider upselling additional services
        - 📧 Send a satisfaction survey
        """)

    st.divider()
    st.subheader("📈 Churn Probability Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("Stay Probability",  f"{round((1 - probability) * 100, 1)}%")
    col_b.metric("Churn Probability", f"{churn_percent}%")
    st.progress(float(round(probability, 4)))

st.divider()
st.caption("Built with Python, XGBoost & Streamlit | Customer Churn Predictor Mini Project")
