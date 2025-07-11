import streamlit as st
import pandas as pd
import shap
import joblib
import os

# Import local SHAP explainer
from shap_explainer import explain_model

# Streamlit page setup
st.set_page_config(page_title="ModelGuard Fraud Detection", layout="wide")
st.title("💳 ModelGuard – Fraud Detection with Explainable AI")

# File upload
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Load model
    model_path = os.path.join("models", "fraud_model.pkl")
    model = joblib.load(model_path)

    # Predict
    preds = model.predict(df[['time', 'amount']])
    df['Prediction'] = preds

    # Show predictions
    st.success("✅ Predictions Generated Successfully!")
    st.dataframe(df)

    # SHAP explainability
    if st.checkbox("📊 Show SHAP Explainability"):
        shap_values = explain_model(df[['time', 'amount']])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("🔍 SHAP Waterfall Plot (first prediction)")
        shap.plots.waterfall(shap_values[0], max_display=4)
        st.pyplot()