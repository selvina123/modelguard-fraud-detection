
import streamlit as st
import pandas as pd
import joblib
from explainability.shap_explainer import explain_model
import shap

st.title("💳 ModelGuard – Fraud Detection with Explainable AI")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model = joblib.load('../models/fraud_model.pkl')
    preds = model.predict(df[['time', 'amount']])
    df['prediction'] = preds
    st.write("Predictions:")
    st.dataframe(df)

    if st.checkbox("Show SHAP Explainability"):
        shap_values = explain_model(df[['time', 'amount']])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0], max_display=4)
        st.pyplot()
