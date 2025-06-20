
import shap
import pandas as pd
import joblib

def explain_model(input_df):
    model = joblib.load('../models/fraud_model.pkl')
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    return shap_values
