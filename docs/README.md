
# ModelGuard 🛡️

Applied Machine Learning project for fraud detection with explainable AI.

## Features:
- Binary classification of credit card transactions
- XGBoost model for high performance
- SHAP for model explainability
- Streamlit dashboard for live interaction

## Run Locally:
```bash
pip install streamlit pandas xgboost shap joblib
cd scripts && python train_model.py
cd ../dashboard && streamlit run app.py
```

## Sample Dataset Format:
- time
- amount
- class (0 = normal, 1 = fraud)
