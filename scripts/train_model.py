
import pandas as pd
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('data/sample_fraud.csv')
X = df[['time', 'amount']]
y = df['class']

# Train model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# Save model
joblib.dump(model, 'models/fraud_model.pkl')
print("Model saved!")
