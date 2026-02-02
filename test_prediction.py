# test_prediction.py
from predictor import HealthPredictor
import pandas as pd

# Load some training data
df = pd.read_csv("data/processed/features_engineered.csv").head(100)

predictor = HealthPredictor(model_type="xgboost")

# Test prediction
result = predictor.predict(df, patient_id='test_patient')

print(f"Risk Score: {result['risk_score']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Alert: {result['alert']}")