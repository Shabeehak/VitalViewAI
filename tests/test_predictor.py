"""
Quick Predictor Test
Tests if the trained model can make predictions
"""

import joblib
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("🧪 TESTING PREDICTOR")
print("="*70)

# Load model
print("\n📂 Loading model...")
model = joblib.load('models/xgboost_model.pkl')
print("   ✅ Model loaded")

# Load test data
print("\n📂 Loading test data...")
df = pd.read_csv('data/processed/features_engineered.csv')
print(f"   ✅ Loaded {len(df)} samples")

# Prepare test sample
exclude_cols = ['timestamp', 'patient_id', 'device_id', 'activity_state', 'label']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Get a few test samples
test_samples = df.sample(n=5, random_state=42)

print("\n🔮 Making predictions...")
print("="*70)

for idx, row in test_samples.iterrows():
    # Prepare features
    features = row[feature_cols].values.reshape(1, -1)
    
    # Make prediction
    risk_score = model.predict_proba(features)[0][1]
    prediction = "⚠️  DETERIORATION" if risk_score > 0.3 else "✅ STABLE"
    
    # Get actual label
    actual = "DETERIORATION" if row['label'] == 1 else "STABLE"
    
    print(f"\nSample {idx}:")
    print(f"   Patient: {row['patient_id']}")
    print(f"   Heart Rate: {row.get('heart_rate', 'N/A'):.1f} bpm")
    print(f"   BP: {row.get('bp_systolic', 'N/A'):.1f}/{row.get('bp_diastolic', 'N/A'):.1f} mmHg")
    print(f"   SpO2: {row.get('spo2', 'N/A'):.1f}%")
    print(f"   Risk Score: {risk_score:.1%}")
    print(f"   Prediction: {prediction}")
    print(f"   Actual: {actual}")
    print(f"   {'✅ CORRECT' if ((risk_score > 0.3) == (row['label'] == 1)) else '❌ INCORRECT'}")

print("\n" + "="*70)
print("✅ PREDICTOR TEST COMPLETE!")
print("="*70)
print("\n🚀 Predictor is working! Ready for dashboard integration.")