"""
Complete System Test
Tests the entire pipeline: API → Data Collection → Prediction
"""

import requests
import time

API_URL = "http://localhost:8000"

print("\n" + "="*70)
print("🏥 COMPLETE SYSTEM TEST")
print("="*70)

# Step 1: Create patient
print("\n1️⃣  Creating patient...")
response = requests.post(f"{API_URL}/patients", json={
    "patient_id": "demo_patient_001",
    "sampling_interval_seconds": 300
})

if response.status_code == 200:
    print("   ✅ Patient created")
else:
    print(f"   ⚠️  Status: {response.status_code}")

# Step 2: Get current vitals
print("\n2️⃣  Getting current vitals...")
response = requests.get(f"{API_URL}/patients/demo_patient_001/current")
data = response.json()['data']
print(f"   ✅ HR: {data['heart_rate']:.1f}, BP: {data['bp_systolic']:.0f}/{data['bp_diastolic']:.0f}")

# Step 3: Get history
print("\n3️⃣  Getting 2-hour history...")
response = requests.get(
    f"{API_URL}/patients/demo_patient_001/history",
    params={'hours': 2, 'interval_minutes': 5}
)
history = response.json()
print(f"   ✅ Retrieved {history['count']} readings")

# Step 4: Make prediction
print("\n4️⃣  Running ML prediction...")
from predictor import HealthPredictor
import pandas as pd

predictor = HealthPredictor(model_type="xgboost")
patient_data = pd.DataFrame(history['data'])

result = predictor.predict(patient_data)

print(f"\n🎯 PREDICTION RESULT:")
print(f"   Risk Score: {result['risk_score']:.3f}")
print(f"   Risk Level: {result['risk_level']}")
print(f"   Alert: {'⚠️ YES' if result['alert'] else '✓ NO'}")
print(f"   Interpretation: {result['interpretation']}")

# Step 5: Test deterioration event
print("\n5️⃣  Testing deterioration detection...")
print("   Triggering hypertensive crisis...")
response = requests.post(
    f"{API_URL}/patients/demo_patient_001/trigger-event",
    json={"patient_id": "demo_patient_001", "event_type": "hypertensive_crisis"}
)
print("   ✅ Event triggered")

time.sleep(2)

# Get new reading
response = requests.get(f"{API_URL}/patients/demo_patient_001/current")
data = response.json()['data']
print(f"   ⚠️  New vitals: HR: {data['heart_rate']:.1f}, BP: {data['bp_systolic']:.0f}/{data['bp_diastolic']:.0f}")

print("\n" + "="*70)
print("✅ SYSTEM TEST COMPLETE!")
print("="*70)