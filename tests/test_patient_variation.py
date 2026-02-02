"""
Test Patient Variation
Verify that different patient profiles generate different baselines
"""

import sys
sys.path.append('src')

from data.realtime_health_simulator import WearableDeviceSimulator
from datetime import datetime

def test_patient_variation():
    """Test that patients have different baselines"""
    
    print("\n" + "="*70)
    print("TESTING PATIENT VARIATION")
    print("="*70)
    
    # Create 3 patients with different profiles
    patients = [
        WearableDeviceSimulator(
            patient_id="patient_healthy_001",
            health_profile='healthy'
        ),
        WearableDeviceSimulator(
            patient_id="patient_atrisk_002",
            health_profile='at_risk'
        ),
        WearableDeviceSimulator(
            patient_id="patient_critical_003",
            health_profile='deteriorating'
        )
    ]
    
    print("\n" + "="*70)
    print("BASELINE VITALS COMPARISON")
    print("="*70)
    
    # Print baselines
    for device in patients:
        print(f"\n{device.health_profile.upper()} - {device.patient_id}")
        print(f"  HR: {device.baseline['heart_rate']:.1f} bpm")
        print(f"  BP: {device.baseline['bp_systolic']:.1f}/{device.baseline['bp_diastolic']:.1f} mmHg")
        print(f"  SpO2: {device.baseline['spo2']:.1f}%")
        print(f"  RR: {device.baseline['respiratory_rate']:.1f} /min")
        print(f"  Temp: {device.baseline['temperature']:.2f}°C")
        print(f"  Variability: {device.baseline_variability}")
    
    # Generate 10 readings from each
    print("\n" + "="*70)
    print("SAMPLE READINGS (Average of 10 readings)")
    print("="*70)
    
    for device in patients:
        readings = []
        for i in range(10):
            reading = device.generate_reading()
            readings.append(reading)
        
        # Calculate averages
        avg_hr = sum(r['heart_rate'] for r in readings) / 10
        avg_bp_sys = sum(r['bp_systolic'] for r in readings) / 10
        avg_spo2 = sum(r['spo2'] for r in readings) / 10
        
        print(f"\n{device.patient_id} ({device.health_profile}):")
        print(f"  Avg HR: {avg_hr:.1f} bpm")
        print(f"  Avg BP: {avg_bp_sys:.1f} mmHg")
        print(f"  Avg SpO2: {avg_spo2:.1f}%")
    
    # Verify differences
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    hr_healthy = patients[0].baseline['heart_rate']
    hr_atrisk = patients[1].baseline['heart_rate']
    hr_critical = patients[2].baseline['heart_rate']
    
    print(f"\nHeart Rate Baseline Differences:")
    print(f"  Healthy:  {hr_healthy:.1f} bpm")
    print(f"  At-risk:  {hr_atrisk:.1f} bpm  (Δ={hr_atrisk-hr_healthy:+.1f})")
    print(f"  Critical: {hr_critical:.1f} bpm (Δ={hr_critical-hr_healthy:+.1f})")
    
    if hr_healthy < hr_atrisk < hr_critical:
        print("\n✅ PASS: Baselines increase with risk level")
    else:
        print("\n❌ FAIL: Baselines should increase with risk level")
        return False
    
    # Check variability
    if (patients[0].baseline_variability < 
        patients[1].baseline_variability < 
        patients[2].baseline_variability):
        print("✅ PASS: Variability increases with risk level")
    else:
        print("❌ FAIL: Variability should increase with risk level")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Patient variation is working!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = test_patient_variation()
    
    if not success:
        print("\n⚠️  Tests failed! Fix the issues above.")
        sys.exit(1)
    
    print("\n🚀 Ready for training data generation!")