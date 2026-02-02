#generate_diverse_training_data.py
"""
Generate Diverse Training Data - FIXED VERSION
Creates data from MULTIPLE patients with GUARANTEED profile distribution
"""

import sys
sys.path.append('src')

from data.realtime_health_simulator import WearableDeviceSimulator
from data.generate_lab_data import LabDataGenerator
from data.data_integration_pipeline import DataIntegrationPipeline
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_multi_patient_dataset(
    n_patients: int = 130,
    days_per_patient: int = 30,
    sampling_minutes: int = 5
):
    """
    Generate training data from MULTIPLE patients with GUARANTEED diversity
    
    Args:
        n_patients: Number of patients to simulate
        days_per_patient: Days of data per patient
        sampling_minutes: Minutes between readings
    """
    
    print("\n" + "="*80)
    print(" "*20 + "MULTI-PATIENT TRAINING DATA GENERATION")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"   Patients: {n_patients}")
    print(f"   Days per patient: {days_per_patient}")
    print(f"   Sampling: Every {sampling_minutes} minutes")
    
    # FIXED: Deterministic profile assignment
    # 60% healthy, 30% at-risk, 10% deteriorating
    n_healthy = int(n_patients * 0.50)      
    n_at_risk = int(n_patients * 0.30)      
    n_deteriorating = n_patients - n_healthy - n_at_risk
    
    print(f"\n👥 Profile Distribution (DETERMINISTIC):")
    print(f"   Healthy: {n_healthy} patients ({n_healthy/n_patients*100:.0f}%)")
    print(f"   At-risk: {n_at_risk} patients ({n_at_risk/n_patients*100:.0f}%)")
    print(f"   Deteriorating: {n_deteriorating} patients ({n_deteriorating/n_patients*100:.0f}%)")
    
    all_wearable_data = []
    all_lab_data = []
    
    patient_num = 0
    
    # Generate healthy patients
    for i in range(n_healthy):
        patient_num += 1
        health_profile = 'healthy'
        patient_id = f"patient_{health_profile}_{i+1:03d}"
        
        print(f"\n{'='*80}")
        print(f"PATIENT {patient_num}/{n_patients} - {health_profile.upper()}")
        print('='*80)
        print(f"Patient ID: {patient_id}")
        
        readings = _generate_patient_data(
            patient_id, health_profile, days_per_patient, sampling_minutes
        )
        
        all_wearable_data.extend(readings['wearable'])
        all_lab_data.append(readings['lab'])
    
    # Generate at-risk patients
    for i in range(n_at_risk):
        patient_num += 1
        health_profile = 'at_risk'
        patient_id = f"patient_{health_profile}_{i+1:03d}"
        
        print(f"\n{'='*80}")
        print(f"PATIENT {patient_num}/{n_patients} - {health_profile.upper()}")
        print('='*80)
        print(f"Patient ID: {patient_id}")
        
        readings = _generate_patient_data(
            patient_id, health_profile, days_per_patient, sampling_minutes
        )
        
        all_wearable_data.extend(readings['wearable'])
        all_lab_data.append(readings['lab'])
    
    # Generate deteriorating patients
    for i in range(n_deteriorating):
        patient_num += 1
        health_profile = 'deteriorating'
        patient_id = f"patient_{health_profile}_{i+1:03d}"
        
        print(f"\n{'='*80}")
        print(f"PATIENT {patient_num}/{n_patients} - {health_profile.upper()}")
        print('='*80)
        print(f"Patient ID: {patient_id}")
        
        readings = _generate_patient_data(
            patient_id, health_profile, days_per_patient, sampling_minutes
        )
        
        all_wearable_data.extend(readings['wearable'])
        all_lab_data.append(readings['lab'])
    
    # Combine all patients
    print("\n" + "="*80)
    print("COMBINING ALL PATIENTS")
    print("="*80)
    
    wearable_df = pd.DataFrame(all_wearable_data)
    lab_df = pd.concat(all_lab_data, ignore_index=True)
    
    print(f"\n📊 Combined Dataset:")
    print(f"   Total wearable readings: {len(wearable_df):,}")
    print(f"   Total lab tests: {len(lab_df):,}")
    print(f"   Unique patients: {wearable_df['patient_id'].nunique()}")
    
    # Verify profile distribution
    print(f"\n✅ Profile Verification:")
    for profile in ['healthy', 'at_risk', 'deteriorating']:
        count = sum(1 for p in wearable_df['patient_id'].unique() if profile in p)
        avg_hr = wearable_df[wearable_df['patient_id'].str.contains(profile)]['heart_rate'].mean()
        print(f"   {profile.capitalize()}: {count} patients, avg HR = {avg_hr:.1f} bpm")
    
    # Save raw data
    wearable_path = 'data/processed/wearable_data_multi.csv'
    lab_path = 'data/processed/lab_results_multi.csv'
    
    wearable_df.to_csv(wearable_path, index=False)
    lab_df.to_csv(lab_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   {wearable_path}")
    print(f"   {lab_path}")
    
    # Run integration pipeline with OPTIMIZED label creation
    print("\n" + "="*80)
    print("RUNNING OPTIMIZED INTEGRATION PIPELINE")
    print("="*80)
    
    # Use optimized pipeline
    features_df = _create_features_optimized(wearable_df, lab_df)
    
    # Save features
    features_path = 'data/processed/features_multi.csv'
    features_df.to_csv(features_path, index=False)
    
    # Final summary
    print("\n" + "="*80)
    print(" "*30 + "🎉 COMPLETE! 🎉")
    print("="*80)
    
    print(f"\n🎯 ML-Ready Dataset Created:")
    print(f"   File: {features_path}")
    print(f"   Samples: {len(features_df):,}")
    print(f"   Features: {len(features_df.columns) - 2}")
    print(f"   Patients: {features_df['patient_id'].nunique()}")
    
    # Class distribution
    n_pos = (features_df['label'] == 1).sum()
    n_neg = (features_df['label'] == 0).sum()
    
    print(f"\n📊 Class Distribution:")
    print(f"   Deterioration: {n_pos:,} ({n_pos/len(features_df)*100:.1f}%)")
    print(f"   Stable: {n_neg:,} ({n_neg/len(features_df)*100:.1f}%)")
    if n_pos > 0:
        print(f"   Imbalance ratio: {n_neg/n_pos:.1f}:1")
    
    print("\n🚀 Next: python train_models.py")
    
    return features_df


def _generate_patient_data(patient_id, health_profile, days, sampling_minutes):
    """Generate data for a single patient"""
    
    # Create device with specific profile
    device = WearableDeviceSimulator(
        patient_id=patient_id,
        sampling_rate_seconds=sampling_minutes * 60,
        health_profile=health_profile
    )
    
    # Generate timeline
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Event probability by profile
    event_probability = {
        'healthy': 0.35,
        'at_risk': 0.65,
        'deteriorating': 0.90
    }
    
    has_event = np.random.random() < event_probability[health_profile]
    
    if has_event:
        event_time = start_time + timedelta(days=np.random.uniform(5, days-5))
        event_types = ['hypertensive_crisis', 'hypoxia', 'sepsis', 'cardiac']
        event_type = np.random.choice(event_types)
        print(f"⚠️  Scheduled event: {event_type} at day {(event_time - start_time).days}")
        event_triggered = False
        event_resolved = False
    else:
        print(f"✅ No deterioration events")
        event_time = None
    
    # Generate readings
    current_time = start_time
    patient_readings = []
    
    while current_time <= end_time:
        # Handle events
        if has_event and not event_triggered and current_time >= event_time:
            device.trigger_deterioration(event_type)
            event_triggered = True
        
        if has_event and event_triggered and not event_resolved:
            hours_since = (current_time - event_time).total_seconds() / 3600
            if hours_since > np.random.uniform(6, 12):
                device.stop_deterioration()
                event_resolved = True
        
        reading = device.generate_reading(current_time)
        patient_readings.append(reading)
        
        current_time += timedelta(minutes=sampling_minutes)
    
    print(f"✅ Generated {len(patient_readings)} readings")
    
    # Generate lab data
    lab_generator = LabDataGenerator()
    patient_lab_df = lab_generator.generate_for_timerange(
        start_date=start_time,
        end_date=end_time,
        test_frequency_days=30,
        patient_id=patient_id
    )
    
    print(f"✅ Generated {len(patient_lab_df)} lab tests")
    
    return {
        'wearable': patient_readings,
        'lab': patient_lab_df
    }


def _create_features_optimized(wearable_df, lab_df):
    """
    OPTIMIZED feature creation - processes patients in batches
    Much faster than processing all 432K rows at once
    """
    import yaml
    from datetime import timedelta
    
    print("\n🔧 Creating features (optimized for speed)...")
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    prediction_window = config['data']['prediction_window_hours']
    
    # FIXED: Convert timestamps BEFORE merging
    print("   Converting timestamps...")
    wearable_df['timestamp'] = pd.to_datetime(wearable_df['timestamp'])
    lab_df['test_date'] = pd.to_datetime(lab_df['test_date'])
    
    # Merge lab data
    print("   Merging lab data...")
    lab_df_merge = lab_df.rename(columns={'test_date': 'timestamp'})
    lab_features = [col for col in lab_df_merge.columns 
                   if col not in ['patient_id', 'timestamp', 'test_type']]
    
    merged_df = pd.merge_asof(
        wearable_df.sort_values('timestamp'),
        lab_df_merge[['timestamp'] + lab_features].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    # OPTIMIZED: Process by patient (much faster)
    print(f"   Creating labels (patient-by-patient)...")
    
    all_features = []
    patients = merged_df['patient_id'].unique()
    
    for i, patient_id in enumerate(patients, 1):
        if i % 10 == 0:
            print(f"      Progress: {i}/{len(patients)} patients")
        
        patient_data = merged_df[merged_df['patient_id'] == patient_id].copy()
        patient_data = patient_data.sort_values('timestamp').reset_index(drop=True)
        
        # Create labels for this patient
        labels = []
        
        for idx in range(len(patient_data)):
            current_time = patient_data.loc[idx, 'timestamp']
            future_time = current_time + timedelta(hours=prediction_window)
            
            # Look ahead in THIS patient's data only
            future_mask = (patient_data['timestamp'] > current_time) & \
                         (patient_data['timestamp'] <= future_time)
            future_data = patient_data[future_mask]
            
            if len(future_data) == 0:
                labels.append(np.nan)
                continue
            
            # FIXED: Check for SUSTAINED deterioration using vectorized operations (MUCH FASTER)
            deterioration = False
            
            # Use vectorized operations instead of iterating rows
            if len(future_data) > 0:
                # Count abnormal readings for each vital
                abnormal_flags = []
                
                # Heart rate thresholds
                if 'heart_rate' in future_data.columns:
                    hr_abnormal = ((future_data['heart_rate'] < 40) | 
                                  (future_data['heart_rate'] > 140))
                    abnormal_flags.append(hr_abnormal)
                
                # Blood pressure thresholds
                if 'bp_systolic' in future_data.columns:
                    bp_abnormal = ((future_data['bp_systolic'] < 80) | 
                                  (future_data['bp_systolic'] > 200))
                    abnormal_flags.append(bp_abnormal)
                
                # SpO2 threshold
                if 'spo2' in future_data.columns:
                    spo2_abnormal = (future_data['spo2'] < 88)
                    abnormal_flags.append(spo2_abnormal)
                
                # Temperature thresholds
                if 'temperature' in future_data.columns:
                    temp_abnormal = ((future_data['temperature'] < 35.5) | 
                                    (future_data['temperature'] > 39.0))
                    abnormal_flags.append(temp_abnormal)
                
                # Combine all flags - if ANY vital is abnormal, count that reading as abnormal
                if abnormal_flags:
                    any_abnormal = pd.concat(abnormal_flags, axis=1).any(axis=1)
                    abnormal_count = any_abnormal.sum()
                    total_count = len(future_data)
                    
                    if total_count > 0 and (abnormal_count / total_count) > 0.25:
                        deterioration = True
            
            labels.append(1 if deterioration else 0)
        
        patient_data['label'] = labels
        all_features.append(patient_data)
    
    # Combine all patients
    features_df = pd.concat(all_features, ignore_index=True)
    features_df = features_df.dropna(subset=['label'])
    
    print(f"   ✅ Features created: {len(features_df):,} samples")
    
    return features_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--patients', type=int, default=130, help='Number of patients')
    parser.add_argument('--days', type=int, default=30, help='Days per patient')
    parser.add_argument('--quick', action='store_true', help='Quick test: 10 patients, 14 days')
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n⚡ QUICK MODE")
        n_patients = 10
        days = 14
    else:
        n_patients = args.patients
        days = args.days
    
    features_df = generate_multi_patient_dataset(
        n_patients=n_patients,
        days_per_patient=days,
        sampling_minutes=5
    )