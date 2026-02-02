#generate_training_data.py
"""
Generate Realistic Training Data
Creates high-quality synthetic health data with realistic temporal patterns

This script generates complete datasets ready for ML training
"""

import sys
sys.path.append('src')

from data.realtime_health_simulator import WearableDeviceSimulator
from data.generate_lab_data import LabDataGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_wearable_data(
    patient_id: str = "patient_001",
    days: int = 30,
    sampling_minutes: int = 5,
    include_events: bool = True,
    output_path: str = "data/processed/wearable_data.csv"
) -> pd.DataFrame:
    """
    Generate realistic wearable data with temporal patterns
    
    Args:
        patient_id: Patient identifier
        days: Number of days to generate
        sampling_minutes: Minutes between readings
        include_events: Whether to include deterioration events
        output_path: Where to save CSV
        
    Returns:
        DataFrame with wearable data
    """
    print("\n" + "="*60)
    print("🏥 GENERATING REALISTIC WEARABLE DATA")
    print("="*60)
    
    print(f"\n📊 Configuration:")
    print(f"   Patient: {patient_id}")
    print(f"   Duration: {days} days")
    print(f"   Sampling: Every {sampling_minutes} minutes")
    print(f"   Deterioration events: {'Yes' if include_events else 'No'}")
    
    # Initialize device
    device = WearableDeviceSimulator(
        patient_id=patient_id,
        sampling_rate_seconds=sampling_minutes * 60,
        health_profile=None
    )
    
    # Calculate timeline
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    total_readings = int((days * 24 * 60) / sampling_minutes)
    
    print(f"\n⏱️  Timeline:")
    print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"   End: {end_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Expected readings: {total_readings:,}")
    
    # Generate readings
    print(f"\n🔄 Generating data...")
    readings = []
    current_time = start_time
    reading_count = 0
    
    # Decide when to trigger events
    event_times = []
    if include_events:
        # 2-3 events spread across the period
        num_events = np.random.randint(2, 4)
        event_types = ['hypertensive_crisis', 'hypoxia', 'sepsis']
        
        for _ in range(num_events):
            # Events in middle 60% of timeline (not at start or end)
            event_offset = np.random.uniform(0.2, 0.8)
            event_time = start_time + timedelta(days=days * event_offset)
            event_type = np.random.choice(event_types)
            event_times.append((event_time, event_type))
        
        event_times.sort()
        print(f"\n⚠️  Planned deterioration events: {len(event_times)}")
        for i, (evt_time, evt_type) in enumerate(event_times, 1):
            print(f"   Event {i}: {evt_type} at {evt_time.strftime('%Y-%m-%d %H:%M')}")
    
    current_event_idx = 0
    active_event = False
    event_start_time = None
    
    # Progress tracking
    next_progress = 0.1
    
    while current_time <= end_time:
        # Check if we should trigger an event
        if (include_events and 
            current_event_idx < len(event_times) and 
            current_time >= event_times[current_event_idx][0] and
            not active_event):
            
            event_type = event_times[current_event_idx][1]
            device.trigger_deterioration(event_type)
            active_event = True
            event_start_time = current_time
            print(f"\n   ⚠️  Event triggered: {event_type}")
        
        # Stop event after 6-12 hours
        if active_event and event_start_time:
            event_duration_hours = (current_time - event_start_time).total_seconds() / 3600
            if event_duration_hours > np.random.uniform(6, 12):
                device.stop_deterioration()
                active_event = False
                event_start_time = None
                current_event_idx += 1
                print(f"   ✅ Event resolved")
        
        # Generate reading
        reading = device.generate_reading(current_time)
        readings.append(reading)
        
        # Progress indicator
        reading_count += 1
        progress = reading_count / total_readings
        if progress >= next_progress:
            print(f"   Progress: {progress*100:.0f}% ({reading_count:,}/{total_readings:,} readings)")
            next_progress += 0.1
        
        # Next timestamp
        current_time += timedelta(minutes=sampling_minutes)
    
    # Create DataFrame
    print(f"\n📊 Creating DataFrame...")
    df = pd.DataFrame(readings)
    
    # Add some realistic missing data patterns (10-15% missing)
    print(f"\n🔍 Adding realistic missing data patterns...")
    for col in ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2']:
        if col in df.columns:
            # Create chunks of missing data (device not worn)
            n_gaps = int(len(df) * 0.02)  # 2% of time in gaps
            for _ in range(n_gaps):
                gap_start = np.random.randint(0, len(df) - 20)
                gap_length = np.random.randint(5, 30)  # 25-150 minutes
                df.loc[gap_start:gap_start+gap_length, col] = np.nan
    
    missing_pct = df.isnull().sum() / len(df) * 100
    print(f"\n   Missing data by feature:")
    for col in df.columns:
        if col not in ['patient_id', 'timestamp', 'device_id', 'activity_state']:
            print(f"   {col}: {missing_pct[col]:.1f}%")
    
    # Save to CSV
    print(f"\n💾 Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("✅ WEARABLE DATA GENERATION COMPLETE")
    print("="*60)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total readings: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Duration: {(pd.to_datetime(df['timestamp'].max()) - pd.to_datetime(df['timestamp'].min())).days} days")
    print(f"\n📈 Vital Signs Statistics:")
    stats_cols = ['heart_rate', 'bp_systolic', 'spo2', 'temperature']
    print(df[stats_cols].describe().round(2).to_string())
    
    return df


def generate_complete_dataset(
    patient_id: str = "patient_001",
    days: int = 30,
    sampling_minutes: int = 5,
    include_events: bool = True
):
    """
    Generate complete dataset: wearables + labs + integrated features
    
    This is the ONE-CLICK solution for getting ML-ready data
    """
    print("\n" + "="*80)
    print(" "*25 + "COMPLETE DATA GENERATION")
    print("="*80)
    
    # Step 1: Generate wearable data
    wearable_df = generate_wearable_data(
        patient_id=patient_id,
        days=days,
        sampling_minutes=sampling_minutes,
        include_events=include_events,
        output_path="data/processed/wearable_data.csv"
    )
    
    # Step 2: Generate lab data
    print("\n" + "-"*80)
    print("STEP 2/3: Generating Lab Data")
    print("-"*80)
    
    lab_generator = LabDataGenerator()
    start_date = pd.to_datetime(wearable_df['timestamp'].min())
    end_date = pd.to_datetime(wearable_df['timestamp'].max())
    
    lab_df = lab_generator.generate_for_timerange(
        start_date=start_date,
        end_date=end_date,
        test_frequency_days=30,
        patient_id=patient_id
    )
    
    lab_output = "data/processed/lab_results.csv"
    lab_df.to_csv(lab_output, index=False)
    print(f"✅ Saved: {lab_output} ({len(lab_df)} lab tests)")
    
    # Step 3: Integrate
    print("\n" + "-"*80)
    print("STEP 3/3: Running Integration Pipeline")
    print("-"*80)
    
    from data.data_integration_pipeline import DataIntegrationPipeline
    
    pipeline = DataIntegrationPipeline()
    features_df = pipeline.run_pipeline(
        wearable_path="data/processed/wearable_data.csv",
        lab_path="data/processed/lab_results.csv",
        output_path="data/processed/features.csv"
    )
    
    # Final summary
    print("\n" + "="*80)
    print(" "*30 + "🎉 ALL DONE! 🎉")
    print("="*80)
    print("\n  Generated Files:")
    print(f"   1. data/processed/wearable_data.csv ({len(wearable_df):,} readings)")
    print(f"   2. data/processed/lab_results.csv ({len(lab_df)} tests)")
    print(f"   3. data/processed/features.csv ({len(features_df):,} samples) 🎯")
    
    print("\n  ML-ready dataset:")
    print(f"   Samples: {len(features_df):,}")
    print(f"   Features: {len(features_df.columns) - 2}")
    print(f"   Deterioration events: {(features_df['label'] == 1).sum()} ({(features_df['label'] == 1).sum() / len(features_df) * 100:.2f}%)")
    
    print("\n  Next Steps:")
    print("   1. Explore data: jupyter notebook")
    print("   2. Train model: python src/models/train.py")
    print("   3. Evaluate: python src/evaluation/evaluate_model.py")
    
    print("      Points:")
    print("   ✓ Generated realistic health data with temporal patterns")
    print("   ✓ Integrated multi-modal sources (wearables + labs)")
    print("   ✓ Applied signal processing (Kalman filtering)")
    print("   ✓ Handled class imbalance (~2% positive cases)")
    print("   ✓ Created production-ready ML pipeline")
    
    return features_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate realistic health monitoring data')
    parser.add_argument('--days', type=int, default=30, help='Days of data to generate')
    parser.add_argument('--sampling', type=int, default=5, help='Minutes between readings')
    parser.add_argument('--no-events', action='store_true', help='Disable deterioration events')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 7 days only')
    
    args = parser.parse_args()
    
    if args.quick:
        print("\n⚡ QUICK MODE: Generating 7 days of data")
        days = 7
    else:
        days = args.days
    
    # Generate complete dataset
    features_df = generate_complete_dataset(
        patient_id="patient_001",
        days=days,
        sampling_minutes=args.sampling,
        include_events=not args.no_events
    )
    
    print("\n✅ Generation complete! Check data/processed/ folder")