#src/data/realtime_health_simulator.py
"""
Real-Time Health Data Simulator
Simulates a wearable device streaming health data in real-time
- Real wearables stream data every few seconds/minutes
- Need to simulate realistic physiological patterns
- Include circadian rhythms, activity responses, noise
- Can inject deterioration events for testing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from typing import Dict, Generator, Optional
import threading
import queue

class WearableDeviceSimulator:
    """
    Simulates a real wearable device (like Apple Watch, Fitbit)
    """
    HEALTH_PROFILES = {
        'healthy': {
            'heart_rate': (60, 72),
            'bp_systolic': (110, 120),
            'bp_diastolic': (70, 78),
            'spo2': (97, 99),
            'respiratory_rate': (12, 15),
            'temperature': (36.4, 36.7),
            'variability': 0.02
        },
        'at_risk': {
            'heart_rate': (75, 88),
            'bp_systolic': (128, 138),
            'bp_diastolic': (80, 88),
            'spo2': (94, 96),
            'respiratory_rate': (16, 19),
            'temperature': (36.6, 37.0),
            'variability': 0.05
        },
        'deteriorating': {
            'heart_rate': (90, 105),
            'bp_systolic': (145, 165),
            'bp_diastolic': (90, 100),
            'spo2': (91, 94),
            'respiratory_rate': (20, 24),
            'temperature': (37.0, 37.5),
            'variability': 0.08
        }
    }
    def __init__(
        self, 
        patient_id: str = "patient_001",
        sampling_rate_seconds: int = 300,  # 5 minutes
        seed: int = 42,
        health_profile: str = None
        ):
        """
        Args:
            patient_id: Unique patient identifier
            sampling_rate_seconds: How often device reports (300s = 5min)
            seed: Random seed for reproducibility
        """
        self.patient_id = patient_id
        self.sampling_rate = sampling_rate_seconds
        # Use patient_id as seed for consistency
        np.random.seed(hash(patient_id) % 2**32)

        # Determine health profile from patient_id
        if health_profile is None:
            if 'stable' in patient_id.lower() or 'healthy' in patient_id.lower():
                health_profile = 'healthy'
            elif 'warning' in patient_id.lower() or 'risk' in patient_id.lower():
                health_profile = 'at_risk'
            elif 'critical' in patient_id.lower() or 'deteriorat' in patient_id.lower():
                health_profile = 'deteriorating'
            else:
                # Random assignment
                health_profile = np.random.choice(['healthy', 'at_risk', 'deteriorating'])
        
        self.health_profile = health_profile
        profile_ranges = self.HEALTH_PROFILES[health_profile]
        
        # Generate UNIQUE baseline within profile range
        self.baseline = {
            'heart_rate': np.random.uniform(*profile_ranges['heart_rate']),
            'bp_systolic': np.random.uniform(*profile_ranges['bp_systolic']),
            'bp_diastolic': np.random.uniform(*profile_ranges['bp_diastolic']),
            'spo2': np.random.uniform(*profile_ranges['spo2']),
            'respiratory_rate': np.random.uniform(*profile_ranges['respiratory_rate']),
            'temperature': np.random.uniform(*profile_ranges['temperature']),
        }
        
        self.baseline_variability = profile_ranges['variability']
        
        print(f"🏥 Created {health_profile.upper()} patient: {patient_id}")
        print(f"   Baseline HR: {self.baseline['heart_rate']:.1f} bpm")
        print(f"   Baseline BP: {self.baseline['bp_systolic']:.1f}/{self.baseline['bp_diastolic']:.1f}")
        
        
        # Activity state affects vitals
        self.activity_state = 'resting'  # resting, walking, exercising, sleeping
        self.activity_multipliers = {
            'sleeping': {'heart_rate': 0.85, 'respiratory_rate': 0.8},
            'resting': {'heart_rate': 1.0, 'respiratory_rate': 1.0},
            'walking': {'heart_rate': 1.2, 'respiratory_rate': 1.15},
            'exercising': {'heart_rate': 1.6, 'respiratory_rate': 1.4},
        }
        
        # Deterioration event simulation
        self.is_deteriorating = False
        self.deterioration_type = None
        self.deterioration_start = None
        
        # Internal state for temporal correlation
        self.last_values = self.baseline.copy()
        
    def get_time_of_day_factor(self, current_time: datetime) -> Dict[str, float]:
        """
        Calculate circadian rhythm adjustments
        
        Real physiology:
        - Heart rate peaks in afternoon (3-6 PM)
        - Lowest in early morning (3-6 AM)
        - Blood pressure follows similar pattern
        - Temperature peaks evening, lowest morning
        """
        hour = current_time.hour
        
        # Cosine wave with peak at 4 PM (hour 16)
        circadian_phase = (hour - 16) * 2 * np.pi / 24
        circadian_factor = np.cos(circadian_phase)
        
        # Different amplitudes for different vitals
        adjustments = {
            'heart_rate': 0.10 * circadian_factor,      # ±10% variation
            'bp_systolic': 0.08 * circadian_factor,     # ±8% variation
            'bp_diastolic': 0.06 * circadian_factor,    # ±6% variation
            'temperature': 0.015 * circadian_factor,    # ±0.5°C variation
        }
        
        return adjustments
    
    def detect_activity_from_time(self, current_time: datetime) -> str:
        """
        Simulate activity based on time of day
        
        Realistic daily pattern:
        - 11 PM - 7 AM: Sleeping
        - 7 AM - 8 AM: Light activity (morning routine)
        - 9 AM - 5 PM: Mixed (sitting, walking)
        - 6 PM - 7 PM: Exercise time (30% chance)
        - Evening: Resting
        """
        hour = current_time.hour
        
        if 23 <= hour or hour < 7:
            return 'sleeping'
        elif 7 <= hour < 9:
            return 'walking'
        elif 18 <= hour < 19 and np.random.random() < 0.3:
            return 'exercising'
        elif 9 <= hour < 17 and np.random.random() < 0.2:
            return 'walking'
        else:
            return 'resting'
    
    def apply_deterioration(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """
        Modify vitals during health deterioration event
        
        Deterioration types:
        1. Hypertensive crisis: BP spikes, HR increases
        2. Hypoxia: SpO2 drops, HR increases (compensatory)
        3. Sepsis: Fever, high HR, low BP
        4. Cardiac: Irregular HR, low BP
        """
        if not self.is_deteriorating:
            return vitals
        
        # Calculate time since deterioration started (gradual onset)
        time_since_start = (datetime.now() - self.deterioration_start).total_seconds() / 3600
        severity = min(time_since_start / 6, 1.0)  # Full severity after 6 hours
        
        if self.deterioration_type == 'hypertensive_crisis':
            vitals['bp_systolic'] *= (1 + 0.3 * severity)
            vitals['bp_diastolic'] *= (1 + 0.2 * severity)
            vitals['heart_rate'] *= (1 + 0.15 * severity)
            
        elif self.deterioration_type == 'hypoxia':
            vitals['spo2'] = max(vitals['spo2'] * (1 - 0.15 * severity), 85)
            vitals['heart_rate'] *= (1 + 0.25 * severity)
            vitals['respiratory_rate'] *= (1 + 0.3 * severity)
            
        elif self.deterioration_type == 'sepsis':
            vitals['temperature'] += 2 * severity
            vitals['heart_rate'] *= (1 + 0.4 * severity)
            vitals['bp_systolic'] *= (1 - 0.15 * severity)
            vitals['respiratory_rate'] *= (1 + 0.25 * severity)
            
        elif self.deterioration_type == 'cardiac':
            vitals['heart_rate'] *= (1 + 0.5 * severity)
            vitals['bp_systolic'] *= (1 - 0.2 * severity)
            
        return vitals
    
    def generate_reading(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Generate single realistic sensor reading
        
        Process:
        1. Start with baseline
        2. Apply circadian rhythm
        3. Apply activity state
        4. Add random walk (temporal correlation)
        5. Add sensor noise
        6. Apply deterioration if active
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Determine activity state
        self.activity_state = self.detect_activity_from_time(current_time)
        
        # Get modifying factors
        circadian = self.get_time_of_day_factor(current_time)
        multipliers = self.activity_multipliers[self.activity_state]
        
        # Generate new vitals
        vitals = {}
        
        for vital in self.baseline.keys():
            # Get baseline and previous value
            baseline_value = self.baseline[vital]
            prev_value = self.last_values[vital]
            
            # Start with baseline
            target = baseline_value
            
            # Apply circadian adjustments
            if vital in circadian:
                target = baseline_value * (1 + circadian[vital])
            
            # Apply activity multipliers
            if vital in multipliers:
                target = baseline_value * multipliers[vital]
            
            # CRITICAL: Add random walk to create VARIATION between patients
            # This ensures patients don't converge to same values
            random_walk = np.random.normal(0, baseline_value * self.baseline_variability)
            target += random_walk
            
            # Temporal correlation: 85% previous, 15% new target
            # This creates smooth transitions
            new_value = 0.85 * prev_value + 0.15 * target
            
            # Add sensor noise (realistic measurement error)
            noise_level = 0.02  # 2% noise
            if vital == 'spo2':
                noise_level = 0.005  # SpO2 sensors are more stable
            elif vital in ['bp_systolic', 'bp_diastolic']:
                noise_level = 0.03  # BP readings are noisier
            
            noise = np.random.normal(0, new_value * noise_level)
            new_value += noise
            
            vitals[vital] = new_value
        
        # Apply deterioration if active
        vitals = self.apply_deterioration(vitals)
        
        # Clip to physiologically plausible ranges
        vitals['heart_rate'] = np.clip(vitals['heart_rate'], 40, 200)
        vitals['bp_systolic'] = np.clip(vitals['bp_systolic'], 80, 220)
        vitals['bp_diastolic'] = np.clip(vitals['bp_diastolic'], 50, 130)
        vitals['spo2'] = np.clip(vitals['spo2'], 85, 100)
        vitals['respiratory_rate'] = np.clip(vitals['respiratory_rate'], 8, 35)
        vitals['temperature'] = np.clip(vitals['temperature'], 35, 41)
        
        # Store for next reading
        self.last_values = vitals.copy()
        
        # Create reading with metadata
        reading = {
            'patient_id': self.patient_id,
            'timestamp': current_time.isoformat(),
            'device_id': f'wearable_{self.patient_id}',
            'activity_state': self.activity_state,
            **{k: round(v, 2) for k, v in vitals.items()}
        }
        
        return reading
    
    def trigger_deterioration(self, deterioration_type: str = 'hypertensive_crisis'):
        """
        Trigger a health deterioration event
        
        Use this to test alert systems and model predictions
        """
        self.is_deteriorating = True
        self.deterioration_type = deterioration_type
        self.deterioration_start = datetime.now()
        print(f"⚠️  Deterioration event triggered: {deterioration_type}")
    
    def stop_deterioration(self):
        """Stop deterioration (patient treated)"""
        self.is_deteriorating = False
        self.deterioration_type = None
        print("✅ Deterioration resolved")


class RealTimeDataStream:
    """
    Streams health data in real-time like a real wearable device
    """
    
    def __init__(self, patient_id: str = "patient_001", sampling_rate: int = 300):
        self.device = WearableDeviceSimulator(patient_id, sampling_rate)
        self.is_streaming = False
        self.data_queue = queue.Queue()
        
    def stream_generator(self) -> Generator[Dict, None, None]:
        """
        Generator that yields readings at regular intervals
        
        This is how real streaming APIs work - continuous data generation
        """
        print(f" Starting real-time stream (1 reading every {self.device.sampling_rate}s)")
        
        while self.is_streaming:
            reading = self.device.generate_reading()
            yield reading
            time.sleep(self.device.sampling_rate)
    
    def start_streaming(self, duration_minutes: Optional[int] = None):
        """
        Start streaming data (non-blocking)
        
        Args:
            duration_minutes: How long to stream (None = infinite)
        """
        self.is_streaming = True
        
        def stream_worker():
            start_time = datetime.now()
            
            while self.is_streaming:
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        self.is_streaming = False
                        break
                
                reading = self.device.generate_reading()
                self.data_queue.put(reading)
                
                # Print to console (simulating real-time monitoring)
                hr = reading['heart_rate']
                bp = f"{reading['bp_systolic']}/{reading['bp_diastolic']}"
                spo2 = reading['spo2']
                print(f"[{reading['timestamp']}] HR:{hr:.0f} BP:{bp} SpO2:{spo2:.1f}%")
                
                time.sleep(self.device.sampling_rate)
        
        # Start in background thread
        thread = threading.Thread(target=stream_worker, daemon=True)
        thread.start()
        print("✅ Streaming started (background thread)")
    
    def stop_streaming(self):
        """Stop the stream"""
        self.is_streaming = False
        print("🛑 Streaming stopped")
    
    def get_collected_data(self) -> pd.DataFrame:
        """Get all collected data as DataFrame"""
        data = []
        while not self.data_queue.empty():
            data.append(self.data_queue.get())
        return pd.DataFrame(data)


# ==================== USAGE EXAMPLES ====================

def demo_single_reading():
    """Demo: Generate single reading"""
    print("\n" + "="*60)
    print("DEMO 1: Single Reading")
    print("="*60)
    
    device = WearableDeviceSimulator(patient_id="demo_patient")
    reading = device.generate_reading()
    
    print("\n📊 Sample Reading:")
    print(json.dumps(reading, indent=2))


def demo_24hour_simulation():
    """Demo: Simulate 24 hours of data (accelerated)"""
    print("\n" + "="*60)
    print("DEMO 2: 24-Hour Simulation")
    print("="*60)
    
    device = WearableDeviceSimulator(patient_id="demo_patient")
    
    # Generate 24 hours of data (1 reading per hour)
    start_time = datetime.now() - timedelta(days=1)
    readings = []
    
    for hour in range(24):
        current_time = start_time + timedelta(hours=hour)
        reading = device.generate_reading(current_time)
        readings.append(reading)
    
    df = pd.DataFrame(readings)
    
    print(f"\n✅ Generated {len(df)} readings over 24 hours")
    print("\n📊 Statistics:")
    print(df[['heart_rate', 'bp_systolic', 'spo2', 'temperature']].describe())
    
    # Save to CSV
    output_path = 'data/processed/simulated_realtime_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    return df


def demo_deterioration_event():
    """Demo: Simulate deterioration event"""
    print("\n" + "="*60)
    print("DEMO 3: Deterioration Event")
    print("="*60)
    
    device = WearableDeviceSimulator(patient_id="at_risk_patient")
    
    readings = []
    
    # Normal readings for 2 hours
    print("\n📊 Normal readings (2 hours)...")
    current_time = datetime.now()
    for i in range(24):  # 24 readings = 2 hours at 5min intervals
        reading = device.generate_reading(current_time)
        readings.append(reading)
        current_time += timedelta(minutes=5)
    
    # Trigger deterioration
    print("\n⚠️  TRIGGERING DETERIORATION EVENT")
    device.trigger_deterioration('hypertensive_crisis')
    
    # Deteriorating readings for 2 hours
    print("\n📊 Deteriorating readings (2 hours)...")
    for i in range(24):
        reading = device.generate_reading(current_time)
        readings.append(reading)
        current_time += timedelta(minutes=5)
        
        if i == 0:  # Show first deteriorating reading
            print(f"   First deteriorating reading:")
            print(f"   BP: {reading['bp_systolic']}/{reading['bp_diastolic']}")
            print(f"   HR: {reading['heart_rate']}")
    
    df = pd.DataFrame(readings)
    df['is_deteriorating'] = ['no']*24 + ['yes']*24
    
    print(f"\n✅ Generated deterioration event simulation")
    print("\n📊 Comparison:")
    print(df.groupby('is_deteriorating')[['heart_rate', 'bp_systolic']].mean())
    
    return df


def demo_realtime_streaming():
    """Demo: Real-time streaming (runs for 2 minutes)"""
    print("\n" + "="*60)
    print("DEMO 4: Real-Time Streaming")
    print("="*60)
    print("\n⏱️  Will stream for 2 minutes (press Ctrl+C to stop early)")
    
    stream = RealTimeDataStream(patient_id="streaming_patient", sampling_rate=10)  # 10s for demo
    
    stream.start_streaming(duration_minutes=2)
    
    try:
        # Keep main thread alive
        time.sleep(120)
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    stream.stop_streaming()
    
    # Get collected data
    df = stream.get_collected_data()
    print(f"\n✅ Collected {len(df)} readings")
    
    return df


if __name__ == "__main__":
    print("\n REAL-TIME HEALTH DATA SIMULATOR")
    print("="*60)
    print("\nChoose a demo:")
    print("1. Single reading")
    print("2. 24-hour simulation")
    print("3. Deterioration event")
    print("4. Real-time streaming (2 min)")
    print("5. Run all demos")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        demo_single_reading()
    elif choice == '2':
        demo_24hour_simulation()
    elif choice == '3':
        demo_deterioration_event()
    elif choice == '4':
        demo_realtime_streaming()
    elif choice == '5':
        demo_single_reading()
        demo_24hour_simulation()
        demo_deterioration_event()
    else:
        print("Invalid choice")