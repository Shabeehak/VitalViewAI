# src/data/wearable_simulator.py
"""
Wearable Device Simulator with VARIED Patient Profiles
Generates realistic physiological data with DISTINCT baselines per patient
"""

import numpy as np
from datetime import datetime
from typing import Dict, Optional

class WearableDevice:
    """
    Simulates a wearable health monitoring device with VARIED patient profiles
    
    Features:
    - DISTINCT baseline vitals per patient based on health profile
    - Realistic circadian rhythms
    - Activity-based variations
    - Temporal correlations
    - Sensor noise
    - Deterioration event simulation
    """
    
    # Define distinct health profiles with DIFFERENT baseline ranges
    HEALTH_PROFILES = {
        'healthy': {
            'heart_rate': (60, 72),
            'bp_systolic': (110, 120),
            'bp_diastolic': (70, 78),
            'spo2': (97, 99),
            'respiratory_rate': (12, 15),
            'temperature': (36.4, 36.7),
            'variability': 0.02  # Low variability
        },
        'at_risk': {
            'heart_rate': (75, 88),
            'bp_systolic': (128, 138),
            'bp_diastolic': (80, 88),
            'spo2': (94, 96),
            'respiratory_rate': (16, 19),
            'temperature': (36.6, 37.0),
            'variability': 0.05  # Moderate variability
        },
        'deteriorating': {
            'heart_rate': (90, 105),
            'bp_systolic': (145, 165),
            'bp_diastolic': (90, 100),
            'spo2': (91, 94),
            'respiratory_rate': (20, 24),
            'temperature': (37.0, 37.5),
            'variability': 0.08  # High variability
        }
    }
    
    def __init__(
        self, 
        patient_id: str = "patient_001", 
        seed: int = None,
        health_profile: str = None  # NEW: Accept health profile
    ):
        """
        Initialize wearable device with specific health profile
        
        Args:
            patient_id: Unique patient identifier
            seed: Random seed for reproducibility
            health_profile: 'healthy', 'at_risk', or 'deteriorating'
        """
        if seed:
            np.random.seed(seed)
        else:
            # Use patient_id as seed for consistent but varied baselines
            np.random.seed(hash(patient_id) % 2**32)
        
        self.patient_id = patient_id
        
        # Determine health profile from patient_id if not specified
        if health_profile is None:
            if 'stable' in patient_id.lower():
                health_profile = 'healthy'
            elif 'warning' in patient_id.lower():
                health_profile = 'at_risk'
            elif 'critical' in patient_id.lower():
                health_profile = 'deteriorating'
            else:
                # Random assignment
                health_profile = np.random.choice(['healthy', 'at_risk', 'deteriorating'])
        
        self.health_profile = health_profile
        profile_ranges = self.HEALTH_PROFILES[health_profile]
        
        # Generate UNIQUE baseline for this patient within profile range
        self.baseline = {
            'heart_rate': np.random.uniform(*profile_ranges['heart_rate']),
            'bp_systolic': np.random.uniform(*profile_ranges['bp_systolic']),
            'bp_diastolic': np.random.uniform(*profile_ranges['bp_diastolic']),
            'spo2': np.random.uniform(*profile_ranges['spo2']),
            'respiratory_rate': np.random.uniform(*profile_ranges['respiratory_rate']),
            'temperature': np.random.uniform(*profile_ranges['temperature']),
        }
        
        self.baseline_variability = profile_ranges['variability']
        
        # Track last values for temporal correlation
        self.last_values = self.baseline.copy()
        
        # Deterioration state
        self.is_deteriorating = False
        self.deterioration_type = None
        self.deterioration_severity = 0.0
        
        print(f"🏥 Created {health_profile.upper()} patient: {patient_id}")
        print(f"   Baseline HR: {self.baseline['heart_rate']:.1f} bpm")
        print(f"   Baseline BP: {self.baseline['bp_systolic']:.1f}/{self.baseline['bp_diastolic']:.1f} mmHg")
        print(f"   Baseline SpO2: {self.baseline['spo2']:.1f}%")
    
    def _get_circadian_factor(self, hour: int) -> float:
        """
        Calculate circadian rhythm adjustment
        
        Physiology:
        - Peak: 4 PM (hour 16)
        - Lowest: 4 AM (hour 4)
        - ~10-15% variation from baseline
        """
        phase = (hour - 16) * 2 * np.pi / 24
        return 0.12 * np.cos(phase)
    
    def _get_activity_level(self, hour: int) -> str:
        """
        Determine activity level based on time of day
        
        Realistic daily pattern:
        - 11 PM - 7 AM: Sleeping
        - 7 AM - 9 AM: Morning routine (light)
        - 9 AM - 5 PM: Work/daily activities
        - 6 PM - 7 PM: Exercise (30% chance)
        - Evening: Resting
        """
        if 23 <= hour or hour < 7:
            return 'sleeping'
        elif 7 <= hour < 9:
            return 'light_activity'
        elif 18 <= hour < 19 and np.random.random() < 0.3:
            return 'exercising'
        elif 9 <= hour < 17 and np.random.random() < 0.25:
            return 'walking'
        else:
            return 'resting'
    
    def _get_activity_multipliers(self, activity: str) -> Dict[str, float]:
        """Heart rate and respiration increase with activity"""
        multipliers = {
            'sleeping': {'heart_rate': 0.80, 'respiratory_rate': 0.75},
            'resting': {'heart_rate': 1.0, 'respiratory_rate': 1.0},
            'light_activity': {'heart_rate': 1.1, 'respiratory_rate': 1.05},
            'walking': {'heart_rate': 1.25, 'respiratory_rate': 1.2},
            'exercising': {'heart_rate': 1.6, 'respiratory_rate': 1.5},
        }
        return multipliers.get(activity, multipliers['resting'])
    
    def _apply_deterioration(self, vitals: Dict[str, float]) -> Dict[str, float]:
        """
        Modify vitals during health deterioration event
        
        Deterioration types:
        - hypertensive_crisis: High BP, elevated HR
        - hypoxia: Low O2, compensatory high HR
        - sepsis: Fever, tachycardia, low BP
        - cardiac: Irregular HR, hypotension
        """
        if not self.is_deteriorating:
            return vitals
        
        severity = self.deterioration_severity
        
        if self.deterioration_type == 'hypertensive_crisis':
            vitals['bp_systolic'] *= (1 + 0.35 * severity)
            vitals['bp_diastolic'] *= (1 + 0.25 * severity)
            vitals['heart_rate'] *= (1 + 0.20 * severity)
            
        elif self.deterioration_type == 'hypoxia':
            vitals['spo2'] = max(vitals['spo2'] * (1 - 0.15 * severity), 85)
            vitals['heart_rate'] *= (1 + 0.30 * severity)
            vitals['respiratory_rate'] *= (1 + 0.35 * severity)
            
        elif self.deterioration_type == 'sepsis':
            vitals['temperature'] += 2.5 * severity
            vitals['heart_rate'] *= (1 + 0.40 * severity)
            vitals['bp_systolic'] *= (1 - 0.20 * severity)
            vitals['respiratory_rate'] *= (1 + 0.30 * severity)
            
        elif self.deterioration_type == 'cardiac':
            vitals['heart_rate'] *= (1 + 0.50 * severity)
            vitals['bp_systolic'] *= (1 - 0.25 * severity)
            vitals['bp_diastolic'] *= (1 - 0.15 * severity)
        
        return vitals
    
    def generate_reading(self, timestamp: Optional[datetime] = None) -> Dict:
        """
        Generate a single realistic sensor reading with VARIED values
        
        Process:
        1. Determine time-based factors (circadian, activity)
        2. Apply temporal correlation (smooth changes)
        3. Add sensor noise AND baseline variability
        4. Apply deterioration if active
        5. Clip to physiological ranges
        
        Returns:
            Dictionary with timestamp and all vital signs
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        hour = timestamp.hour
        
        # Get modifying factors
        circadian_factor = self._get_circadian_factor(hour)
        activity = self._get_activity_level(hour)
        activity_mults = self._get_activity_multipliers(activity)
        
        # Generate new vitals
        vitals = {}
        
        for vital, baseline_value in self.baseline.items():
            # Start with last value (temporal correlation)
            prev_value = self.last_values[vital]
            
            # Calculate target value
            target = baseline_value
            
            # Apply circadian rhythm (except temperature which has different pattern)
            if vital in ['heart_rate', 'bp_systolic', 'bp_diastolic']:
                target *= (1 + circadian_factor)
            
            # Apply activity
            if vital in activity_mults:
                target *= activity_mults[vital]
            
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
            elif vital == 'bp_systolic' or vital == 'bp_diastolic':
                noise_level = 0.03  # BP readings are noisier
            
            noise = np.random.normal(0, new_value * noise_level)
            new_value += noise
            
            vitals[vital] = new_value
        
        # Apply deterioration
        vitals = self._apply_deterioration(vitals)
        
        # Clip to physiologically plausible ranges
        vitals['heart_rate'] = np.clip(vitals['heart_rate'], 40, 200)
        vitals['bp_systolic'] = np.clip(vitals['bp_systolic'], 80, 220)
        vitals['bp_diastolic'] = np.clip(vitals['bp_diastolic'], 50, 130)
        vitals['spo2'] = np.clip(vitals['spo2'], 85, 100)
        vitals['respiratory_rate'] = np.clip(vitals['respiratory_rate'], 8, 40)
        vitals['temperature'] = np.clip(vitals['temperature'], 35.0, 41.0)
        
        # Store for next reading
        self.last_values = vitals.copy()
        
        # Format reading
        reading = {
            'timestamp': timestamp.isoformat(),
            'patient_id': self.patient_id,
            'device_id': f'wearable_{self.patient_id}',
            'activity_state': activity,
            'heart_rate': round(vitals['heart_rate'], 1),
            'bp_systolic': round(vitals['bp_systolic'], 1),
            'bp_diastolic': round(vitals['bp_diastolic'], 1),
            'spo2': round(vitals['spo2'], 1),
            'respiratory_rate': round(vitals['respiratory_rate'], 1),
            'temperature': round(vitals['temperature'], 1),
        }
        
        return reading
    
    def trigger_deterioration(self, event_type: str = 'hypertensive_crisis'):
        """
        Trigger a health deterioration event
        
        Args:
            event_type: Type of event (hypertensive_crisis, hypoxia, sepsis, cardiac)
        """
        self.is_deteriorating = True
        self.deterioration_type = event_type
        self.deterioration_severity = 0.0
        print(f"⚠️  Deterioration triggered: {event_type} for {self.patient_id}")
    
    def update_deterioration_severity(self, hours_elapsed: float):
        """
        Update deterioration severity based on time elapsed
        
        Severity increases gradually over 6 hours to full severity
        """
        if self.is_deteriorating:
            self.deterioration_severity = min(hours_elapsed / 6.0, 1.0)
    
    def resolve_deterioration(self):
        """Stop deterioration (patient recovered/treated)"""
        self.is_deteriorating = False
        self.deterioration_type = None
        self.deterioration_severity = 0.0
        print(f"✅ Deterioration resolved for {self.patient_id}")


# Quick test
if __name__ == "__main__":
    print("Testing Wearable Device Simulator with VARIED profiles...\n")
    
    # Create 3 different patients
    patients = [
        WearableDevice(patient_id="patient_stable_001", health_profile='healthy'),
        WearableDevice(patient_id="patient_warning_002", health_profile='at_risk'),
        WearableDevice(patient_id="patient_critical_003", health_profile='deteriorating')
    ]
    
    print("\n" + "="*70)
    print("COMPARING BASELINE VITALS ACROSS PROFILES")
    print("="*70)
    
    for device in patients:
        print(f"\n{device.health_profile.upper()} - {device.patient_id}")
        print(f"  HR: {device.baseline['heart_rate']:.1f} bpm")
        print(f"  BP: {device.baseline['bp_systolic']:.1f}/{device.baseline['bp_diastolic']:.1f} mmHg")
        print(f"  SpO2: {device.baseline['spo2']:.1f}%")
        print(f"  RR: {device.baseline['respiratory_rate']:.1f} /min")
        print(f"  Temp: {device.baseline['temperature']:.2f}°C")
    
    print("\n" + "="*70)
    print("GENERATING SAMPLE READINGS")
    print("="*70)
    
    for device in patients:
        reading = device.generate_reading()
        print(f"\n{device.patient_id}:")
        print(f"  HR: {reading['heart_rate']} bpm, BP: {reading['bp_systolic']}/{reading['bp_diastolic']} mmHg")
    
    print("\n✅ Profiles are DIFFERENT - predictions should vary!")

