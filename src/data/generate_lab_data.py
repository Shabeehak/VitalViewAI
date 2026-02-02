"""
Lab Test Data Generator - FIXED VERSION
Generates realistic lab test results with patient-specific variation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

class LabDataGenerator:
    """
    Generates realistic lab test results with PATIENT-SPECIFIC variation
    """
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: Base random seed (will be modified per patient)
        """
        self.base_seed = seed if seed is not None else 42
        
        # Normal lab value ranges (for healthy adults)
        self.normal_ranges = {
            'glucose_fasting': (70, 100),
            'glucose_random': (70, 140),
            'creatinine': (0.7, 1.3),
            'hemoglobin': (12, 16),
            'wbc': (4, 11),
            'platelets': (150, 400),
            'cholesterol_total': (125, 200),
            'cholesterol_ldl': (0, 100),
            'cholesterol_hdl': (40, 60),
            'triglycerides': (0, 150),
        }
    
    def generate_for_timerange(
        self, 
        start_date: datetime,
        end_date: datetime,
        test_frequency_days: int = 30,
        patient_id: str = 'patient_001'
    ) -> pd.DataFrame:
        """
        Generate lab tests for a date range with PATIENT-SPECIFIC values
        
        Args:
            start_date: Start of monitoring period
            end_date: End of monitoring period
            test_frequency_days: How often labs are done
            patient_id: Patient identifier (used to create unique seed)
        """
        # FIXED: Create patient-specific seed from patient_id
        patient_seed = self.base_seed + hash(patient_id) % 10000
        np.random.seed(patient_seed)
        
        # Calculate number of tests
        total_days = (end_date - start_date).days
        n_tests = max(1, total_days // test_frequency_days)
        
        # Generate test dates
        test_dates = []
        current_date = start_date
        for i in range(n_tests):
            # Add random jitter (±3 days)
            jitter = timedelta(days=np.random.randint(-3, 4))
            test_date = current_date + jitter
            test_dates.append(test_date)
            current_date += timedelta(days=test_frequency_days)
        
        # Generate PATIENT-SPECIFIC baseline values
        baseline = {}
        for test_name, (low, high) in self.normal_ranges.items():
            # Each patient gets their own unique baseline
            baseline[test_name] = np.random.uniform(low, high)
        
        # Generate lab results with temporal correlation
        lab_results = []
        
        for test_date in test_dates:
            result = {
                'patient_id': patient_id,
                'test_date': test_date,
                'test_type': 'comprehensive_metabolic_panel'
            }
            
            # For each lab value, add small random walk from baseline
            for test_name, baseline_value in baseline.items():
                # Random walk: ±5% variation from baseline
                variation = np.random.normal(0, baseline_value * 0.05)
                value = baseline_value + variation
                
                # Clip to physiologically plausible range
                low, high = self.normal_ranges[test_name]
                value = np.clip(value, low * 0.8, high * 1.2)
                
                result[test_name] = round(value, 2)
            
            lab_results.append(result)
        
        df = pd.DataFrame(lab_results)
        return df
    
    def add_deterioration_event(
        self,
        df: pd.DataFrame,
        event_date: datetime,
        event_type: str = 'hyperglycemia'
    ) -> pd.DataFrame:
        """
        Modify lab values to reflect health deterioration event
        """
        # Find lab test closest to event date
        df['days_to_event'] = (df['test_date'] - event_date).dt.days.abs()
        closest_idx = df['days_to_event'].idxmin()
        
        if event_type == 'hyperglycemia':
            df.loc[closest_idx, 'glucose_random'] *= 2.0
            df.loc[closest_idx, 'glucose_fasting'] *= 1.8
            
        elif event_type == 'kidney_issue':
            df.loc[closest_idx, 'creatinine'] *= 2.5
            
        elif event_type == 'infection':
            df.loc[closest_idx, 'wbc'] *= 2.0
            
        df = df.drop('days_to_event', axis=1)
        return df