#data\features\feature_engineering.py
"""
Feature Engineering for Health Monitoring
Creates temporal and interaction features from raw vitals

WHY THIS MATTERS:
Raw values alone aren't enough:
- HR=120 could be exercise or crisis
- But HR trend 70→80→95→120 clearly shows deterioration

This module creates features that capture PATTERNS over time.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

class HealthFeatureEngineering:
    """
    Transform raw health data into ML-ready features
    
    Adds:
    1. Rolling statistics (mean, std, min, max over time windows)
    2. Trends (is value increasing/decreasing?)
    3. Variability (how stable are vitals?)
    4. Interaction features (combined effects)
    5. Time-based features (hour of day, day of week)
    """
    
    def __init__(self):
        # Which columns to create temporal features for
        self.vital_columns = [
            'heart_rate', 
            'bp_systolic', 
            'bp_diastolic', 
            'spo2', 
            'respiratory_rate', 
            'temperature'
        ]
        
        # Time windows for rolling statistics (in hours)
        # Adjusted for better data retention
        self.windows = [1, 6, 12]  # Removed 24h to keep more samples
        
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling statistics over time windows
        
        WHY THIS IS IMPORTANT:
        - Mean: Average level over period
        - Std: Variability (high std = unstable)
        - Min/Max: Extreme values in period
        - These capture TRENDS not visible in single reading
        
        Example:
        HR at 10:00 = 95 bpm (could be normal after exercise)
        HR mean last 6h = 92 bpm (consistently elevated - concern!)
        HR std last 6h = 15 bpm (very variable - unstable!)
        """
        print("\n🔄 Creating rolling features...")
        
        # Ensure timestamp is datetime and sorted
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Set timestamp as index for rolling operations
        df_indexed = df.set_index('timestamp')
        
        for vital in self.vital_columns:
            if vital not in df.columns:
                continue
            
            print(f"   Processing {vital}...")
            
            for window_hours in self.windows:
                window_str = f"{window_hours}h"
                
                # Rolling mean
                df_indexed[f'{vital}_mean_{window_str}'] = (
                    df_indexed[vital].rolling(window=f'{window_hours}H', min_periods=1).mean()
                )
                
                # Rolling std (variability)
                df_indexed[f'{vital}_std_{window_str}'] = (
                    df_indexed[vital].rolling(window=f'{window_hours}H', min_periods=1).std()
                )
                
                # Rolling min/max
                df_indexed[f'{vital}_min_{window_str}'] = (
                    df_indexed[vital].rolling(window=f'{window_hours}H', min_periods=1).min()
                )
                
                df_indexed[f'{vital}_max_{window_str}'] = (
                    df_indexed[vital].rolling(window=f'{window_hours}H', min_periods=1).max()
                )
        
        # Reset index
        df_result = df_indexed.reset_index()
        
        print(f"   ✅ Created {len(df_result.columns) - len(df.columns)} new features")
        return df_result
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trends (rate of change)
        
        WHY THIS MATTERS:
        Model needs to know if values are:
        - Increasing (deteriorating?)
        - Decreasing (improving?)
        - Stable (monitoring)
        
        Example:
        HR: 70→75→80→85→90 (slope = +5 bpm/hour → concerning!)
        HR: 90→85→80→75→70 (slope = -5 bpm/hour → improving)
        """
        print("\n📈 Creating trend features...")
        
        for vital in self.vital_columns:
            if vital not in df.columns:
                continue
            
            # Simple difference from previous reading
            df[f'{vital}_diff'] = df[vital].diff()
            
            # Slope over last 6 readings (~30 min if 5-min sampling)
            df[f'{vital}_slope_6'] = (
                df[vital].rolling(window=6, min_periods=2)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )
            
            # Slope over last 12 readings (~1 hour)
            df[f'{vital}_slope_12'] = (
                df[vital].rolling(window=12, min_periods=2)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )
        
        print(f"   ✅ Created trend features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture interactions between vitals
        
        WHY THIS MATTERS:
        Vitals interact! Combined effects are important:
        
        Examples:
        - High HR + High BP = Cardiovascular stress
        - Low SpO2 + High respiratory rate = Respiratory distress
        - High glucose + High HR = Metabolic stress
        
        These combinations are MORE informative than individual values!
        """
        print("\n🔗 Creating interaction features...")
        
        # Cardiovascular stress index
        if 'heart_rate' in df.columns and 'bp_systolic' in df.columns:
            df['cv_stress_index'] = (
                (df['heart_rate'] / 100) * (df['bp_systolic'] / 120)
            )
        
        # Respiratory efficiency
        if 'spo2' in df.columns and 'respiratory_rate' in df.columns:
            df['respiratory_efficiency'] = df['spo2'] / df['respiratory_rate']
        
        # Metabolic demand
        if 'glucose' in df.columns and 'heart_rate' in df.columns:
            df['metabolic_demand'] = df['glucose'] * df['heart_rate'] / 1000
        
        # Blood pressure pulse pressure (systolic - diastolic)
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            df['pulse_pressure'] = df['bp_systolic'] - df['bp_diastolic']
        
        # Mean arterial pressure (MAP)
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            df['mean_arterial_pressure'] = (
                df['bp_diastolic'] + (df['pulse_pressure'] / 3)
            )
        
        print(f"   ✅ Created interaction features")
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features
        
        WHY THIS MATTERS:
        Time of day affects vitals:
        - 3 AM: Low HR, low BP (sleep)
        - 6 PM: High HR, high BP (activity)
        
        Model needs to know context!
        """
        print("\n⏰ Creating time features...")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Hour of day (0-23)
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Cyclical encoding (so hour 23 is close to hour 0)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Weekend flag
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        print(f"   ✅ Created time features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Create lagged features (previous values)
        
        WHY THIS MATTERS:
        Model can see recent history:
        - HR now: 95 bpm
        - HR 5 min ago: 90 bpm
        - HR 10 min ago: 85 bpm
        → Clear increasing trend!
        
        Note: For small datasets, we use fewer lags to preserve samples
        """
        # Adjust lags based on dataset size
        dataset_size = len(df)
        if dataset_size < 100:
            lags = [1]  # Only 1 lag for very small datasets
            print(f"   ⚠️  Small dataset detected ({dataset_size} samples)")
            print(f"   Using only 1 lag to preserve samples")
        elif dataset_size < 500:
            lags = [1, 2]  # 2 lags for small datasets
        
        print(f"\n⏮️  Creating lag features (lags: {lags})...")
        
        for vital in self.vital_columns:
            if vital not in df.columns:
                continue
            
            for lag in lags:
                df[f'{vital}_lag_{lag}'] = df[vital].shift(lag)
        
        print(f"   ✅ Created lag features")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations
        
        This is the main function to call!
        """
        print("\n" + "="*70)
        print("🔧 FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        original_features = len(df.columns)
        
        # Apply all transformations
        df = self.create_rolling_features(df)
        df = self.create_trend_features(df)
        df = self.create_interaction_features(df)
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        
        new_features = len(df.columns) - original_features
        
        print("\n" + "="*70)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"   Original features: {original_features}")
        print(f"   New features created: {new_features}")
        print(f"   Total features: {len(df.columns)}")
        
        return df


def engineer_features_for_training(
    input_path: str = "data/processed/features.csv",
    output_path: str = "data/processed/features_engineered.csv"
) -> pd.DataFrame:
    """
    Main function to engineer features from raw data
    
    Usage:
        df = engineer_features_for_training()
    """
    print("\n📂 Loading data...")
    df = pd.read_csv(input_path)
    print(f"   Loaded: {len(df)} samples")
    
    # Initialize feature engineer
    engineer = HealthFeatureEngineering()
    
    # Engineer features
    df_engineered = engineer.engineer_all_features(df)
    
    # Remove rows with NaN from lag/rolling features (first few rows)
    print(f"\n🧹 Cleaning data...")
    original_len = len(df_engineered)
    
    # For small datasets, fill NaN instead of dropping
    if original_len < 100:
        print(f"   ⚠️  Small dataset: Using forward fill instead of dropping")
        df_engineered = df_engineered.fillna(method='ffill').fillna(method='bfill')
        dropped = 0
    else:
        df_engineered = df_engineered.dropna()
        dropped = original_len - len(df_engineered)
    
    print(f"   Dropped {dropped} rows with missing values")
    print(f"   Final dataset: {len(df_engineered)} samples")
    
    # Check if we have enough data
    if len(df_engineered) < 50:
        print(f"\n   ⚠️  WARNING: Only {len(df_engineered)} samples!")
        print(f"   This is too small for reliable training")
        print(f"   Recommendation: Generate more data")
        print(f"   Run: python quick_generate_data.py --days 30")
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    df_engineered.to_csv(output_path, index=False)
    
    print("\n✅ Feature engineering complete!")
    return df_engineered


if __name__ == "__main__":
    # Run feature engineering
    df = engineer_features_for_training()
    
    print("\n📋 Sample of engineered features:")
    print(df.head(3).to_string())
    
    print("\n📊 Feature types breakdown:")
    feature_types = {
        'Original vitals': [c for c in df.columns if not any(x in c for x in ['_mean', '_std', '_min', '_max', '_diff', '_slope', '_lag', 'cv_', 'respiratory_', 'metabolic_', 'pulse_', 'hour_', 'day_', 'is_'])],
        'Rolling stats': [c for c in df.columns if any(x in c for x in ['_mean', '_std', '_min', '_max'])],
        'Trends': [c for c in df.columns if any(x in c for x in ['_diff', '_slope'])],
        'Interactions': [c for c in df.columns if any(x in c for x in ['cv_', 'respiratory_', 'metabolic_', 'pulse_', 'mean_arterial'])],
        'Time features': [c for c in df.columns if any(x in c for x in ['hour_', 'day_', 'is_weekend'])],
        'Lag features': [c for c in df.columns if '_lag_' in c],
    }
    
    for feature_type, features in feature_types.items():
        print(f"   {feature_type}: {len(features)}")