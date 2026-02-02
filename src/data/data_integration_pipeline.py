# src/data/data_integration_pipeline.py
"""
Data Integration Pipeline with Comprehensive Logging
Merges wearable data with lab results and creates ML-ready features

Enhanced Features:
- Structured logging for all pipeline steps
- Performance tracking for each operation
- Audit trail for data processing
- Error handling with context
- Progress monitoring

Task #1 Deliverable: Complete data preprocessing pipeline
"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pykalman import KalmanFilter
from typing import Tuple, Dict
import yaml
import time
import uuid
from pathlib import Path

import sys
sys.path.append('../..')

from logging_config import (
    setup_logging, get_logger,
    PerformanceLogger, AuditLogger, ErrorLogger
)

# Initialize logging
setup_logging(log_level="INFO", enable_json=True)
logger = get_logger(__name__)
perf_logger = PerformanceLogger()
audit_logger = AuditLogger()
error_logger = ErrorLogger()


class DataIntegrationPipeline:
    """
    Integrates multiple health data sources into ML-ready feature matrix with logging
    
    This is an end-to-end data pipeline with comprehensive logging that handles 
    multiple data sources with different frequencies, applies signal processing, 
    and creates time-aligned features for machine learning
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Load configuration with logging"""
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            f"Initializing DataIntegrationPipeline",
            extra={'pipeline_id': self.pipeline_id, 'config_path': config_path}
        )

        try:
        
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.prediction_window = self.config['data']['prediction_window_hours']
            self.smoothing_enabled = self.config['data']['smoothing']['enabled']
            
            logger.info(
                f"Configuration loaded",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'prediction_window_hours': self.prediction_window,
                    'smoothing_enabled': self.smoothing_enabled
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={
                'operation': 'init',
                'config_path': config_path
            })
            raise
    
    def load_data(
        self, 
        wearable_path: str = 'data/processed/apple_health_data.csv',
        lab_path: str = 'data/processed/lab_results.csv'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both data sources with logging"""
        logger.info(
            f"Loading data sources",
            extra={
                'pipeline_id': self.pipeline_id,
                'wearable_path': wearable_path,
                'lab_path': lab_path
            }
        )
        
        print("\n📥 Loading data sources...")
        
        start_time = time.time()
        
        try:
            # Load wearable data
            wearable_df = pd.read_csv(wearable_path)
            wearable_df['timestamp'] = pd.to_datetime(wearable_df['timestamp'])
            print(f"   ✅ Wearable data: {len(wearable_df)} records")
            
            logger.info(
                f"Wearable data loaded",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'records': len(wearable_df),
                    'columns': len(wearable_df.columns),
                    'date_range': f"{wearable_df['timestamp'].min()} to {wearable_df['timestamp'].max()}"
                }
            )
            
            # Load lab data
            lab_df = pd.read_csv(lab_path)
            lab_df['test_date'] = pd.to_datetime(lab_df['test_date'])
            print(f"   ✅ Lab data: {len(lab_df)} records")
            
            logger.info(
                f"Lab data loaded",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'records': len(lab_df),
                    'columns': len(lab_df.columns)
                }
            )
            
            load_time = time.time() - start_time
            
            logger.info(
                f"Data loading completed",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'duration_seconds': load_time,
                    'total_records': len(wearable_df) + len(lab_df)
                }
            )
            
            # Audit trail
            audit_logger.log_access(
                user_id='system',
                action='load_data',
                resource='health_data',
                success=True,
                details={
                    'pipeline_id': self.pipeline_id,
                    'wearable_records': len(wearable_df),
                    'lab_records': len(lab_df)
                }
            )
            
            return wearable_df, lab_df
            
        except Exception as e:
            logger.error(
                f"Data loading failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'load_data',
                'pipeline_id': self.pipeline_id,
                'wearable_path': wearable_path,
                'lab_path': lab_path
            })
            raise
    
    def apply_kalman_filter(self, series: pd.Series, process_variance: float = 0.01) -> pd.Series:
        """Apply Kalman filter to smooth noisy sensor data with logging"""
        logger.debug(
            f"Applying Kalman filter",
            extra={
                'pipeline_id': self.pipeline_id,
                'series_length': len(series),
                'process_variance': process_variance
            }
        )
        
        try:
            # Handle NaN values
            valid_data = series.dropna()
            if len(valid_data) < 2:
                logger.warning(
                    f"Insufficient data for Kalman filter",
                    extra={
                        'pipeline_id': self.pipeline_id,
                        'valid_count': len(valid_data)
                    }
                )
                return series
            
            # Initialize Kalman filter
            kf = KalmanFilter(
                transition_matrices=[1],
                observation_matrices=[1],
                initial_state_mean=valid_data.iloc[0],
                initial_state_covariance=1,
                observation_covariance=1,
                transition_covariance=process_variance
            )
            
            # Apply filter
            measurements = valid_data.values.reshape(-1, 1)
            state_means, _ = kf.filter(measurements)
            smoothed = pd.Series(state_means.flatten(), index=valid_data.index)
            
            # Reindex to original series
            result = series.copy()
            result.loc[valid_data.index] = smoothed
            
            logger.debug(
                f"Kalman filter applied successfully",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'smoothed_points': len(valid_data)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Kalman filter failed: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={
                'operation': 'apply_kalman_filter',
                'pipeline_id': self.pipeline_id
            })
            return series  # Return original on error
    
    def smooth_wearable_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Kalman filtering to noisy wearable signals with logging"""
        logger.info(
            f"Starting signal smoothing",
            extra={'pipeline_id': self.pipeline_id}
        )
        
        print("\n🔧 Applying Kalman filter for signal smoothing...")
        
        start_time = time.time()
        signals_to_smooth = ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2']
        smoothed_count = 0
        
        try:
            for signal in signals_to_smooth:
                if signal in df.columns:
                    print(f"   Smoothing {signal}...", end=' ')
                    signal_start = time.time()
                    
                    df[f'{signal}_raw'] = df[signal]  # Keep original
                    df[signal] = self.apply_kalman_filter(df[signal])
                    
                    signal_time = time.time() - signal_start
                    smoothed_count += 1
                    
                    logger.info(
                        f"Signal smoothed",
                        extra={
                            'pipeline_id': self.pipeline_id,
                            'signal': signal,
                            'duration_seconds': signal_time
                        }
                    )
                    
                    print("✅")
            
            total_time = time.time() - start_time
            
            logger.info(
                f"Signal smoothing completed",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'signals_smoothed': smoothed_count,
                    'duration_seconds': total_time
                }
            )
            
            return df
            
        except Exception as e:
            logger.error(
                f"Signal smoothing failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'smooth_wearable_signals',
                'pipeline_id': self.pipeline_id
            })
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values with logging"""
        logger.info(
            f"Starting missing value imputation",
            extra={'pipeline_id': self.pipeline_id}
        )
        
        print("\n🔍 Handling missing values...")
        
        start_time = time.time()
        max_gap = self.config['data']['imputation']['max_gap_hours']
        total_filled = 0
        
        try:
            for col in df.columns:
                if col == 'timestamp':
                    continue
                
                missing_before = df[col].isna().sum()
                if missing_before == 0:
                    continue
                
                # Forward fill with limit
                df[col] = df[col].fillna(method='ffill', limit=max_gap * 12)
                
                missing_after = df[col].isna().sum()
                filled = missing_before - missing_after
                total_filled += filled
                
                print(f"   {col}: {filled} values filled, {missing_after} remain missing")
                
                logger.debug(
                    f"Missing values imputed for column",
                    extra={
                        'pipeline_id': self.pipeline_id,
                        'column': col,
                        'filled': filled,
                        'remaining': missing_after
                    }
                )
            
            imputation_time = time.time() - start_time
            
            logger.info(
                f"Missing value imputation completed",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'total_filled': total_filled,
                    'duration_seconds': imputation_time
                }
            )
            
            return df
            
        except Exception as e:
            logger.error(
                f"Missing value imputation failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'handle_missing_values',
                'pipeline_id': self.pipeline_id
            })
            raise
    
    def merge_lab_data(self, wearable_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
        """Merge periodic lab data with continuous wearable data with logging"""
        logger.info(
            f"Starting data merge",
            extra={
                'pipeline_id': self.pipeline_id,
                'wearable_records': len(wearable_df),
                'lab_records': len(lab_df)
            }
        )
        
        print("\n🔗 Merging lab data with wearable data...")
        
        start_time = time.time()
        
        try:
            # Prepare lab data
            lab_df_for_merge = lab_df.rename(columns={'test_date': 'timestamp'})
            lab_features = [col for col in lab_df_for_merge.columns 
                           if col not in ['patient_id', 'timestamp', 'test_type']]
            
            # Merge using asof
            merged_df = pd.merge_asof(
                wearable_df.sort_values('timestamp'),
                lab_df_for_merge[['timestamp'] + lab_features].sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            
            merge_time = time.time() - start_time
            
            print(f"   ✅ Merged: {len(merged_df)} records with {len(lab_features)} lab features")
            
            logger.info(
                f"Data merge completed",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'merged_records': len(merged_df),
                    'lab_features': len(lab_features),
                    'duration_seconds': merge_time
                }
            )
            
            return merged_df
            
        except Exception as e:
            logger.error(
                f"Data merge failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'merge_lab_data',
                'pipeline_id': self.pipeline_id
            })
            raise
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels with logging"""
        logger.info(
            f"Starting label creation",
            extra={
                'pipeline_id': self.pipeline_id,
                'prediction_window_hours': self.prediction_window
            }
        )
        
        print(f"\n🏷️ Creating labels (deterioration in next {self.prediction_window}h)...")
        
        start_time = time.time()
        
        try:
            # Define deterioration thresholds
            thresholds = {
                'heart_rate': (50, 120),
                'bp_systolic': (90, 180),
                'bp_diastolic': (60, 110),
                'spo2': (90, 100),
                'glucose_random': (60, 250)
            }
            
            labels = []
            
            for idx in range(len(df)):
                current_time = df.loc[idx, 'timestamp']
                future_time = current_time + timedelta(hours=self.prediction_window)
                
                # Get future window
                future_data = df[(df['timestamp'] > current_time) & 
                                (df['timestamp'] <= future_time)]
                
                if len(future_data) == 0:
                    labels.append(np.nan)
                    continue
                
                # Check if any threshold violated
                deterioration = False
                
                for feature, (low, high) in thresholds.items():
                    if feature in future_data.columns:
                        values = future_data[feature].dropna()
                        if len(values) > 0:
                            if (values < low).any() or (values > high).any():
                                deterioration = True
                                break
                
                labels.append(1 if deterioration else 0)
            
            df['label'] = labels
            df = df.dropna(subset=['label'])
            
            # Count class distribution
            n_deterioration = int((df['label'] == 1).sum())
            n_stable = int((df['label'] == 0).sum())
            imbalance_ratio = n_deterioration / len(df) * 100
            
            label_time = time.time() - start_time
            
            print(f"   ✅ Labels created:")
            print(f"      Deterioration events: {n_deterioration} ({imbalance_ratio:.2f}%)")
            print(f"      Stable periods: {n_stable} ({100-imbalance_ratio:.2f}%)")
            print(f"      ⚠️ Class imbalance: {n_stable/n_deterioration:.1f}:1 ratio")
            
            logger.info(
                f"Label creation completed",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'total_samples': len(df),
                    'deterioration_events': n_deterioration,
                    'stable_periods': n_stable,
                    'imbalance_ratio': n_stable/n_deterioration if n_deterioration > 0 else 0,
                    'duration_seconds': label_time
                }
            )
            
            # Audit trail
            audit_logger.log_access(
                user_id='system',
                action='create_labels',
                resource='dataset',
                success=True,
                details={
                    'pipeline_id': self.pipeline_id,
                    'positive_class': n_deterioration,
                    'negative_class': n_stable
                }
            )
            
            return df
            
        except Exception as e:
            logger.error(
                f"Label creation failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'create_labels',
                'pipeline_id': self.pipeline_id
            })
            raise
    
    def save_features(self, df: pd.DataFrame, output_path: str = 'data/processed/features.csv'):
        """Save final feature matrix with logging"""
        logger.info(
            f"Saving features",
            extra={
                'pipeline_id': self.pipeline_id,
                'output_path': output_path,
                'samples': len(df),
                'features': len(df.columns)
            }
        )
        
        print(f"\n💾 Saving features to {output_path}...")
        
        try:
            df.to_csv(output_path, index=False)
            
            logger.info(
                f"Features saved successfully",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'output_path': output_path,
                    'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
                }
            )
            
            print("\n" + "=" * 60)
            print("✅ DATA INTEGRATION PIPELINE COMPLETE!")
            print("=" * 60)
            print(f"\n📊 Final Dataset:")
            print(f"   Samples: {len(df)}")
            print(f"   Features: {len(df.columns) - 2}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"\n🎯 Ready for ML training!")
            
            # Audit trail
            audit_logger.log_data_export(
                user_id='system',
                data_type='processed_features',
                record_count=len(df),
                approved_by='pipeline'
            )
            
        except Exception as e:
            logger.error(
                f"Feature saving failed: {str(e)}",
                exc_info=True,
                extra={'pipeline_id': self.pipeline_id}
            )
            error_logger.log_error(e, context={
                'operation': 'save_features',
                'pipeline_id': self.pipeline_id,
                'output_path': output_path
            })
            raise
    
    def run_pipeline(
        self,
        wearable_path: str = 'data/processed/apple_health_data.csv',
        lab_path: str = 'data/processed/lab_results.csv',
        output_path: str = 'data/processed/features.csv'
    ) -> pd.DataFrame:
        """Run complete data integration pipeline with comprehensive logging"""
        pipeline_start = time.time()
        
        logger.info(
            f"Starting data integration pipeline",
            extra={'pipeline_id': self.pipeline_id}
        )
        
        print("\n" + "=" * 60)
        print("🔄 DATA INTEGRATION PIPELINE")
        print("=" * 60)
        print(f"Pipeline ID: {self.pipeline_id}")
        
        try:
            # Step 1: Load
            wearable_df, lab_df = self.load_data(wearable_path, lab_path)
            
            # Step 2: Smooth signals
            if self.smoothing_enabled:
                wearable_df = self.smooth_wearable_signals(wearable_df)
            else:
                logger.info(f"Signal smoothing disabled")
            
            # Step 3: Handle missing values
            wearable_df = self.handle_missing_values(wearable_df)
            
            # Step 4: Merge lab data
            merged_df = self.merge_lab_data(wearable_df, lab_df)
            
            # Step 5: Create labels
            final_df = self.create_labels(merged_df)
            
            # Step 6: Save
            self.save_features(final_df, output_path)
            
            total_time = time.time() - pipeline_start
            
            logger.info(
                f"Data integration pipeline completed successfully",
                extra={
                    'pipeline_id': self.pipeline_id,
                    'total_duration_seconds': total_time,
                    'output_samples': len(final_df),
                    'output_features': len(final_df.columns)
                }
            )
            
            print(f"\n⏱️ Total pipeline time: {total_time:.2f} seconds")
            print(f"📝 Check logs/ directory for detailed execution logs")
            
            return final_df
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            
            logger.error(
                f"Data integration pipeline failed: {str(e)}",
                exc_info=True,
                extra={
                    'pipeline_id': self.pipeline_id,
                    'elapsed_time_seconds': total_time
                }
            )
            
            error_logger.log_error(e, context={
                'operation': 'run_pipeline',
                'pipeline_id': self.pipeline_id
            })
            
            raise


if __name__ == "__main__":
    import os
    
    # Run the complete pipeline
    pipeline = DataIntegrationPipeline()
    features_df = pipeline.run_pipeline()
    
    print("\n📈 Sample of final features:")
    print(features_df.head())