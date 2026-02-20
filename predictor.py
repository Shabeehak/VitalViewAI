# predictor.py
"""
Health Deterioration Predictor with Comprehensive Logging
Production-ready prediction interface with full observability

Enhanced Features:
- Structured logging for all operations
- Performance tracking for predictions
- Audit trail for patient data access
- Error handling with context
- Request correlation for debugging
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from typing import Dict, List, Union, Optional
import sys
import time
import uuid

sys.path.append('src')

from features.feature_engineering import HealthFeatureEngineering
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


class HealthPredictor:
    """
    Production-ready deterioration risk predictor with comprehensive logging
    
    Features:
    - Loads trained model
    - Performs feature engineering
    - Returns risk score [0, 1]
    - Provides risk interpretation
    - Full logging and audit trail
    """
    
    def __init__(
        self, 
        model_path: str = "models/xgboost_model.pkl",
        model_type: str = "xgboost"
    ):
        """
        Load trained model with logging
        
        Args:
            model_path: Path to saved model
            model_type: 'xgboost' or 'lstm'
        """
        self.model_type = model_type
        self.model_path = model_path
        self.prediction_count = 0
        
        logger.info(
            f"Initializing HealthPredictor",
            extra={
                'model_type': model_type,
                'model_path': model_path
            }
        )
        try:
            # Check if model exists, if not download from Google Drive
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}, downloading from Google Drive...")
                self._download_model_from_gdrive(model_path)
            
            # Load model
            if model_type == "xgboost":
                self.model = joblib.load(model_path)
                metadata_path = model_path.replace('.pkl', '_metadata.json')
            else:  # lstm
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.scaler = joblib.load(model_path.replace('.h5', '_scaler.pkl'))
                metadata_path = model_path.replace('.h5', '_metadata.json')
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Initialize feature engineer
            self.feature_engineer = HealthFeatureEngineering()
            
            logger.info(
                f"Model loaded successfully",
                extra={
                    'model_type': model_type,
                    'trained_date': self.metadata['trained_date'],
                    'pr_auc': self.metadata['performance']['pr_auc'],
                    'feature_count': len(self.metadata.get('feature_names', []))
                }
            )
            
            print(f"✅ Model loaded successfully")
            print(f"   Trained: {self.metadata['trained_date']}")
            print(f"   Performance: PR-AUC = {self.metadata['performance']['pr_auc']:.3f}")
            
        except FileNotFoundError as e:
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg, exc_info=True)
            error_logger.log_error(e, context={'model_path': model_path})
            raise
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            error_logger.log_error(e, context={
                'model_type': model_type,
                'model_path': model_path
            })
            raise
    
    def preprocess_input(
        self, 
        patient_data: pd.DataFrame,
        correlation_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Prepare patient data for prediction with logging
        
        Steps:
        1. Engineer features (rolling stats, trends, interactions)
        2. Select features used in training
        3. Handle missing values
        """
        start_time = time.time()
        
        logger.debug(
            f"Preprocessing input data",
            extra={
                'correlation_id': correlation_id,
                'input_rows': len(patient_data),
                'input_cols': len(patient_data.columns)
            }
        )
        
        try:
            # Engineer features
            data_engineered = self.feature_engineer.engineer_all_features(patient_data)
            
            # Get features used in training
            if self.model_type == "xgboost":
                training_features = self.metadata['feature_names']
                
                # Ensure all training features exist
                missing_features = []
                for feature in training_features:
                    if feature not in data_engineered.columns:
                        data_engineered[feature] = 0
                        missing_features.append(feature)
                
                # if missing_features:
                #     logger.warning(
                #         f"Missing features filled with zeros",
                #         extra={
                #             'correlation_id': correlation_id,
                #             'missing_count': len(missing_features),
                #             'missing_features': missing_features[:5]  # Log first 5
                #         }
                #     )
                
                if missing_features:
                    logger.warning(
                        f"Missing features filled with zeros",
                        extra={
                            'correlation_id': correlation_id,
                            'missing_count': len(missing_features),
                            'missing_features': missing_features[:5],  # Log first 5
                            'total_features': len(training_features),
                            'missing_percentage': len(missing_features) / len(training_features) * 100
                        }
                    )
                    
                    # If too many features are missing, warn
                    if len(missing_features) > len(training_features) * 0.3:  # >30% missing
                        logger.error(f"TOO MANY MISSING FEATURES: {len(missing_features)}/{len(training_features)}")
                # Select only training features in correct order
                X = data_engineered[training_features].values
            else:  # lstm
                # For LSTM, need to create sequences
                exclude_cols = ['timestamp', 'patient_id', 'device_id', 'activity_state', 'label']
                feature_cols = [c for c in data_engineered.columns if c not in exclude_cols]
                X = data_engineered[feature_cols].values
                X = self.scaler.transform(X)
                
                sequence_length = self.metadata['sequence_length']
                if len(X) >= sequence_length:
                    X = X[-sequence_length:].reshape(1, sequence_length, -1)
                else:
                    # Pad if not enough data
                    pad_length = sequence_length - len(X)
                    padding = np.zeros((pad_length, X.shape[1]))
                    X = np.vstack([padding, X]).reshape(1, sequence_length, -1)
                    
                    logger.warning(
                        f"Input data padded for LSTM",
                        extra={
                            'correlation_id': correlation_id,
                            'required_length': sequence_length,
                            'actual_length': len(X),
                            'pad_length': pad_length
                        }
                    )
            
            preprocessing_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Preprocessing completed",
                extra={
                    'correlation_id': correlation_id,
                    'output_shape': X.shape,
                    'preprocessing_time_ms': preprocessing_time_ms
                }
            )
            
            return X
            
        except Exception as e:
            logger.error(
                f"Preprocessing failed: {str(e)}",
                exc_info=True,
                extra={'correlation_id': correlation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'preprocess_input',
                'input_shape': patient_data.shape
            })
            raise
    
    def predict(
        self, 
        patient_data: Union[pd.DataFrame, Dict],
        patient_id: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> Dict:
        """
        Predict deterioration risk with comprehensive logging
        
        Args:
            patient_data: DataFrame or dict with patient vitals
            patient_id: Patient identifier for audit logging
            user_id: User making prediction for audit
            correlation_id: Request correlation ID for tracing
            
        Returns:
            Dictionary with:
            - risk_score: Float [0, 1]
            - risk_level: 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
            - alert: Boolean
            - interpretation: Human-readable message
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        logger.info(
            f"Starting prediction",
            extra={
                'correlation_id': correlation_id,
                'patient_id': patient_id,
                'user_id': user_id,
                'model_type': self.model_type
            }
        )
        
        try:
            # Convert dict to DataFrame if needed
            if isinstance(patient_data, dict):
                patient_data = pd.DataFrame([patient_data])
            
            # Preprocess
            X = self.preprocess_input(patient_data, correlation_id)
            
            # Predict
            prediction_start = time.time()
            
            if self.model_type == "xgboost":
                risk_score = float(self.model.predict_proba(X)[0, 1])
            else:  # lstm
                risk_score = float(self.model.predict(X)[0, 0])
            
            prediction_time_ms = (time.time() - prediction_start) * 1000
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "LOW"
                alert = False
                interpretation = "Patient vitals are stable. Continue routine monitoring."
            elif risk_score < 0.5:
                risk_level = "MEDIUM"
                alert = False
                interpretation = "Patient showing minor deviations. Increased monitoring recommended."
            elif risk_score < 0.7:
                risk_level = "HIGH"
                alert = True
                interpretation = "Patient at elevated risk of deterioration. Close monitoring required."
            else:
                risk_level = "CRITICAL"
                alert = True
                interpretation = "Patient at high risk of deterioration in next 48 hours. Immediate clinical assessment recommended."
            
            total_time_ms = (time.time() - start_time) * 1000
            
            # Update prediction count
            self.prediction_count += 1
            
            # Log performance metrics
            perf_logger.log_prediction(
                patient_id=patient_id or 'unknown',
                inference_time_ms=total_time_ms,
                risk_score=risk_score,
                alert=alert
            )
            
            # Log audit trail
            if patient_id and user_id:
                audit_logger.log_access(
                    user_id=user_id,
                    action='predict_deterioration',
                    resource=patient_id,
                    success=True,
                    details={
                        'risk_score': risk_score,
                        'risk_level': risk_level,
                        'alert': alert,
                        'model_type': self.model_type
                    }
                )
            
            logger.info(
                f"Prediction completed",
                extra={
                    'correlation_id': correlation_id,
                    'patient_id': patient_id,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'alert': alert,
                    'inference_time_ms': total_time_ms,
                    'prediction_time_ms': prediction_time_ms
                }
            )
            
            # Log alert if critical
            if alert:
                logger.warning(
                    f"ALERT: High risk deterioration detected",
                    extra={
                        'correlation_id': correlation_id,
                        'patient_id': patient_id,
                        'risk_score': risk_score,
                        'risk_level': risk_level
                    }
                )
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'alert': alert,
                'interpretation': interpretation,
                'timestamp': datetime.now().isoformat(),
                'correlation_id': correlation_id,
                'inference_time_ms': total_time_ms,
                'model_type': self.model_type
            }
            
        except Exception as e:
            total_time_ms = (time.time() - start_time) * 1000
            
            logger.error(
                f"Prediction failed: {str(e)}",
                exc_info=True,
                extra={
                    'correlation_id': correlation_id,
                    'patient_id': patient_id,
                    'elapsed_time_ms': total_time_ms
                }
            )
            
            error_logger.log_error(e, context={
                'operation': 'predict',
                'patient_id': patient_id,
                'user_id': user_id,
                'correlation_id': correlation_id
            })
            
            # Log failed audit trail
            if patient_id and user_id:
                audit_logger.log_access(
                    user_id=user_id,
                    action='predict_deterioration',
                    resource=patient_id,
                    success=False,
                    details={'error': str(e)}
                )
            
            raise
    
    def predict_from_api(
        self, 
        api_url: str, 
        patient_id: str,
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Get current data from streaming API and predict with logging
        
        Args:
            api_url: Base URL of streaming API
            patient_id: Patient identifier
            user_id: User making request
            
        Returns:
            Prediction result dictionary
        """
        import requests
        
        correlation_id = str(uuid.uuid4())
        
        logger.info(
            f"Fetching data from API for prediction",
            extra={
                'correlation_id': correlation_id,
                'api_url': api_url,
                'patient_id': patient_id
            }
        )
        
        try:
            # Get current reading from API
            response = requests.get(
                f"{api_url}/patients/{patient_id}/current",
                timeout=10
            )
            
            if response.status_code == 200:
                current_data = response.json()['data']
                
                # For better prediction, get recent history
                history_response = requests.get(
                    f"{api_url}/patients/{patient_id}/history",
                    params={'hours': 2, 'interval_minutes': 5},
                    timeout=30
                )
                
                if history_response.status_code == 200:
                    history_data = history_response.json()['data']
                    patient_data = pd.DataFrame(history_data)
                    
                    logger.debug(
                        f"Retrieved historical data",
                        extra={
                            'correlation_id': correlation_id,
                            'records': len(history_data)
                        }
                    )
                else:
                    patient_data = pd.DataFrame([current_data])
                    
                    logger.warning(
                        f"Could not get history, using current only",
                        extra={'correlation_id': correlation_id}
                    )
                
                # Predict
                result = self.predict(
                    patient_data,
                    patient_id=patient_id,
                    user_id=user_id,
                    correlation_id=correlation_id
                )
                result['patient_id'] = patient_id
                
                return result
            else:
                error_msg = f"API returned status {response.status_code}"
                logger.error(
                    error_msg,
                    extra={
                        'correlation_id': correlation_id,
                        'status_code': response.status_code
                    }
                )
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            logger.error(
                f"API request failed: {str(e)}",
                exc_info=True,
                extra={'correlation_id': correlation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'predict_from_api',
                'api_url': api_url,
                'patient_id': patient_id
            })
            raise Exception(f"Failed to get data from API: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """Get predictor statistics"""
        stats = {
            'model_type': self.model_type,
            'model_path': self.model_path,
            'predictions_made': self.prediction_count,
            'model_performance': self.metadata['performance'],
            'trained_date': self.metadata['trained_date']
        }
        
        logger.info(f"Statistics requested", extra=stats)
        
        return stats

    def _download_model_from_gdrive(self, model_path: str):
        """Download model from Google Drive if not present"""
        import requests
        import os
        
        # Your Google Drive file ID
        file_id = "1lnmxCHDiCCS1ro__Iwtqn6mT6tqXQbYH"
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        logger.info(f"Downloading model from Google Drive...")
        print(f"📥 Downloading model from Google Drive...")
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Download with session to handle large files
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Handle Google Drive's virus scan warning for large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    params = {'confirm': value}
                    response = session.get(url, params=params, stream=True)
                    break
            
            # Save the file
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"✅ Model downloaded successfully to {model_path}")
            print(f"✅ Model downloaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise Exception(f"Could not download model from Google Drive: {str(e)}")

def demo_prediction():
    """Demonstrate predictor usage with logging"""
    print("\n" + "="*70)
    print(" "*20 + "HEALTH PREDICTOR DEMO")
    print("="*70)
    
    # Load predictor
    predictor = HealthPredictor(model_type="xgboost")
    
    # Example 1: Predict from DataFrame
    print("\n📊 Example 1: Predicting from DataFrame")
    
    try:
        # Load some test data
        df = pd.read_csv("data/processed/features_engineered.csv").head(50)
        
        result = predictor.predict(
            df,
            patient_id='demo_patient_001',
            user_id='demo_user'
        )
        
        print(f"\n🎯 Prediction Result:")
        print(f"   Risk Score: {result['risk_score']:.3f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Alert: {'⚠️ YES' if result['alert'] else '✓ NO'}")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")
        print(f"   Correlation ID: {result['correlation_id']}")
        
    except FileNotFoundError:
        print("\n⚠️ Test data not found. Run data generation first.")
    
    # Example 2: Predict from API (if running)
    print("\n\n📡 Example 2: Predicting from Streaming API")
    print("   (Make sure streaming_api_server.py is running)")
    
    try:
        result = predictor.predict_from_api(
            api_url="http://localhost:8000",
            patient_id="demo_patient_001",
            user_id="demo_user"
        )
        
        print(f"\n🎯 Prediction Result:")
        print(f"   Patient: {result['patient_id']}")
        print(f"   Risk Score: {result['risk_score']:.3f}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Alert: {'⚠️ YES' if result['alert'] else '✓ NO'}")
        print(f"   Interpretation: {result['interpretation']}")
        print(f"   Correlation ID: {result['correlation_id']}")
    
    except Exception as e:
        print(f"   ⚠️ Could not connect to API: {e}")
        print(f"   To enable: python streaming_api_server.py")
    
    # Show statistics
    print("\n\n📈 Predictor Statistics:")
    stats = predictor.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n📝 Check logs/ directory for detailed logging")
    print("="*70 + "\n")


if __name__ == "__main__":
    demo_prediction()