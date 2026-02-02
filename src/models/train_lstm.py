# src/models/train_lstm.py
"""
LSTM Model Training with Comprehensive Logging
Enhanced with structured logging, performance tracking, and audit trail

WHY LSTM?
- Captures temporal dependencies (current value depends on past)
- Remembers long-term patterns
- Natural fit for time-series health data
- Can learn gradual deterioration patterns
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import time

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


class LSTMHealthPredictor:
    """LSTM-based deterioration prediction model with comprehensive logging"""
    
    def __init__(self, sequence_length: int = 24, lstm_units: int = 64, dropout: float = 0.2):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.training_id = f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Initializing LSTMHealthPredictor",
            extra={
                'training_id': self.training_id,
                'sequence_length': sequence_length,
                'lstm_units': lstm_units,
                'dropout': dropout
            }
        )
        
    def create_sequences_per_patient(self, X, y, patient_ids):
        X_seq = []
        y_seq = []
        seq_patient_ids = []
        
        for patient_id in np.unique(patient_ids):
            # Get this patient's data
            patient_mask = (patient_ids == patient_id)
            X_patient = X[patient_mask]
            y_patient = y[patient_mask]
            
            # Create sequences for this patient
            for i in range(len(X_patient) - self.sequence_length):
                X_seq.append(X_patient[i:i+self.sequence_length])
                y_seq.append(y_patient[i+self.sequence_length])
                seq_patient_ids.append(patient_id)
        
        return np.array(X_seq), np.array(y_seq), np.array(seq_patient_ids)
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        """Prepare LSTM data with PATIENT-LEVEL splitting (NO DATA LEAKAGE)"""

        logger.info(
            "Starting LSTM data preparation with patient-level splitting",
            extra={
                'training_id': self.training_id,
                'total_samples': len(df),
                'total_patients': df['patient_id'].nunique()
            }
        )

        start_time = time.time()

        print("\n" + "="*70)
        print("📊 LSTM DATA PREPARATION (PATIENT-LEVEL SPLITTING)")
        print("="*70)

        # Remove non-feature columns
        exclude_cols = ['timestamp', 'patient_id', 'device_id', 'activity_state', 'label']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        print(f"\n📋 Dataset info:")
        print(f"   Total samples: {len(df)}")
        print(f"   Total patients: {df['patient_id'].nunique()}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Sequence length: {self.sequence_length}")

        # -----------------------------
        # PATIENT-LEVEL SPLIT
        # -----------------------------
        patients = df['patient_id'].unique()

        train_patients, test_patients = train_test_split(
            patients, test_size=test_size, random_state=42
        )

        train_patients, val_patients = train_test_split(
            train_patients,
            test_size=val_size / (1 - test_size),
            random_state=42
        )

        print(f"\n👥 Patient split:")
        print(f"   Train patients: {len(train_patients)}")
        print(f"   Val patients: {len(val_patients)}")
        print(f"   Test patients: {len(test_patients)}")

        # -----------------------------
        # SPLIT DATAFRAMES
        # -----------------------------
        train_df = df[df['patient_id'].isin(train_patients)].copy()
        val_df = df[df['patient_id'].isin(val_patients)].copy()
        test_df = df[df['patient_id'].isin(test_patients)].copy()

        # -----------------------------
        # EXTRACT ARRAYS
        # -----------------------------
        X_train_raw = train_df[feature_cols].values
        y_train_raw = train_df['label'].values
        train_patient_ids = train_df['patient_id'].values

        X_val_raw = val_df[feature_cols].values
        y_val_raw = val_df['label'].values
        val_patient_ids = val_df['patient_id'].values

        X_test_raw = test_df[feature_cols].values
        y_test_raw = test_df['label'].values
        test_patient_ids = test_df['patient_id'].values

        # -----------------------------
        # CREATE SEQUENCES (PER PATIENT)
        # -----------------------------
        print("\n🔄 Creating sequences (per patient)...")
        X_train, y_train, _ = self.create_sequences_per_patient(
            X_train_raw, y_train_raw, train_patient_ids
        )

        X_val, y_val, _ = self.create_sequences_per_patient(
            X_val_raw, y_val_raw, val_patient_ids
        )

        X_test, y_test, _ = self.create_sequences_per_patient(
            X_test_raw, y_test_raw, test_patient_ids
        )

        print(f"\n📊 Sequence splits:")
        print(f"   Train: {len(X_train)} ({np.sum(y_train)} deterioration)")
        print(f"   Val:   {len(X_val)} ({np.sum(y_val)} deterioration)")
        print(f"   Test:  {len(X_test)} ({np.sum(y_test)} deterioration)")

        # -----------------------------
        # CLASS WEIGHTS
        # -----------------------------
        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)

        self.class_weights = {
            0: 1.0,
            1: neg / pos if pos > 0 else 1.0
        }

        print(f"\n⚖️  Class weights:")
        print(f"   Stable (0): {self.class_weights[0]:.2f}")
        print(f"   Deterioration (1): {self.class_weights[1]:.2f}")

        logger.info(
            "LSTM data preparation completed",
            extra={
                'training_id': self.training_id,
                'train_sequences': len(X_train),
                'val_sequences': len(X_val),
                'test_sequences': len(X_test)
            }
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_shape):
        """Build LSTM architecture with logging"""
        logger.info(
            "Building LSTM model",
            extra={
                'training_id': self.training_id,
                'input_shape': input_shape,
                'lstm_units': self.lstm_units,
                'dropout': self.dropout
            }
        )
        
        print("\n" + "="*70)
        print("🏗️  BUILDING LSTM MODEL")
        print("="*70)
        
        try:
            model = keras.Sequential([
                layers.LSTM(
                    self.lstm_units, 
                    return_sequences=True, 
                    input_shape=input_shape,
                    name='lstm_1'
                ),
                layers.Dropout(self.dropout, name='dropout_1'),
                layers.LSTM(self.lstm_units, name='lstm_2'),
                layers.Dropout(self.dropout, name='dropout_2'),
                layers.Dense(32, activation='relu', name='dense_1'),
                layers.Dense(1, activation='sigmoid', name='output')
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.AUC(name='pr_auc', curve='PR')
                ]
            )
            
            print(f"\n📋 Model Architecture:")
            model.summary()
            
            # Count parameters
            total_params = model.count_params()
            
            logger.info(
                "LSTM model built successfully",
                extra={
                    'training_id': self.training_id,
                    'total_parameters': total_params,
                    'layers': len(model.layers)
                }
            )
            
            return model
            
        except Exception as e:
            logger.error(
                f"Model building failed: {str(e)}",
                exc_info=True,
                extra={'training_id': self.training_id}
            )
            error_logger.log_error(e, context={
                'operation': 'build_model',
                'training_id': self.training_id
            })
            raise
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train LSTM model with logging"""
        logger.info(
            "Starting LSTM training",
            extra={
                'training_id': self.training_id,
                'epochs': epochs,
                'batch_size': batch_size,
                'train_samples': len(X_train)
            }
        )
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("🤖 TRAINING LSTM MODEL")
        print("="*70)
        
        try:
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_pr_auc',
                    patience=10,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            print(f"\n⚙️  Training configuration:")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Early stopping: patience=10")
            
            print(f"\n🔄 Training...")
            
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                class_weight=self.class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            actual_epochs = len(self.history.history['loss'])
            
            print(f"\n✅ Training complete!")
            print(f"   Training time: {training_time:.2f}s")
            print(f"   Epochs completed: {actual_epochs}")
            
            logger.info(
                "LSTM training completed",
                extra={
                    'training_id': self.training_id,
                    'duration_seconds': training_time,
                    'epochs_completed': actual_epochs,
                    'final_train_loss': float(self.history.history['loss'][-1]),
                    'final_val_loss': float(self.history.history['val_loss'][-1]),
                    'final_val_pr_auc': float(self.history.history['val_pr_auc'][-1])
                }
            )
            
            # Log performance metrics
            perf_logger.log_model_training(
                model_type='lstm',
                training_time_seconds=training_time,
                samples=len(X_train),
                performance_metrics={
                    'final_val_pr_auc': float(self.history.history['val_pr_auc'][-1]),
                    'epochs': actual_epochs
                }
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            
            logger.error(
                f"LSTM training failed: {str(e)}",
                exc_info=True,
                extra={
                    'training_id': self.training_id,
                    'elapsed_time_seconds': training_time
                }
            )
            error_logger.log_error(e, context={
                'operation': 'train',
                'training_id': self.training_id
            })
            raise
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance with logging"""
        logger.info(
            "Starting evaluation",
            extra={
                'training_id': self.training_id,
                'test_samples': len(X_test),
                'threshold': threshold
            }
        )
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("📈 MODEL EVALUATION")
        print("="*70)
        
        try:
            # Predictions
            y_pred_proba = self.model.predict(X_test).flatten()
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            print(f"\n📊 Performance Metrics:")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   PR-AUC: {pr_auc:.4f} ⭐")
            
            print(f"\n📋 Classification Report (threshold={threshold}):")
            print(classification_report(y_test, y_pred, target_names=['Stable', 'Deterioration']))
            
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n🔢 Confusion Matrix:")
            print(f"   True Negatives:  {cm[0,0]}")
            print(f"   False Positives: {cm[0,1]}")
            print(f"   False Negatives: {cm[1,0]} ⚠️")
            print(f"   True Positives:  {cm[1,1]} ✅")
            
            eval_time = time.time() - start_time
            
            logger.info(
                "Evaluation completed",
                extra={
                    'training_id': self.training_id,
                    'roc_auc': float(roc_auc),
                    'pr_auc': float(pr_auc),
                    'duration_seconds': eval_time,
                    'true_positives': int(cm[1,1]),
                    'false_negatives': int(cm[1,0])
                }
            )
            
            self.evaluation = {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Audit log
            audit_logger.log_access(
                user_id='system',
                action='model_evaluation',
                resource='lstm_model',
                success=True,
                details={
                    'training_id': self.training_id,
                    'pr_auc': float(pr_auc),
                    'roc_auc': float(roc_auc)
                }
            )
            
            return pr_auc
            
        except Exception as e:
            logger.error(
                f"Evaluation failed: {str(e)}",
                exc_info=True,
                extra={'training_id': self.training_id}
            )
            error_logger.log_error(e, context={
                'operation': 'evaluate',
                'training_id': self.training_id
            })
            raise
    
    def plot_training_history(self, save_path="models/lstm_training_history.png"):
        """Plot training curves with logging"""
        logger.debug(f"Plotting training history: {save_path}")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
            axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Model Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy
            axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy')
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Model Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # ROC-AUC
            axes[1, 0].plot(self.history.history['auc'], label='Train AUC')
            axes[1, 0].plot(self.history.history['val_auc'], label='Val AUC')
            axes[1, 0].set_title('ROC-AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # PR-AUC
            axes[1, 1].plot(self.history.history['pr_auc'], label='Train PR-AUC')
            axes[1, 1].plot(self.history.history['val_pr_auc'], label='Val PR-AUC')
            axes[1, 1].set_title('PR-AUC (Primary Metric)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PR-AUC')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Saved training history: {save_path}")
            plt.close()
            
            logger.info(f"Training history plot saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot training history: {str(e)}", exc_info=True)
    
    def save_model(self, model_path="models/lstm_model.h5"):
        """Save trained model and scaler with logging"""
        logger.info(
            "Saving LSTM model",
            extra={
                'training_id': self.training_id,
                'model_path': model_path
            }
        )
        
        try:
            # Save model
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            # Save metadata
            metadata = {
                'model_type': 'LSTM',
                'trained_date': datetime.now().isoformat(),
                'training_id': self.training_id,
                'sequence_length': self.sequence_length,
                'lstm_units': self.lstm_units,
                'dropout': self.dropout,
                'performance': {
                    'roc_auc': float(self.evaluation['roc_auc']),
                    'pr_auc': float(self.evaluation['pr_auc'])
                }
            }
            
            metadata_path = model_path.replace('.h5', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n💾 Model saved:")
            print(f"   Model: {model_path}")
            print(f"   Scaler: {scaler_path}")
            print(f"   Metadata: {metadata_path}")
            
            logger.info(
                "LSTM model saved successfully",
                extra={
                    'training_id': self.training_id,
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'metadata_path': metadata_path
                }
            )
            
            # Audit log
            audit_logger.log_access(
                user_id='system',
                action='save_model',
                resource='lstm_model',
                success=True,
                details={
                    'training_id': self.training_id,
                    'model_path': model_path
                }
            )
            
        except Exception as e:
            logger.error(
                f"Model save failed: {str(e)}",
                exc_info=True,
                extra={'training_id': self.training_id}
            )
            error_logger.log_error(e, context={
                'operation': 'save_model',
                'training_id': self.training_id
            })
            raise


def train_lstm_model():
    """Main LSTM training pipeline with logging"""
    pipeline_start = time.time()
    
    logger.info("Starting LSTM training pipeline")
    
    print("\n" + "="*70)
    print(" "*20 + "LSTM TRAINING PIPELINE")
    print("="*70)
    
    try:
        print("\n📂 Loading engineered features...")
        df = pd.read_csv("data/processed/features_engineered.csv")
        print(f"   Loaded: {len(df)} samples")
        
        predictor = LSTMHealthPredictor(
            sequence_length=24,
            lstm_units=64,
            dropout=0.2
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df)
        
        predictor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
        
        pr_auc = predictor.evaluate(X_test, y_test)
        
        predictor.plot_training_history()
        
        predictor.save_model()
        
        total_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("✅ LSTM TRAINING COMPLETE!")
        print("="*70)
        print(f"\n🎯 Final Performance:")
        print(f"   PR-AUC: {pr_auc:.4f}")
        print(f"   Training time: {total_time:.2f}s")
        print(f"\n📁 Outputs:")
        print(f"   Model: models/lstm_model.h5")
        print(f"   Plots: models/lstm_*.png")
        
        logger.info(
            "LSTM training pipeline completed",
            extra={
                'duration_seconds': total_time,
                'pr_auc': float(pr_auc)
            }
        )
        
        return predictor
        
    except Exception as e:
        total_time = time.time() - pipeline_start
        
        logger.error(
            f"LSTM training pipeline failed: {str(e)}",
            exc_info=True,
            extra={'elapsed_time_seconds': total_time}
        )
        error_logger.log_error(e, context={'operation': 'train_lstm_model'})
        raise


if __name__ == "__main__":
    predictor = train_lstm_model()