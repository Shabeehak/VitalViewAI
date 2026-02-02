
# src/models/train_xgboost.py
"""
XGBoost Model Training with Comprehensive Logging
Enhanced with structured logging, performance tracking, and audit trail
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime

try:
    from xgboost import XGBClassifier
except ImportError:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

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


class XGBoostHealthPredictor:
    """XGBoost-based deterioration prediction model with comprehensive logging"""
    
    def __init__(self, use_smote: bool = True, class_weight_ratio: int = 10):
        self.use_smote = use_smote
        self.class_weight_ratio = class_weight_ratio
        self.model = None
        self.feature_names = None
        self.training_history = {}
        self.training_id = f"xgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            "Initializing XGBoostHealthPredictor",
            extra={
                'training_id': self.training_id,
                'use_smote': use_smote,
                'class_weight_ratio': class_weight_ratio
            }
        )

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        """Prepare data with PATIENT-LEVEL splitting (prevents data leakage)"""
        logger.info(
            "Starting data preparation with patient-level splitting",
            extra={
                'training_id': self.training_id,
                'total_samples': len(df),
                'total_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 'unknown'
            }
        )
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("📊 DATA PREPARATION (PATIENT-LEVEL SPLITTING)")
        print("="*70)
        
        try:
            # Remove non-feature columns
            exclude_cols = ['timestamp', 'patient_id', 'device_id', 'activity_state', 'label']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            
            print(f"\n📋 Dataset info:")
            print(f"   Total samples: {len(df)}")
            print(f"   Total patients: {df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A'}")
            print(f"   Features: {len(feature_cols)}")
            print(f"   Deterioration cases: {sum(df['label'])} ({sum(df['label'])/len(df)*100:.2f}%)")
            
            # CRITICAL: Split by PATIENT, not by samples
            if 'patient_id' in df.columns:
                print(f"\n🔒 Performing PATIENT-LEVEL split (prevents data leakage)...")
                
                # Get unique patients
                patients = df['patient_id'].unique()
                
                # Split patients into train/test
                train_patients, test_patients = train_test_split(
                    patients, 
                    test_size=test_size, 
                    random_state=42
                )
                
                # Further split train into train/val
                train_patients, val_patients = train_test_split(
                    train_patients,
                    test_size=val_size/(1-test_size),
                    random_state=42
                )
                
                # Filter data by patient assignment
                train_mask = df['patient_id'].isin(train_patients)
                val_mask = df['patient_id'].isin(val_patients)
                test_mask = df['patient_id'].isin(test_patients)
                
                X_train = df.loc[train_mask, feature_cols].values
                y_train = df.loc[train_mask, 'label'].values
                
                X_val = df.loc[val_mask, feature_cols].values
                y_val = df.loc[val_mask, 'label'].values
                
                X_test = df.loc[test_mask, feature_cols].values
                y_test = df.loc[test_mask, 'label'].values
                
                print(f"   Train patients: {len(train_patients)}")
                print(f"   Val patients: {len(val_patients)}")
                print(f"   Test patients: {len(test_patients)}")
                print(f"\n   ✅ No patient appears in multiple splits!")
                
            else:
                # Fallback if no patient_id column
                print(f"\n⚠️  No patient_id column - using random split")
                X = df[feature_cols].values
                y = df['label'].values
                
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size/(1-test_size), 
                    random_state=42, stratify=y_temp
                )
            
            self.feature_names = feature_cols
            
            print(f"\n📊 Data splits:")
            print(f"   Train: {len(X_train)} samples ({sum(y_train)} deterioration)")
            print(f"   Val: {len(X_val)} samples ({sum(y_val)} deterioration)")
            print(f"   Test: {len(X_test)} samples ({sum(y_test)} deterioration)")
            
            # Apply SMOTE if requested
            if self.use_smote:
                minority_count = min(sum(y_train), len(y_train) - sum(y_train))
                
                if minority_count > 5:
                    print(f"\n🔄 Applying SMOTE to training data...")
                    smote_start = time.time()
                    
                    smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count-1))
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    
                    smote_time = time.time() - smote_start
                    
                    print(f"   After SMOTE: {len(X_train)} samples ({sum(y_train)} deterioration)")
                    print(f"   New ratio: {sum(y_train==0)/sum(y_train==1):.1f}:1")
                    
                    logger.info(
                        "SMOTE applied",
                        extra={
                            'training_id': self.training_id,
                            'samples_after_smote': len(X_train),
                            'duration_seconds': smote_time
                        }
                    )
            
            prep_time = time.time() - start_time
            
            logger.info(
                "Data preparation completed",
                extra={
                    'training_id': self.training_id,
                    'duration_seconds': prep_time,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test)
                }
            )
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            raise
    
    def train(self, X_train, y_train, X_val, y_val, n_estimators=100, max_depth=6, learning_rate=0.1):
        """Train XGBoost model with logging"""
        logger.info(
            "Starting model training",
            extra={
                'training_id': self.training_id,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
        )
        
        start_time = time.time()
        
        print("\n" + "="*70)
        print("🤖 MODEL TRAINING")
        print("="*70)
        
        try:
            scale_pos_weight = self.class_weight_ratio if not self.use_smote else 1
            
            print(f"\n⚙️  Hyperparameters:")
            print(f"   n_estimators: {n_estimators}")
            print(f"   max_depth: {max_depth}")
            print(f"   learning_rate: {learning_rate}")
            print(f"   scale_pos_weight: {scale_pos_weight}")
            
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=scale_pos_weight,
                eval_metric='aucpr',
                random_state=42,
                use_label_encoder=False
            )
            
            print(f"\n🔄 Training model...")
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            results = self.model.evals_result()
            self.training_history = {
                'train_aucpr': results['validation_0']['aucpr'],
                'val_aucpr': results['validation_1']['aucpr']
            }
            
            final_train_aucpr = self.training_history['train_aucpr'][-1]
            final_val_aucpr = self.training_history['val_aucpr'][-1]
            
            training_time = time.time() - start_time
            
            print(f"\n✅ Training complete!")
            print(f"   Final train PR-AUC: {final_train_aucpr:.4f}")
            print(f"   Final val PR-AUC: {final_val_aucpr:.4f}")
            
            logger.info(
                "Training completed",
                extra={
                    'training_id': self.training_id,
                    'duration_seconds': training_time,
                    'final_train_pr_auc': float(final_train_aucpr),
                    'final_val_pr_auc': float(final_val_aucpr)
                }
            )
            
            # Log performance metrics
            perf_logger.log_model_training(
                model_type='xgboost',
                training_time_seconds=training_time,
                samples=len(X_train),
                performance_metrics={
                    'train_pr_auc': float(final_train_aucpr),
                    'val_pr_auc': float(final_val_aucpr)
                }
            )
            
        except Exception as e:
            logger.error(
                f"Training failed: {str(e)}",
                exc_info=True,
                extra={'training_id': self.training_id}
            )
            error_logger.log_error(e, context={
                'operation': 'train',
                'training_id': self.training_id
            })
            raise
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """Comprehensive model evaluation with logging"""
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
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
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
                    'duration_seconds': eval_time
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
                resource='xgboost_model',
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
    
    def plot_training_history(self, save_path="models/training_history.png"):
        """Plot training curves"""
        logger.debug(f"Plotting training history: {save_path}")
        
        plt.figure(figsize=(10, 6))
        
        iterations = range(len(self.training_history['train_aucpr']))
        plt.plot(iterations, self.training_history['train_aucpr'], label='Train PR-AUC')
        plt.plot(iterations, self.training_history['val_aucpr'], label='Val PR-AUC')
        
        plt.xlabel('Iterations')
        plt.ylabel('PR-AUC')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved training history plot: {save_path}")
        plt.close()
        
        logger.info(f"Training history plot saved: {save_path}")
    
    def plot_confusion_matrix(self, save_path="models/confusion_matrix.png"):
        """Plot confusion matrix"""
        logger.debug(f"Plotting confusion matrix: {save_path}")
        
        plt.figure(figsize=(8, 6))
        
        cm = self.evaluation['confusion_matrix']
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stable', 'Deterioration'],
            yticklabels=['Stable', 'Deterioration']
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved confusion matrix: {save_path}")
        plt.close()
        
        logger.info(f"Confusion matrix saved: {save_path}")
    
    def plot_feature_importance(self, top_n=20, save_path="models/feature_importance.png"):
        """Plot feature importance"""
        logger.debug(f"Plotting feature importance: {save_path}")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_n)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved feature importance: {save_path}")
        plt.close()
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        logger.info(
            f"Feature importance saved: {save_path}",
            extra={
                'training_id': self.training_id,
                'top_feature': feature_importance.iloc[0]['feature'],
                'top_importance': float(feature_importance.iloc[0]['importance'])
            }
        )
        
        return feature_importance
    
    def plot_pr_curve(self, save_path="models/pr_curve.png"):
        """Plot Precision-Recall curve"""
        logger.debug(f"Plotting PR curve: {save_path}")
        
        precision, recall, _ = precision_recall_curve(
            self.evaluation['y_test'], 
            self.evaluation['y_pred_proba']
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"Precision-Recall Curve (AUC = {self.evaluation['pr_auc']:.3f})")
        plt.grid(True)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved PR curve: {save_path}")
        plt.close()
        
        logger.info(f"PR curve saved: {save_path}")
    
    def save_model(self, model_path="models/xgboost_model.pkl"):
        """Save trained model with logging"""
        logger.info(
            "Saving model",
            extra={
                'training_id': self.training_id,
                'model_path': model_path
            }
        )
        
        try:
            joblib.dump(self.model, model_path)
            
            metadata = {
                'model_type': 'XGBoost',
                'trained_date': datetime.now().isoformat(),
                'training_id': self.training_id,
                'feature_names': self.feature_names,
                'use_smote': self.use_smote,
                'class_weight_ratio': self.class_weight_ratio,
                'performance': {
                    'roc_auc': float(self.evaluation['roc_auc']),
                    'pr_auc': float(self.evaluation['pr_auc'])
                }
            }
            
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n💾 Model saved:")
            print(f"   Model: {model_path}")
            print(f"   Metadata: {metadata_path}")
            
            logger.info(
                "Model saved successfully",
                extra={
                    'training_id': self.training_id,
                    'model_path': model_path,
                    'metadata_path': metadata_path
                }
            )
            
            # Audit log
            audit_logger.log_access(
                user_id='system',
                action='save_model',
                resource='xgboost_model',
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


def train_xgboost_model():
    """Main training pipeline with logging"""
    pipeline_start = time.time()
    
    logger.info("Starting XGBoost training pipeline")
    
    print("\n" + "="*70)
    print(" "*20 + "XGBOOST TRAINING PIPELINE")
    print("="*70)
    
    try:
        print("\n📂 Loading engineered features...")
        df = pd.read_csv("data/processed/features_engineered.csv")
        print(f"   Loaded: {len(df)} samples")
        
        predictor = XGBoostHealthPredictor(use_smote=True, class_weight_ratio=10)
        
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(df)
        
        predictor.train(X_train, y_train, X_val, y_val)
        
        pr_auc = predictor.evaluate(X_test, y_test)
        
        print("\n📊 Generating visualizations...")
        predictor.plot_training_history()
        predictor.plot_confusion_matrix()
        predictor.plot_feature_importance()
        predictor.plot_pr_curve()
        
        predictor.save_model()
        
        total_time = time.time() - pipeline_start
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"\n🎯 Final Performance:")
        print(f"   PR-AUC: {pr_auc:.4f}")
        print(f"   Training time: {total_time:.2f}s")
        print(f"\n📁 Outputs:")
        print(f"   Model: models/xgboost_model.pkl")
        print(f"   Plots: models/*.png")
        
        logger.info(
            "XGBoost training pipeline completed",
            extra={
                'duration_seconds': total_time,
                'pr_auc': float(pr_auc)
            }
        )
        
        return predictor
        
    except Exception as e:
        total_time = time.time() - pipeline_start
        
        logger.error(
            f"XGBoost training pipeline failed: {str(e)}",
            exc_info=True,
            extra={'elapsed_time_seconds': total_time}
        )
        error_logger.log_error(e, context={'operation': 'train_xgboost_model'})
        raise


if __name__ == "__main__":
    predictor = train_xgboost_model()