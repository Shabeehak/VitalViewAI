# evaluate_model.py
"""
Comprehensive Model Evaluation Script with Logging
Generates detailed performance metrics and visualizations

Enhanced Features:
- Structured logging for evaluation process
- Performance metrics tracking
- Audit trail for model evaluation
- Error handling with context
- Progress tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import joblib
import json
import argparse
from datetime import datetime
import warnings
import time
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')

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


class ModelEvaluator:
    """Comprehensive model evaluation with healthcare-specific metrics and logging"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.results = {}
        self.evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(
            f"Initializing ModelEvaluator",
            extra={
                'evaluation_id': self.evaluation_id,
                'model_type': model_type
            }
        )
        
        # Load model and data
        print(f"\n{'='*70}")
        print(f"  LOADING {model_type.upper()} MODEL FOR EVALUATION")
        print(f"{'='*70}\n")
        
        try:
            if model_type == 'xgboost':
                self.model = joblib.load('models/xgboost_model.pkl')
                with open('models/xgboost_model_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
            else:
                import tensorflow as tf
                self.model = tf.keras.models.load_model('models/lstm_model.h5')
                self.scaler = joblib.load('models/lstm_model_scaler.pkl')
                with open('models/lstm_model_metadata.json', 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(
                f"Model loaded successfully",
                extra={
                    'evaluation_id': self.evaluation_id,
                    'model_type': model_type,
                    'trained_date': self.metadata['trained_date'],
                    'training_pr_auc': self.metadata['performance']['pr_auc']
                }
            )
            
            print(f"✅ Model loaded successfully")
            print(f"   Trained: {self.metadata['trained_date']}")
            print(f"   Training Performance: PR-AUC = {self.metadata['performance']['pr_auc']:.4f}")
            
        except Exception as e:
            logger.error(
                f"Failed to load model: {str(e)}",
                exc_info=True,
                extra={'evaluation_id': self.evaluation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'load_model',
                'model_type': model_type
            })
            raise
    
    def load_test_data(self):
        """Load and prepare test data with logging"""
        logger.info(
            f"Loading test data",
            extra={'evaluation_id': self.evaluation_id}
        )
        
        print(f"\n🔄 Loading test data...")
        
        try:
            # Load engineered features
            df = pd.read_csv('data/processed/features_engineered.csv')
            
            # Split same way as training
            train_size = int(0.7 * len(df))
            val_size = int(0.15 * len(df))
            
            test_df = df[train_size + val_size:].reset_index(drop=True)
            
            # Prepare features
            exclude_cols = ['timestamp', 'patient_id', 'device_id', 'activity_state', 'label']
            
            if self.model_type == 'xgboost':
                feature_cols = self.metadata['feature_names']
                X_test = test_df[feature_cols].values
            else:
                feature_cols = [c for c in test_df.columns if c not in exclude_cols]
                X_test = test_df[feature_cols].values
                X_test = self.scaler.transform(X_test)
                
                # Create sequences
                sequence_length = self.metadata['sequence_length']
                X_sequences = []
                y_sequences = []
                
                for i in range(len(X_test) - sequence_length + 1):
                    X_sequences.append(X_test[i:i+sequence_length])
                    y_sequences.append(test_df['label'].iloc[i+sequence_length-1])
                
                X_test = np.array(X_sequences)
                test_df = test_df.iloc[sequence_length-1:].reset_index(drop=True)
            
            y_test = test_df['label'].values
            
            if self.model_type == 'lstm':
                y_test = np.array(y_sequences)
            
            logger.info(
                f"Test data loaded",
                extra={
                    'evaluation_id': self.evaluation_id,
                    'total_samples': len(y_test),
                    'positive_samples': int(np.sum(y_test)),
                    'negative_samples': int(len(y_test) - np.sum(y_test)),
                    'positive_rate': float(np.sum(y_test) / len(y_test))
                }
            )
            
            print(f"✅ Test data loaded: {len(y_test)} samples")
            print(f"   Class distribution: {np.sum(y_test==0)} stable, {np.sum(y_test==1)} deteriorating")
            
            return X_test, y_test, test_df
            
        except Exception as e:
            logger.error(
                f"Failed to load test data: {str(e)}",
                exc_info=True,
                extra={'evaluation_id': self.evaluation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'load_test_data',
                'model_type': self.model_type
            })
            raise
    
    def generate_predictions(self, X_test):
        """Generate predictions and probabilities with timing"""
        logger.info(
            f"Generating predictions",
            extra={
                'evaluation_id': self.evaluation_id,
                'samples': len(X_test)
            }
        )
        
        print(f"\n🔄 Generating predictions...")
        
        start_time = time.time()
        
        try:
            if self.model_type == 'xgboost':
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = self.model.predict(X_test).flatten()
            
            # Use optimal threshold from training
            threshold = self.metadata.get('optimal_threshold', 0.5)
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            inference_time_ms = (time.time() - start_time) * 1000
            avg_time_per_sample = inference_time_ms / len(X_test)
            
            logger.info(
                f"Predictions generated",
                extra={
                    'evaluation_id': self.evaluation_id,
                    'total_time_ms': inference_time_ms,
                    'avg_time_per_sample_ms': avg_time_per_sample,
                    'threshold': threshold
                }
            )
            
            print(f"✅ Predictions generated (threshold: {threshold:.3f})")
            print(f"   Total time: {inference_time_ms:.2f}ms")
            print(f"   Avg per sample: {avg_time_per_sample:.2f}ms")
            
            return y_pred, y_pred_proba, threshold
            
        except Exception as e:
            logger.error(
                f"Prediction generation failed: {str(e)}",
                exc_info=True,
                extra={'evaluation_id': self.evaluation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'generate_predictions',
                'samples': len(X_test)
            })
            raise
    
    def calculate_metrics(self, y_test, y_pred, y_pred_proba):
        """Calculate comprehensive metrics with logging"""
        logger.info(
            f"Calculating metrics",
            extra={'evaluation_id': self.evaluation_id}
        )
        
        print(f"\n📊 Calculating metrics...")
        
        try:
            # Check if we have both classes in test set
            unique_classes = np.unique(y_test)
            if len(unique_classes) < 2:
                logger.warning(
                    f"Test set only contains single class",
                    extra={
                        'evaluation_id': self.evaluation_id,
                        'class': int(unique_classes[0])
                    }
                )
                print(f"\n⚠️ WARNING: Test set only contains class {unique_classes[0]}")
                print(f"   Cannot calculate meaningful metrics with single class")
                return None
            
            # Basic metrics
            cm = confusion_matrix(y_test, y_pred)
            
            # Handle case where confusion matrix is not 2x2
            if cm.shape != (2, 2):
                logger.warning(
                    f"Unexpected confusion matrix shape",
                    extra={
                        'evaluation_id': self.evaluation_id,
                        'shape': cm.shape
                    }
                )
                cm_padded = np.zeros((2, 2), dtype=int)
                cm_padded[:cm.shape[0], :cm.shape[1]] = cm
                tn, fp, fn, tp = cm_padded.ravel()
            else:
                tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # AUC metrics
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                logger.warning("Cannot calculate ROC-AUC")
                roc_auc = np.nan
            
            try:
                pr_auc = average_precision_score(y_test, y_pred_proba)
            except ValueError:
                logger.warning("Cannot calculate PR-AUC")
                pr_auc = np.nan
            
            # Healthcare-specific metrics
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics = {
                'confusion_matrix': {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)},
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'npv': float(npv),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
                'false_negative_rate': float(false_negative_rate),
                'false_positive_rate': float(false_positive_rate),
                'total_samples': int(len(y_test)),
                'positive_samples': int(np.sum(y_test)),
                'negative_samples': int(len(y_test) - np.sum(y_test))
            }
            
            self.results = metrics
            
            logger.info(
                f"Metrics calculated",
                extra={
                    'evaluation_id': self.evaluation_id,
                    'pr_auc': pr_auc,
                    'roc_auc': roc_auc,
                    'recall': recall,
                    'precision': precision,
                    'f1_score': f1
                }
            )
            
            # Log audit trail
            audit_logger.log_access(
                user_id='system',
                action='model_evaluation',
                resource=self.model_type,
                success=True,
                details={
                    'evaluation_id': self.evaluation_id,
                    'pr_auc': pr_auc,
                    'samples': len(y_test)
                }
            )
            
            print(f"✅ Metrics calculated")
            
            return metrics
            
        except Exception as e:
            logger.error(
                f"Metrics calculation failed: {str(e)}",
                exc_info=True,
                extra={'evaluation_id': self.evaluation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'calculate_metrics'
            })
            raise
    
    def print_metrics_report(self, metrics):
        """Print detailed metrics report"""
        print(f"\n{'='*70}")
        print(f"  EVALUATION RESULTS - {self.model_type.upper()}")
        print(f"{'='*70}\n")
        
        logger.info(
            f"Generating metrics report",
            extra={'evaluation_id': self.evaluation_id}
        )
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        print(f"📊 Confusion Matrix:")
        print(f"   ┌─────────────────┬──────────────┐")
        print(f"   │                 │   Predicted  │")
        print(f"   │                 ├──────┬───────┤")
        print(f"   │                 │  No  │  Yes  │")
        print(f"   ├─────────────────┼──────┼───────┤")
        print(f"   │ Actual │   No   │ {cm['TN']:4d} │ {cm['FP']:4d}  │")
        print(f"   │        │  Yes   │ {cm['FN']:4d} │ {cm['TP']:4d}  │")
        print(f"   └─────────────────┴──────┴───────┘")
        
        # Key Metrics
        print(f"\n🎯 Classification Metrics:")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f}")
        print(f"   F1-Score:   {metrics['f1_score']:.4f}")
        print(f"   Specificity:{metrics['specificity']:.4f}")
        
        # AUC Metrics
        print(f"\n📈 AUC Metrics:")
        print(f"   ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:     {metrics['pr_auc']:.4f} ⭐")
        
        # Healthcare-Critical Metrics
        print(f"\n🏥 Healthcare-Critical Analysis:")
        print(f"   False Negative Rate: {metrics['false_negative_rate']:.4f} ⚠️")
        print(f"   False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"   False Negatives:     {cm['FN']:4d} ⚠️ CRITICAL")
        
        # Clinical Interpretation
        print(f"\n💡 Clinical Interpretation:")
        if metrics['recall'] >= 0.95:
            print(f"   ✅ EXCELLENT: Catching >95% of deterioration events")
        elif metrics['recall'] >= 0.85:
            print(f"   ✓ GOOD: Catching >85% of deterioration events")
        elif metrics['recall'] >= 0.70:
            print(f"   ⚠️ ACCEPTABLE: Catching >70% of deterioration events")
        else:
            print(f"   ❌ INSUFFICIENT: Missing >30% of deterioration events")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Generate confusion matrix visualization"""
        logger.debug(f"Plotting confusion matrix")
        
        try:
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                        xticklabels=['Stable', 'Deteriorating'],
                        yticklabels=['Stable', 'Deteriorating'],
                        annot_kws={'size': 16, 'weight': 'bold'})
            
            plt.title(f'Confusion Matrix - {self.model_type.upper()}\n', 
                      fontsize=14, fontweight='bold')
            plt.ylabel('Actual', fontsize=12, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            save_path = f'models/evaluation_{self.model_type}_confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved: {save_path}")
            print(f"   💾 Saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'plot_confusion_matrix'})
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Generate ROC curve"""
        logger.debug(f"Plotting ROC curve")
        
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, 
                    label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
            plt.title(f'ROC Curve - {self.model_type.upper()}\n', 
                      fontsize=14, fontweight='bold')
            plt.legend(loc='lower right', fontsize=11)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            save_path = f'models/evaluation_{self.model_type}_roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"ROC curve saved: {save_path}")
            print(f"   💾 Saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'plot_roc_curve'})
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """Generate Precision-Recall curve"""
        logger.debug(f"Plotting PR curve")
        
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, color='#A23B72', linewidth=2.5,
                    label=f'PR Curve (AUC = {pr_auc:.4f})')
            
            baseline = np.sum(y_test) / len(y_test)
            plt.plot([0, 1], [baseline, baseline], 'k--', linewidth=1.5,
                    label=f'Random Classifier (Baseline = {baseline:.3f})')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12, fontweight='bold')
            plt.ylabel('Precision', fontsize=12, fontweight='bold')
            plt.title(f'Precision-Recall Curve - {self.model_type.upper()}\n', 
                      fontsize=14, fontweight='bold')
            plt.legend(loc='lower left', fontsize=11)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            save_path = f'models/evaluation_{self.model_type}_pr_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PR curve saved: {save_path}")
            print(f"   💾 Saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to plot PR curve: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'plot_pr_curve'})
    
    def save_evaluation_report(self, metrics):
        """Save comprehensive evaluation report"""
        logger.info(f"Saving evaluation report")
        
        try:
            report = {
                'evaluation_id': self.evaluation_id,
                'model_type': self.model_type,
                'evaluation_date': datetime.now().isoformat(),
                'metrics': metrics,
                'visualizations': [
                    f'evaluation_{self.model_type}_confusion_matrix.png',
                    f'evaluation_{self.model_type}_roc_curve.png',
                    f'evaluation_{self.model_type}_pr_curve.png'
                ]
            }
            
            output_path = f'models/evaluation_{self.model_type}_report.json'
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved: {output_path}")
            print(f"\n💾 Saved evaluation report: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}", exc_info=True)
            error_logger.log_error(e, context={'operation': 'save_evaluation_report'})
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline with logging"""
        start_time = time.time()
        
        logger.info(
            f"Starting complete evaluation",
            extra={'evaluation_id': self.evaluation_id}
        )
        
        print(f"\n{'='*70}")
        print(f"  STARTING COMPREHENSIVE EVALUATION")
        print(f"  Evaluation ID: {self.evaluation_id}")
        print(f"{'='*70}")
        
        try:
            # Load data
            X_test, y_test, test_df = self.load_test_data()
            
            # Generate predictions
            y_pred, y_pred_proba, threshold = self.generate_predictions(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            if metrics is None:
                return None
            
            # Print report
            self.print_metrics_report(metrics)
            
            # Generate visualizations
            print(f"\n📊 Generating visualizations...")
            self.plot_confusion_matrix(y_test, y_pred)
            self.plot_roc_curve(y_test, y_pred_proba)
            self.plot_precision_recall_curve(y_test, y_pred_proba)
            
            # Save report
            self.save_evaluation_report(metrics)
            
            total_time = time.time() - start_time
            
            logger.info(
                f"Evaluation completed",
                extra={
                    'evaluation_id': self.evaluation_id,
                    'total_time_seconds': total_time,
                    'pr_auc': metrics['pr_auc']
                }
            )
            
            print(f"\n{'='*70}")
            print(f"  ✅ EVALUATION COMPLETE!")
            print(f"  Total time: {total_time:.2f} seconds")
            print(f"{'='*70}\n")
            
            return metrics
            
        except Exception as e:
            logger.error(
                f"Evaluation failed: {str(e)}",
                exc_info=True,
                extra={'evaluation_id': self.evaluation_id}
            )
            error_logger.log_error(e, context={
                'operation': 'run_complete_evaluation',
                'evaluation_id': self.evaluation_id
            })
            raise


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description='Evaluate ML models')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lstm', 'both'],
                       help='Which model to evaluate')
    
    args = parser.parse_args()
    
    logger.info(f"Starting model evaluation", extra={'model': args.model})
    
    if args.model == 'both':
        # Evaluate both models
        xgb_evaluator = ModelEvaluator('xgboost')
        xgb_evaluator.run_complete_evaluation()
        
        print("\n" + "="*70 + "\n")
        
        lstm_evaluator = ModelEvaluator('lstm')
        lstm_evaluator.run_complete_evaluation()
    else:
        evaluator = ModelEvaluator(args.model)
        evaluator.run_complete_evaluation()
    
    print("\n✅ Evaluation complete! Check models/ and logs/ directories.")
    logger.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()