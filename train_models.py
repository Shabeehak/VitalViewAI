# train_models.py
"""
Master Training Script with Comprehensive Logging
Orchestrates the complete model training pipeline

Enhanced Features:
- Structured logging for all training steps
- Performance metrics tracking
- Progress monitoring
- Error handling with context
- Training audit trail

Usage:
    python train_models.py
"""

import sys
import time
from datetime import datetime

sys.path.append('src')

from features.feature_engineering import engineer_features_for_training
from models.train_xgboost import train_xgboost_model
from models.train_lstm import train_lstm_model
import pandas as pd
import matplotlib.pyplot as plt

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


def compare_models(xgboost_predictor, lstm_predictor):
    """
    Compare performance of both models with logging
    """
    logger.info("Starting model comparison")
    
    print("\n" + "="*70)
    print(" "*20 + "MODEL COMPARISON")
    print("="*70)
    
    try:
        results = pd.DataFrame({
            'Model': ['XGBoost', 'LSTM'],
            'ROC-AUC': [
                xgboost_predictor.evaluation['roc_auc'],
                lstm_predictor.evaluation['roc_auc']
            ],
            'PR-AUC': [
                xgboost_predictor.evaluation['pr_auc'],
                lstm_predictor.evaluation['pr_auc']
            ]
        })
        
        print("\n📊 Performance Comparison:")
        print(results.to_string(index=False))
        
        # Determine best model
        best_model = 'XGBoost' if results.loc[0, 'PR-AUC'] > results.loc[1, 'PR-AUC'] else 'LSTM'
        best_score = max(results['PR-AUC'])
        
        logger.info(
            f"Model comparison completed",
            extra={
                'best_model': best_model,
                'best_score': best_score,
                'xgboost_pr_auc': results.loc[0, 'PR-AUC'],
                'lstm_pr_auc': results.loc[1, 'PR-AUC']
            }
        )
        
        print(f"\n🏆 Best Model: {best_model} (PR-AUC: {best_score:.4f})")
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC-AUC comparison
        axes[0].bar(results['Model'], results['ROC-AUC'], color=['#3498db', '#e74c3c'])
        axes[0].set_ylabel('ROC-AUC')
        axes[0].set_title('ROC-AUC Comparison')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # PR-AUC comparison
        axes[1].bar(results['Model'], results['PR-AUC'], color=['#3498db', '#e74c3c'])
        axes[1].set_ylabel('PR-AUC')
        axes[1].set_title('PR-AUC Comparison (Primary Metric)')
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = 'models/model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved comparison plot: {save_path}")
        plt.close()
        
        logger.info(f"Comparison plot saved: {save_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}", exc_info=True)
        error_logger.log_error(e, context={'operation': 'compare_models'})
        raise


def main():
    """Main training pipeline with comprehensive logging"""
    pipeline_start_time = time.time()
    pipeline_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(
        f"Starting training pipeline",
        extra={'pipeline_id': pipeline_id}
    )
    
    print("\n" + "="*80)
    print(" "*20 + "🏥 COMPLETE MODEL TRAINING PIPELINE 🏥")
    print("="*80)
    print(f"\nPipeline ID: {pipeline_id}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Feature Engineering
        print("\n" + "━"*80)
        print("STEP 1/4: FEATURE ENGINEERING")
        print("━"*80)
        
        logger.info(
            "Starting feature engineering",
            extra={'pipeline_id': pipeline_id, 'step': 1}
        )
        
        step1_start = time.time()
        
        try:
            df_engineered = engineer_features_for_training(
                input_path="data/processed/features_multi.csv",
                output_path="data/processed/features_engineered.csv"
            )
            
            step1_time = time.time() - step1_start
            
            logger.info(
                f"Feature engineering completed",
                extra={
                    'pipeline_id': pipeline_id,
                    'step': 1,
                    'duration_seconds': step1_time,
                    'features_count': len(df_engineered.columns),
                    'samples_count': len(df_engineered)
                }
            )
            
            print(f"\n✅ Feature engineering completed in {step1_time:.2f}s")
            
        except FileNotFoundError:
            error_msg = "features.csv not found"
            logger.error(
                error_msg,
                extra={'pipeline_id': pipeline_id, 'step': 1}
            )
            
            print("\n❌ features.csv not found!")
            print("\n📝 Generate data first:")
            print("   python quick_generate_data.py")
            return
        
        # Step 2: Train XGBoost
        print("\n" + "━"*80)
        print("STEP 2/4: XGBOOST TRAINING")
        print("━"*80)
        
        logger.info(
            "Starting XGBoost training",
            extra={'pipeline_id': pipeline_id, 'step': 2}
        )
        
        step2_start = time.time()
        xgboost_predictor = train_xgboost_model()
        step2_time = time.time() - step2_start
        
        logger.info(
            f"XGBoost training completed",
            extra={
                'pipeline_id': pipeline_id,
                'step': 2,
                'duration_seconds': step2_time,
                'pr_auc': xgboost_predictor.evaluation['pr_auc']
            }
        )
        
        # Log performance metrics
        perf_logger.log_model_training(
            model_type='xgboost',
            training_time_seconds=step2_time,
            samples=len(df_engineered),
            performance_metrics={
                'pr_auc': xgboost_predictor.evaluation['pr_auc'],
                'roc_auc': xgboost_predictor.evaluation['roc_auc']
            }
        )
        
        print(f"\n✅ XGBoost training completed in {step2_time:.2f}s")
        
        # Step 3: Train LSTM
        print("\n" + "━"*80)
        print("STEP 3/4: LSTM TRAINING")
        print("━"*80)
        
        logger.info(
            "Starting LSTM training",
            extra={'pipeline_id': pipeline_id, 'step': 3}
        )
        
        step3_start = time.time()
        lstm_predictor = train_lstm_model()
        step3_time = time.time() - step3_start
        
        logger.info(
            f"LSTM training completed",
            extra={
                'pipeline_id': pipeline_id,
                'step': 3,
                'duration_seconds': step3_time,
                'pr_auc': lstm_predictor.evaluation['pr_auc']
            }
        )
        
        # Log performance metrics
        perf_logger.log_model_training(
            model_type='lstm',
            training_time_seconds=step3_time,
            samples=len(df_engineered),
            performance_metrics={
                'pr_auc': lstm_predictor.evaluation['pr_auc'],
                'roc_auc': lstm_predictor.evaluation['roc_auc']
            }
        )
        
        print(f"\n✅ LSTM training completed in {step3_time:.2f}s")
        
        # Step 4: Compare Models
        print("\n" + "━"*80)
        print("STEP 4/4: MODEL COMPARISON")
        print("━"*80)
        
        logger.info(
            "Starting model comparison",
            extra={'pipeline_id': pipeline_id, 'step': 4}
        )
        
        step4_start = time.time()
        comparison = compare_models(xgboost_predictor, lstm_predictor)
        step4_time = time.time() - step4_start
        
        logger.info(
            f"Model comparison completed",
            extra={
                'pipeline_id': pipeline_id,
                'step': 4,
                'duration_seconds': step4_time
            }
        )
        
        # Calculate total time
        total_time = time.time() - pipeline_start_time
        
        # Log audit trail
        audit_logger.log_access(
            user_id='system',
            action='model_training_pipeline',
            resource='training_pipeline',
            success=True,
            details={
                'pipeline_id': pipeline_id,
                'total_time_seconds': total_time,
                'best_model': 'XGBoost' if comparison.loc[0, 'PR-AUC'] > comparison.loc[1, 'PR-AUC'] else 'LSTM',
                'xgboost_pr_auc': float(xgboost_predictor.evaluation['pr_auc']),
                'lstm_pr_auc': float(lstm_predictor.evaluation['pr_auc'])
            }
        )
        
        # Final Summary
        print("\n" + "="*80)
        print(" "*25 + "✅ PIPELINE COMPLETE! ✅")
        print("="*80)
        
        print(f"\n⏱️  Total Time: {total_time:.2f} seconds")
        print(f"   Step 1 (Feature Engineering): {step1_time:.2f}s")
        print(f"   Step 2 (XGBoost Training): {step2_time:.2f}s")
        print(f"   Step 3 (LSTM Training): {step3_time:.2f}s")
        print(f"   Step 4 (Model Comparison): {step4_time:.2f}s")
        
        print("\n📁 Generated Files:")
        print("\n   Data:")
        print("   └─ data/processed/features_engineered.csv")
        
        print("\n   Models:")
        print("   ├─ models/xgboost_model.pkl")
        print("   ├─ models/xgboost_model_metadata.json")
        print("   ├─ models/lstm_model.h5")
        print("   ├─ models/lstm_model_scaler.pkl")
        print("   └─ models/lstm_model_metadata.json")
        
        print("\n   Visualizations:")
        print("   ├─ models/training_history.png")
        print("   ├─ models/confusion_matrix.png")
        print("   ├─ models/feature_importance.png")
        print("   ├─ models/pr_curve.png")
        print("   ├─ models/lstm_training_history.png")
        print("   └─ models/model_comparison.png")
        
        print("\n   Logs:")
        print("   ├─ logs/application.log")
        print("   ├─ logs/errors.log")
        print("   ├─ logs/performance.log")
        print("   └─ logs/audit.log")
        
        print("\n🎯 Next Steps:")
        print("   1. Test predictor: python predictor.py")
        print("   2. Evaluate models: python evaluate_model.py")
        print("   3. Review logs: check logs/ directory")
        print("   4. Start API server: python streaming_api_server.py")
        
        print("\n Key Achievements:")
        print("   ✓ Engineered 100+ features from raw time-series data")
        print("   ✓ Trained both gradient boosting (XGBoost) and deep learning (LSTM)")
        print("   ✓ Handled severe class imbalance with SMOTE + class weights")
        print("   ✓ Used PR-AUC as primary metric (better for imbalanced data)")
        print("   ✓ Implemented comprehensive logging and monitoring")
        print("   ✓ Achieved {:.1f}% PR-AUC (primary metric)".format(
            max(comparison['PR-AUC']) * 100
        ))
        
        print("\n✅ Training pipeline finished successfully!")
        print(f"📝 Check logs/ directory for detailed execution logs")
        
        logger.info(
            f"Training pipeline completed successfully",
            extra={
                'pipeline_id': pipeline_id,
                'total_time_seconds': total_time,
                'status': 'success'
            }
        )
        
    except Exception as e:
        total_time = time.time() - pipeline_start_time
        
        logger.error(
            f"Training pipeline failed: {str(e)}",
            exc_info=True,
            extra={
                'pipeline_id': pipeline_id,
                'elapsed_time_seconds': total_time
            }
        )
        
        error_logger.log_error(e, context={
            'operation': 'training_pipeline',
            'pipeline_id': pipeline_id
        })
        
        # Log failed audit trail
        audit_logger.log_access(
            user_id='system',
            action='model_training_pipeline',
            resource='training_pipeline',
            success=False,
            details={
                'pipeline_id': pipeline_id,
                'error': str(e),
                'elapsed_time_seconds': total_time
            }
        )
        
        print(f"\n❌ Training pipeline failed!")
        print(f"   Error: {str(e)}")
        print(f"   Check logs/errors.log for details")
        
        raise


if __name__ == "__main__":
    main()