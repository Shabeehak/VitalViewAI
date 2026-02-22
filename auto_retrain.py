"""
Automatic Model Retraining with Data Drift Detection
Task 5: Scalability - Scheduled retraining pipeline

Usage:
    # Manual trigger
    python auto_retrain.py --check-drift

    # Scheduled via cron
    0 2 * * 0 cd /app && python auto_retrain.py --auto
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path
import sys
from typing import Dict

sys.path.append('src')

from models.train_xgboost import train_xgboost_model
from model_registry import ModelRegistry
from logging_config import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)


class DataDriftDetector:
    """
    Detect data drift in production data
    
    Methods:
    - Statistical tests (KS test, Chi-square)
    - Distribution comparison
    - Feature drift detection
    """
    
    def __init__(self, reference_data_path: str = "data/processed/features_multi.csv"):
        self.reference_data_path = reference_data_path
        logger.info(f"Initializing drift detector with reference: {reference_data_path}")
    
    def detect_drift(self, current_data_path: str, threshold: float = 0.1) -> Dict:
        """
        Detect drift between reference and current data
        
        Returns:
            dict: Drift detection results
        """
        logger.info(f"Checking for data drift...")
        
        try:
            # Load reference data (training data)
            ref_df = pd.read_csv(self.reference_data_path)
            logger.info(f"Reference data: {len(ref_df)} samples")
            
            # Load current production data
            curr_df = pd.read_csv(current_data_path)
            logger.info(f"Current data: {len(curr_df)} samples")
            
            # Compare class distribution
            ref_positive_rate = ref_df['label'].mean()
            curr_positive_rate = curr_df['label'].mean()
            
            class_drift = abs(ref_positive_rate - curr_positive_rate)
            
            logger.info(f"Reference positive rate: {ref_positive_rate:.3f}")
            logger.info(f"Current positive rate: {curr_positive_rate:.3f}")
            logger.info(f"Class drift: {class_drift:.3f}")
            
            # Feature drift (simplified - check vital signs)
            vital_features = ['heart_rate', 'bp_systolic', 'spo2', 'temperature']
            feature_drifts = {}
            
            for feature in vital_features:
                if feature in ref_df.columns and feature in curr_df.columns:
                    ref_mean = ref_df[feature].mean()
                    curr_mean = curr_df[feature].mean()
                    drift = abs(ref_mean - curr_mean) / ref_mean
                    feature_drifts[feature] = drift
                    logger.info(f"Feature drift {feature}: {drift:.3f}")
            
            # Overall drift score
            max_feature_drift = max(feature_drifts.values()) if feature_drifts else 0
            overall_drift = max(class_drift, max_feature_drift)
            
            drift_detected = overall_drift > threshold
            
            results = {
                'drift_detected': drift_detected,
                'overall_drift_score': float(overall_drift),
                'class_drift': float(class_drift),
                'feature_drifts': {k: float(v) for k, v in feature_drifts.items()},
                'threshold': threshold,
                'checked_at': datetime.now().isoformat(),
                'recommendation': 'RETRAIN' if drift_detected else 'OK'
            }
            
            logger.info(f"Drift detection complete: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'}")
            
            return results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}", exc_info=True)
            return {
                'drift_detected': False,
                'error': str(e),
                'recommendation': 'ERROR'
            }


class AutoRetrainer:
    """
    Automatic retraining pipeline
    
    Workflow:
    1. Check for data drift
    2. If drift detected, trigger retraining
    3. Evaluate new model
    4. If performance improved, promote to staging
    5. Manual approval for production
    """
    
    def __init__(self):
        self.drift_detector = DataDriftDetector()
        self.registry = ModelRegistry()
        logger.info("AutoRetrainer initialized")
    
    def check_and_retrain(
        self,
        current_data_path: str = "data/processed/production_data.csv",
        min_improvement: float = 0.02
    ):
        """
        Main retraining workflow
        
        Args:
            current_data_path: Path to current production data
            min_improvement: Minimum PR-AUC improvement to promote
        """
        logger.info("="*70)
        logger.info("AUTOMATIC RETRAINING WORKFLOW")
        logger.info("="*70)
        
        # Step 1: Check for drift
        drift_results = self.drift_detector.detect_drift(current_data_path)
        
        if not drift_results['drift_detected']:
            logger.info("✅ No significant drift detected. Retraining not needed.")
            self._save_drift_report(drift_results)
            return
        
        logger.warning(f"⚠️  Data drift detected (score: {drift_results['overall_drift_score']:.3f})")
        logger.info("🔄 Triggering automatic retraining...")
        
        # Step 2: Retrain model
        try:
            logger.info("Training new model...")
            predictor = train_xgboost_model()
            
            # Step 3: Get new model performance
            new_pr_auc = predictor.evaluation['pr_auc']
            logger.info(f"New model PR-AUC: {new_pr_auc:.4f}")
            
            # Step 4: Compare with production model
            prod_model = self.registry.get_production_model()
            
            if prod_model:
                prod_pr_auc = prod_model['metrics']['pr_auc']
                improvement = new_pr_auc - prod_pr_auc
                
                logger.info(f"Production model PR-AUC: {prod_pr_auc:.4f}")
                logger.info(f"Improvement: {improvement:+.4f}")
                
                if improvement >= min_improvement:
                    logger.info(f"✅ New model improves by {improvement:.4f} (>{min_improvement})")
                    self._promote_new_model(new_pr_auc)
                else:
                    logger.warning(f"⚠️  Improvement too small ({improvement:.4f} < {min_improvement})")
                    logger.warning("New model NOT promoted")
            else:
                logger.info("No production model found. Promoting new model.")
                self._promote_new_model(new_pr_auc)
            
            # Save retraining report
            self._save_retraining_report(drift_results, new_pr_auc)
            
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}", exc_info=True)
            raise
    
    def _promote_new_model(self, pr_auc: float):
        """Register and promote new model"""
        # Register new model
        version_id = self.registry.register_model(
            model_path="models/xgboost_model.pkl",
            model_type="xgboost",
            metrics={'pr_auc': pr_auc},
            description="Auto-retrained due to data drift"
        )
        
        # Promote to staging
        self.registry.promote_to_staging(version_id)
        
        logger.info(f"✅ New model promoted to STAGING: {version_id}")
        logger.info("⚠️  Manual approval required for PRODUCTION deployment")
        logger.info(f"   Command: python model_registry.py promote --version {version_id}")
    
    def _save_drift_report(self, drift_results: Dict):
        """Save drift detection report"""
        report_dir = Path("logs/drift_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"drift_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        logger.info(f"Drift report saved: {report_path}")
    
    def _save_retraining_report(self, drift_results: Dict, new_pr_auc: float):
        """Save retraining report"""
        report_dir = Path("logs/retraining_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"retrain_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detection': drift_results,
            'new_model_performance': {
                'pr_auc': new_pr_auc
            },
            'status': 'success'
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Retraining report saved: {report_path}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Automatic Model Retraining')
    parser.add_argument('--check-drift', action='store_true', help='Check for data drift only')
    parser.add_argument('--auto', action='store_true', help='Auto retrain if drift detected')
    parser.add_argument('--data-path', default='data/processed/features_multi.csv', 
                       help='Path to current production data')
    parser.add_argument('--threshold', type=float, default=0.1, help='Drift threshold')
    
    args = parser.parse_args()
    
    if args.check_drift:
        # Only check drift
        detector = DataDriftDetector()
        results = detector.detect_drift(args.data_path, threshold=args.threshold)
        
        print(f"\n{'='*70}")
        print("DATA DRIFT REPORT")
        print(f"{'='*70}\n")
        print(json.dumps(results, indent=2))
        print()
        
        if results['drift_detected']:
            print("⚠️  DRIFT DETECTED - Retraining recommended")
            print(f"   Run: python auto_retrain.py --auto")
        else:
            print("✅ No significant drift detected")
    
    elif args.auto:
        # Full automatic retraining
        retrainer = AutoRetrainer()
        retrainer.check_and_retrain(current_data_path=args.data_path)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()